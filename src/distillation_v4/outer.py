from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import ArtifactStore
from .fingerprints import mapping_fingerprint
from .training import _metrics


class Student(torch.nn.Module):
    def __init__(self, width: int, hidden: int = 32):
        super().__init__(); self.encoder=torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.ReLU()); self.head=torch.nn.Linear(hidden,1)
    def forward(self,x):
        representation=self.encoder(x); return self.head(representation).squeeze(1),representation


def _pipeline(features: list[str]) -> Pipeline:
    return Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())])


def fit_outer_route(
    *,
    route: str,
    frame: pd.DataFrame,
    outer_test_year: int,
    seed: int,
    output: Path,
    route_fingerprint: str,
    max_epochs: int = 30,
    execution_namespace: str = "historical_v4",
) -> dict[str, Any]:
    if execution_namespace != "historical_v4":
        raise RuntimeError(
            "OUTER_TEACHER_REFIT_FORBIDDEN_IN_REPAIR_NAMESPACE"
        )

    if route not in {"supervised","fine_tune","prediction_kd","representation_kd","combined_kd","missing_aware"}:
        raise ValueError(f"UNKNOWN_OUTER_ROUTE:{route}")
    calibration_year=outer_test_year-1
    train=frame.loc[frame.year<calibration_year].copy(); calibration=frame.loc[frame.year.eq(calibration_year)].copy(); test=frame.loc[frame.year.eq(outer_test_year)].copy()
    if train.empty or calibration.empty or test.empty: raise ValueError("OUTER_PARTITION_EMPTY")
    weather=[c for c in frame if c.startswith("weather_")]; soil=[c for c in frame if c.startswith("soil_") and not c.endswith("__missing")]
    use_soil=route=="missing_aware"; features=weather+soil if use_soil else weather
    preprocess=_pipeline(features); xtr=torch.tensor(preprocess.fit_transform(train[features]),dtype=torch.float32); xca=torch.tensor(preprocess.transform(calibration[features]),dtype=torch.float32); xte=torch.tensor(preprocess.transform(test[features]),dtype=torch.float32)
    torch.manual_seed(seed); student=Student(xtr.shape[1]); optimizer=torch.optim.AdamW(student.parameters(),lr=1e-3,weight_decay=1e-4); ytr=torch.tensor(train.target_yield.to_numpy(),dtype=torch.float32)
    teacher_prediction=None; teacher_representation=None; teacher_hash=None
    if route in {"prediction_kd","combined_kd"}:
        tfeatures=weather+soil; teacher=Pipeline([("preprocess",ColumnTransformer([("numeric",_pipeline(tfeatures),tfeatures)])),("model",HistGradientBoostingRegressor(max_iter=160,learning_rate=.05,max_leaf_nodes=31,l2_regularization=.1,random_state=seed,early_stopping=False))]); teacher.fit(train[tfeatures],train.target_yield); teacher_prediction=torch.tensor(teacher.predict(train[tfeatures]),dtype=torch.float32); output.mkdir(parents=True,exist_ok=True); joblib.dump(teacher,output/"prediction_teacher.joblib"); teacher_hash=hashlib.sha256((output/"prediction_teacher.joblib").read_bytes()).hexdigest()
    if route in {"representation_kd","combined_kd"}:
        tfeatures=weather+soil; tp=_pipeline(tfeatures); tx=torch.tensor(tp.fit_transform(train[tfeatures]),dtype=torch.float32); teacher_net=Student(tx.shape[1]); topt=torch.optim.AdamW(teacher_net.parameters(),lr=1e-3,weight_decay=1e-4)
        for _ in range(max_epochs):
            topt.zero_grad(); pred,_=teacher_net(tx); loss=torch.nn.functional.huber_loss(pred,ytr); loss.backward(); topt.step()
        teacher_net.eval();
        with torch.no_grad(): _,teacher_representation=teacher_net(tx)
        output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":teacher_net.state_dict(),"features":tfeatures},output/"representation_teacher.pt"); joblib.dump(tp,output/"representation_teacher_preprocessor.joblib"); teacher_hash=hashlib.sha256((output/"representation_teacher.pt").read_bytes()).hexdigest()
    if route=="fine_tune" and soil:
        sp=_pipeline(soil); target_soil=torch.tensor(sp.fit_transform(train[soil]),dtype=torch.float32); projection=torch.nn.Linear(32,target_soil.shape[1]); preopt=torch.optim.AdamW(list(student.encoder.parameters())+list(projection.parameters()),lr=1e-3)
        for _ in range(max_epochs):
            preopt.zero_grad(); representation=student.encoder(xtr); loss=torch.nn.functional.mse_loss(projection(representation),target_soil); loss.backward(); preopt.step()
        output.mkdir(parents=True,exist_ok=True); torch.save({"encoder":student.encoder.state_dict(),"soil_projection":projection.state_dict()},output/"multimodal_pretraining.pt")
    trajectory=[]; start=time.perf_counter()
    for epoch in range(max_epochs):
        optimizer.zero_grad(); pred,rep=student(xtr); supervised=torch.nn.functional.huber_loss(pred,ytr); pkd=torch.tensor(0.0); rkd=torch.tensor(0.0)
        if teacher_prediction is not None: pkd=torch.nn.functional.mse_loss(pred,teacher_prediction)
        if teacher_representation is not None: rkd=torch.nn.functional.mse_loss(rep,teacher_representation)
        loss=supervised+pkd+rkd; loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); optimizer.step(); trajectory.append({"epoch":epoch,"supervised":float(supervised.detach()),"prediction_imitation":float(pkd.detach()),"representation_alignment":float(rkd.detach()),"total":float(loss.detach())})
    fit_time=time.perf_counter()-start; start=time.perf_counter(); student.eval()
    with torch.no_grad(): pcal,_=student(xca); ptest,_=student(xte)
    inference=time.perf_counter()-start; output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":student.state_dict(),"features":features,"route":route},output/"model.pt"); joblib.dump(preprocess,output/"preprocessor.joblib"); (output/"loss_ledger.json").write_text(json.dumps(trajectory,indent=2)+"\n")
    pd.DataFrame({"sample_id":calibration.sample_id.astype(str),"y_true":calibration.target_yield,"y_pred":pcal.numpy()}).to_csv(output/"calibration_predictions.csv",index=False)
    metrics=_metrics(test.target_yield.to_numpy(),ptest.numpy(),test["group"] if "group" in test else test.sample_id)
    store=ArtifactStore(output); store.write_predictions(pd.DataFrame({"sample_id":test.sample_id.astype(str),"y_true":test.target_yield,"y_pred":ptest.numpy()})); store.write_fold_ids(pd.DataFrame({"sample_id":test.sample_id.astype(str),"split":"test"})); store.write_metrics(metrics)
    runtime={"wall_time_seconds":fit_time+inference,"fit_time_seconds":fit_time,"inference_time_seconds":inference,"optimizer_steps":max_epochs,"device":"cpu","peak_memory_bytes":None,"runtime_evidence_status":"RUNTIME_EVIDENCE_INCOMPLETE"}
    model_ledger={"route":route,"parameters":sum(p.numel() for p in student.parameters()),"teacher_checkpoint_sha256":teacher_hash}
    fps={"code_tree":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"config":mapping_fingerprint({"route":route,"epochs":max_epochs}),"data":hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),"view":mapping_fingerprint({"columns":list(frame),"rows":len(frame)}),"split":mapping_fingerprint({"outer_test_year":outer_test_year,"calibration_year":calibration_year}),"feature":mapping_fingerprint(features),"model":mapping_fingerprint(model_ledger),"runtime":mapping_fingerprint({"python":platform.python_version(),"torch":torch.__version__})}
    store.write_manifest({"route":route,"fold":f"test_{outer_test_year}","seed":seed,"outer_test_role":"FINAL_ESTIMATION_ONLY","outer_test_may_enable_downstream_jobs":False,"route_fingerprint_before_outer_test":route_fingerprint,"fingerprints":fps,"runtime":runtime,"model":model_ledger}); store.accept(); store.complete()
    return {"engineering_status":"ENGINEERING_ACCEPTED","metrics":metrics,"runtime":runtime}
