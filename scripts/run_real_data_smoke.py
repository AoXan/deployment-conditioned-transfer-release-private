"""Execute real CY-Bench preprocessing, recovered training and fresh diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.distillation_v4.universal_weather_v2.adapters import build_cybench_view
from src.distillation_v4.australian.execution import execute_job
from src.distillation_v4.australian.splits import build_regional_splits


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--component', choices=['supervised','prediction_kd','representation_kd','combined_kd','reference'], action='append')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in a.raw_root.glob('*_wheat_AU.csv')}
    if len(sources)!=6:raise FileNotFoundError('All six authentic AU raw tables are required')
    dump(a.output/'protocol.json',{'scope':'Real-data execution validation; regional adapted protocol, not full cross-crop primary parity','seeds':[101],'source_sha256':sources,'historical_results_used':False,'synthetic_inputs_used':False})
    path=a.output/'processed/CY-Bench_wheat_AU.csv.gz';path.parent.mkdir()
    metadata=build_cybench_view(dataset_id='CY-Bench_wheat_AU',directory=a.raw_root,crop='wheat',country='AU',output_path=path)
    dump(a.output/'preprocessing.json',metadata)
    frame=pd.read_csv(path)
    numeric=set(frame.select_dtypes(include='number').columns)
    features={'deployable':[c for c in frame if c.startswith('weather_') and c in numeric], 'soil':[c for c in frame if c.startswith('privileged_soil_') and c in numeric]}
    if not features['deployable']:raise ValueError('Weather feature contract is empty')
    features['weather']=features['deployable']
    splits=build_regional_splits(frame)
    if not splits:raise ValueError('No valid actual-data split')
    split=splits[0]
    dump(a.output/'split.json',split)
    results=[]
    for component in a.component or ['supervised','prediction_kd','representation_kd','combined_kd','reference']:
        job={'component_id':component,'seed':101,'job_id':'real_data_smoke_'+component,'dataset_id':'CY-Bench_wheat_AU','fold':split['fold'],'axis':split['axis'],'planned_protocol':'ENGINEERING_REAL_DATA_ONLY','sample_unit':'ADMIN_REGION_HARVEST_YEAR','target_unit':'tonne_ha','smoke_only':True}
        directory=a.output/'models'/component
        result=execute_job(job,frame,features,split,directory)
        results.append({'component':component,**result['metrics']})
        if component=='supervised':
            from captum.attr import IntegratedGradients
            import shap
            saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
            hidden=saved['student']['encoder.0.weight'].shape[0]
            prep=joblib.load(directory/'preprocessor.joblib')
            names=prep['deployable_features']
            train=frame[frame.sample_id.astype(str).isin(split['train_ids'])]
            test=frame[frame.sample_id.astype(str).isin(split['test_ids'])]
            transform=prep['deployable']
            xx=transform.transform(test[names]).astype('float32')
            background=transform.transform(train[names].iloc[:min(16,len(train))]).astype('float32')
            model=nn.ModuleDict({'encoder':nn.Sequential(nn.Linear(len(names),hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU()),'head':nn.Linear(hidden,1)})
            model.load_state_dict(saved['student']);model.eval()
            forward=lambda x:model['head'](model['encoder'](x))
            tensors=torch.tensor(xx)
            with torch.no_grad():replayed=forward(tensors).view(-1).numpy()
            original=pd.read_csv(directory/'predictions.csv').y_pred.to_numpy()
            if not np.allclose(replayed,original,rtol=1e-5,atol=1e-5):raise AssertionError('Fresh checkpoint/preprocessor replay mismatch')
            sample=tensors[:min(8,len(tensors))]
            baseline=torch.tensor(background.mean(axis=0)).reshape(1,-1)
            attr,delta=IntegratedGradients(forward).attribute(sample,baselines=baseline,n_steps=32,return_convergence_delta=True)
            predict=lambda value:forward(torch.tensor(value,dtype=torch.float32)).detach().numpy().reshape(-1)
            explanation=shap.Explainer(predict,background,algorithm='permutation')(sample.numpy(),max_evals=2*len(names)+1)
            np.savez_compressed(a.output/'diagnostics.npz',ig=attr.detach().numpy(),shap=explanation.values,convergence_delta=delta.detach().numpy())
            pd.DataFrame({'feature':names,'mean_signed_ig':attr.detach().numpy().mean(0),'mean_signed_shap':explanation.values.mean(0)}).to_csv(a.output/'fresh_attribution_summary.csv',index=False)
            dump(a.output/'fresh_checkpoint_lineage.json',{'checkpoint_sha256':hashlib.sha256((directory/'checkpoint.pt').read_bytes()).hexdigest(),'preprocessor_sha256':hashlib.sha256((directory/'preprocessor.joblib').read_bytes()).hexdigest(),'training_source':'current run only','checkpoint_replay_verified':True,'samples':len(sample),'background':len(background),'ig_steps':32,'shap_permutations':'minimum one permutation; smoke scope only'})
    table=pd.DataFrame(results);table.to_csv(a.output/'fresh_model_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax=table.set_index('component')[['mae','rmse']].plot.bar()
    ax.figure.tight_layout();ax.figure.savefig(a.output/'fresh_model_metrics.pdf')
    table.to_latex(a.output/'fresh_model_metrics.tex',index=False)
    dump(a.output/'verification.json',{'status':'PIPELINE_VERIFIED','overall_paper_status':'CONDITIONALLY_REPRODUCIBLE','raw_rows':len(frame),'train':len(split['train_ids']),'validation':len(split['validation_ids']),'test':len(split['test_ids']),'components':table.component.tolist(),'source_scope':'CY-Bench Australian wheat only; no full US-to-AU campaign or accepted numeric parity','diagnostics':'Fresh trained neural model: checkpoint replay, SHAP and IG executed','APSIM':'not executed; licensed runtime and experiment inputs required','full_numeric_reproduced':False})


if __name__=='__main__':
    main()
