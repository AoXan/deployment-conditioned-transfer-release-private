"""Real-data PyTorch smoke for T1/M2 in the isolated test namespace."""
from __future__ import annotations
import hashlib, json, os, random, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from .supplemental_torch import build_modality_model, privileged_student_loss, require_torch

G2F=Path(os.environ.get("AGRITECH_G2F_VIEW", "data/external/g2f/g2f_native_hybrid_env.csv.gz"))

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def _atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
def run_torch_smoke(route:str,out:Path,config:Path,seed:int=101):
 torch,nn=require_torch(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
 df=pd.read_csv(G2F,low_memory=False); df=df[pd.to_numeric(df.target_yield,errors='coerce').notna()].tail(64).copy()
 numeric=[c for c in df.select_dtypes(include='number') if c not in {'target_yield','Year'} and not df[c].isna().all()][:12]
 if len(numeric)<6: raise ValueError('insufficient safe numeric features')
 groups={'environment':numeric[:4],'soil':numeric[4:8],'management':numeric[8:12]}; train=df.iloc[:48]; test=df.iloc[48:]
 modalities={}; test_modalities={}; masks={}; test_masks={}
 for name,cols in groups.items():
  med=train[cols].median(); scale=train[cols].std().replace(0,1); modalities[name]=torch.tensor(((train[cols].fillna(med)-med)/scale).to_numpy(np.float32)); test_modalities[name]=torch.tensor(((test[cols].fillna(med)-med)/scale).to_numpy(np.float32)); masks[name]=torch.ones(len(train)); test_masks[name]=torch.ones(len(test))
 y=torch.tensor(train.target_yield.to_numpy(np.float32)); model=build_modality_model({k:len(v) for k,v in groups.items()},hidden=16); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
 pred,latent,_=model(modalities,masks)
 loss_ledger={}
 if route=='T1': loss=torch.nn.functional.huber_loss(pred,y); loss_ledger={'supervised_huber':float(loss.detach())}
 elif route=='M2':
  teacher=build_modality_model({k:len(v) for k,v in groups.items()},hidden=16); teacher.eval()
  with torch.no_grad(): teacher_pred,teacher_latent,_=teacher(modalities,masks)
  student_masks={k:v.clone() for k,v in masks.items()}; student_masks['soil'][::2]=0; student_pred,student_latent,_=model(modalities,student_masks); recon=nn.Linear(16,sum(map(len,groups.values()))); reconstruction=recon(student_latent); target=torch.cat([modalities[k] for k in groups],dim=1); rmask=torch.ones_like(target); loss,parts=privileged_student_loss(student_prediction=student_pred,target=y,student_latent=student_latent,teacher_prediction=teacher_pred,teacher_latent=teacher_latent,reconstructed=reconstruction,reconstruction_target=target,reconstruction_mask=rmask,weights={'supervised':1.,'prediction_kd':.5,'latent':.5,'reconstruction':.25}); loss_ledger={k:float(v.detach()) for k,v in parts.items()}; opt=torch.optim.Adam(list(model.parameters())+list(recon.parameters()),lr=1e-3)
 else: raise ValueError(route)
 opt.zero_grad(); loss.backward(); opt.step()
 route_out=out/route; route_out.mkdir(parents=True,exist_ok=True); checkpoint=route_out/'checkpoint.pt'; torch.save({'model':model.state_dict(),'seed':seed,'route':route},checkpoint); restored=build_modality_model({k:len(v) for k,v in groups.items()},hidden=16); restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']); restored.eval()
 with torch.no_grad(): test_pred,_,_=restored(test_modalities,test_masks)
 rows=pd.DataFrame({'sample_id':test.sample_id.astype(str),'y_true':test.target_yield.astype(float),'y_pred':test_pred.numpy(),'evidence_status':'SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE'}); rows.to_csv(route_out/'predictions.csv',index=False); pd.DataFrame({'sample_id':df.sample_id.astype(str),'split':['train']*48+['test']*16}).to_csv(route_out/'fold_assignments.csv',index=False)
 fp_inputs={'code':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'config':_sha(config),'data':_sha(G2F),'view':_sha(G2F),'split':hashlib.sha256('\n'.join(sorted(test.sample_id.astype(str))).encode()).hexdigest(),'feature':hashlib.sha256('\n'.join(numeric).encode()).hexdigest()}; fp=hashlib.sha256(json.dumps(fp_inputs,sort_keys=True).encode()).hexdigest(); metrics={'n':len(rows),'mae':float(np.mean(np.abs(rows.y_true-rows.y_pred))),'loss_ledger':loss_ledger}; _atomic(route_out/'metrics.json',metrics); manifest={'route':route,'seed':seed,'device':'cpu','python':__import__('platform').python_version(),'torch':torch.__version__,'fingerprint':fp,'fingerprint_inputs':fp_inputs,'checkpoint_sha256':_sha(checkpoint),'evidence_status':'SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE','forward':True,'backward':True,'optimizer_step':True,'checkpoint_reload':True}; _atomic(route_out/'manifest.json',manifest); _atomic(route_out/'completion_marker.json',{'artifact_status':'COMPLETE_UNVERIFIED','fingerprint':fp}); valid=len(rows)==16 and np.isfinite(rows[['y_true','y_pred']]).all().all() and set(rows.sample_id)==set(test.sample_id.astype(str)); _atomic(route_out/'acceptance_marker.json',{'artifact_status':'ARTIFACT_ACCEPTED' if valid else 'INVALIDATED','fingerprint':fp,'checks':{'ids_equal':valid,'finite':valid,'checkpoint_reload':True}})
 return {'route':route,'status':'PASS' if valid else 'FAIL','n':len(rows),'loss':float(loss.detach()),'output':str(route_out),'mps_available':bool(torch.backends.mps.is_available())}
