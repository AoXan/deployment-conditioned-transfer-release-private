#!/usr/bin/env python3
"""Execute genuine Stage 8 formal adapters on frozen, snapshot-bound views."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.stage8.formal_routes import T1_WINDOWS, deterministic_fraction_ids
from src.stage8.formal_training import feature_columns, make_pipeline, write_artifacts
from src.stage8.protocol_helpers import conditional_quantiles, select_by_validation_mae
from src.stage8.metrics import conformal_metrics, deployable_aurc
from src.stage8.supplemental_routes import finite_sample_quantile, learning_curve_aulc, soft_route, uncertainty_endpoints, weighted_quantile
from src.stage8.reruns import route_is_fully_accepted

G2F=Path(os.environ.get('AGRITECH_G2F_VIEW', ROOT/'data/external/g2f/g2f_native_hybrid_env.csv.gz'))
CYV=Path(os.environ.get('AGRITECH_CYBENCH_VIEW_ROOT', ROOT/'data/external/cybench/views'))

_HASH_CACHE = {}
PROGRESS_PATH = None
ONLY_FOLD = None
ONLY_VARIANT = None
ONLY_PATTERN = None
ONLY_FRACTION = None
def file_hash(path):
    path=Path(path); key=(str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)
    if key not in _HASH_CACHE:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        _HASH_CACHE[key]=h.hexdigest()
    return _HASH_CACHE[key]

def load_g2f():
    df=pd.read_csv(G2F); df=df[pd.to_numeric(df.target_yield,errors='coerce').notna()].copy(); df.Year=pd.to_numeric(df.Year).astype(int); return df
def xy(df):
    cols=feature_columns(df,'target_yield'); return df[cols].replace([np.inf,-np.inf],np.nan),df.target_yield.astype(float),cols
def save(out, test, pred, meta, *, source_path, feature_cols, config_path, extra_predictions=None):
    p=pd.DataFrame({'sample_id':test.sample_id.astype(str),'fold':meta['fold'],'y_true':test.target_yield.astype(float),'y_pred':np.asarray(pred,float),'model':meta['model'],'seed':meta['seed']})
    for name,values in (extra_predictions or {}).items(): p[name]=np.asarray(values)
    folds=pd.DataFrame({'sample_id':test.sample_id.astype(str),'fold':meta['fold'],'split':'test'})
    split_hash=hashlib.sha256('\n'.join(sorted(folds.sample_id)).encode()).hexdigest()
    feature_hash=hashlib.sha256('\n'.join(sorted(map(str,feature_cols))).encode()).hexdigest()
    try: code_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: code_hash=file_hash(Path(__file__))
    sources=[Path(value) for value in source_path] if isinstance(source_path,(list,tuple)) else [Path(source_path)]; source_hash=hashlib.sha256('\n'.join(file_hash(value) for value in sources).encode()).hexdigest()
    inputs={'code':code_hash,'config':file_hash(config_path),'data':source_hash,'view':source_hash,'split':split_hash,'feature':feature_hash}
    metrics=write_artifacts(out,p,meta,expected_ids=set(test.sample_id.astype(str)),fold_assignments=folds,fingerprint_inputs=inputs)
    event={'timestamp':datetime.now(timezone.utc).isoformat(),'event':'artifact_complete_unverified','run_path':str(out),'route':meta.get('route'),'fold':meta.get('fold'),'seed':meta.get('seed'),'variant':meta.get('variant'),'pattern':meta.get('pattern'),'metrics':metrics}
    if PROGRESS_PATH is not None:
     PROGRESS_PATH.parent.mkdir(parents=True,exist_ok=True)
     with PROGRESS_PATH.open('a') as handle: handle.write(json.dumps(event,default=str)+'\n'); handle.flush()
    print(json.dumps(event,default=str),flush=True)
    return metrics
def run_t1(out,seeds,config_path):
    df=load_g2f(); results=[]
    for year,w in T1_WINDOWS.items():
      pre=df[df.Year.isin(w['pretrain'])]; adapt=df[df.Year.eq(w['adapt'])]; val=df[df.Year.eq(w['validation'])]; test=df[df.Year.eq(year)]
      for seed in seeds:
       fractions=deterministic_fraction_ids(adapt.sample_id,[.01,.05,.1,.2],seed)
       for frac,ids in fractions.items():
        few=adapt[adapt.sample_id.isin(ids)]
        for variant,train in [('scratch',few),('no_adaptation',pre),('pretrain_adapt',pd.concat([pre,few]))]:
         if ONLY_FOLD and ONLY_FOLD != f'test_{year}': continue
         if ONLY_VARIANT and ONLY_VARIANT != variant: continue
         if ONLY_FRACTION is not None and not np.isclose(ONLY_FRACTION,frac): continue
         X,y,cols=xy(train); candidates={}; fitted={}
         for alpha in (1e-5,1e-4,1e-3):
          candidate=make_pipeline(X,'huber',seed,model_params={'alpha':alpha}); candidate.fit(X,y); candidates[str(alpha)]=candidate.predict(val[cols]); fitted[str(alpha)]=candidate
         selected=select_by_validation_mae(val.target_yield,candidates); model=fitted[selected]; pred=model.predict(test[cols])
         meta={'route':'T1','dataset':'g2f_native','fold':f'test_{year}','test_year':year,'fraction':frac,'variant':variant,'model':'modality_preprocessed_huber','seed':seed,'pretrain_years':list(w['pretrain']),'adapt_year':w['adapt'],'validation_year':w['validation'],'selection_source':'validation_only','selected_alpha':float(selected),'adaptation_ids_hash':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),'loss':'huber','deployment_modalities_only':True}
         metrics=save(out/f'test_{year}'/f'fraction_{frac}'/f'seed_{seed}'/variant,test,pred,meta,source_path=G2F,feature_cols=cols,config_path=config_path); results.append({**meta,**metrics})
    summary=pd.DataFrame(results); summary.to_csv(out/'summary.csv',index=False)
    curves=[]
    for keys,g in summary.groupby(['fold','seed','variant']):
      g=g.sort_values('fraction'); a=learning_curve_aulc(g.fraction,g.mae); curves.append({'fold':keys[0],'seed':keys[1],'variant':keys[2],**a})
    pd.DataFrame(curves).to_csv(out/'learning_curves_aulc.csv',index=False)
def run_m1(out,seeds,config_path):
    df=load_g2f(); blocks={'weather':[c for c in df if c.startswith('weather_')],'soil':[c for c in df if c.startswith('soil_')],'ec':[c for c in df if c.startswith('ec_')]}; results=[]
    for year in (2021,2022,2023):
      train=df[df.Year.lt(year)]; test=df[df.Year.eq(year)]
      for seed in seeds:
       for pattern,missing in [('natural',[]),('synthetic_no_soil',['soil']),('synthetic_no_weather',['weather'])]:
        blocked={c for b in missing for c in blocks[b]}; cols=[c for c in feature_columns(train,'target_yield') if c not in blocked]
        for variant in ('complete_case','imputation','missing_indicators','late_fusion'):
         if variant=='complete_case':
          complete=train[cols].notna().all(axis=1); fit=train.loc[complete]
          if len(fit)<20: continue
          model=make_pipeline(fit[cols],'ridge',seed); model.fit(fit[cols],fit.target_yield); pred=model.predict(test[cols]); model_name='ridge_complete_case'
         elif variant=='late_fusion':
          modality_cols={name:[c for c in cols if c in bcols] for name,bcols in blocks.items()}; modality_cols={k:v for k,v in modality_cols.items() if v}
          predictions=[]; availability=[]
          for name,bcols in modality_cols.items():
           model=make_pipeline(train[bcols],'rf',seed); model.fit(train[bcols],train.target_yield); predictions.append(model.predict(test[bcols])); availability.append(test[bcols].notna().any(axis=1).to_numpy(float))
          matrix=np.column_stack(predictions); weights=np.column_stack(availability); weights=np.where(weights.sum(1,keepdims=True)>0,weights,1.0); pred=(matrix*weights).sum(1)/weights.sum(1); model_name='modality_specific_rf_late_fusion'
         else:
          engineered=train[cols].copy(); engineered_test=test[cols].copy()
          if variant=='missing_indicators':
           for c in cols:
            engineered[f'{c}__missing']=engineered[c].isna().astype(int); engineered_test[f'{c}__missing']=engineered_test[c].isna().astype(int)
          model=make_pipeline(engineered,'ridge',seed); model.fit(engineered,train.target_yield); pred=model.predict(engineered_test); model_name='ridge'
         meta={'route':'M1','dataset':'g2f_native','fold':f'test_{year}','pattern':pattern,'missingness_type':'natural' if pattern=='natural' else 'synthetic','variant':variant,'model':model_name,'seed':seed}; results.append({**meta,**save(out/pattern/f'test_{year}'/f'seed_{seed}'/variant,test,pred,meta,source_path=G2F,feature_cols=cols,config_path=config_path)})
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
def run_t2(out,seeds,config_path):
    us=pd.read_csv(CYV/'maize_US_safe.csv.gz'); br=pd.read_csv(CYV/'maize_BR_safe.csv.gz'); common=sorted((set(us.columns)&set(br.columns))-{'target_yield','sample_id','adm_id','season_year'})
    source=us[us.season_year.le(2020)]; adapt=br[br.season_year.le(2020)]; val=br[br.season_year.eq(2021)]; test=br[br.season_year.eq(2022)]; results=[]
    if not len(test): raise ValueError('T2 outer test 2022 is empty')
    for seed in seeds:
      for variant in ('source_only','target_scratch','pooled','harmonised_transfer'):
       if variant=='source_only': train=source.copy()
       elif variant=='target_scratch': train=adapt.copy()
       else: train=pd.concat([source,adapt],ignore_index=True)
       target=train.target_yield.astype(float)
       if variant=='harmonised_transfer':
        stats=train.groupby(train.get('country',pd.Series(['domain']*len(train)))).target_yield.agg(['mean','std']) if 'country' in train else None
        # Domain-wise train-only standardisation; predictions are restored to target scale.
        sm,ss=source.target_yield.mean(),source.target_yield.std(); tm,ts=adapt.target_yield.mean(),adapt.target_yield.std(); target=pd.concat([(source.target_yield-sm)/ss,(adapt.target_yield-tm)/ts],ignore_index=True)
       candidate_predictions={}; fitted={}
       for leaf in (2,5,10):
        candidate=make_pipeline(train[common],'rf',seed,model_params={'min_samples_leaf':leaf}); candidate.fit(train[common],target); val_pred=candidate.predict(val[common]);
        if variant=='harmonised_transfer': val_pred=val_pred*ts+tm
        candidate_predictions[str(leaf)]=val_pred; fitted[str(leaf)]=candidate
       selected=select_by_validation_mae(val.target_yield,candidate_predictions); model=fitted[selected]; pred=model.predict(test[common]);
       if variant=='harmonised_transfer': pred=pred*ts+tm
       meta={'route':'T2','dataset':'cybench_maize_us_to_br','fold':'BR_test_2022','variant':variant,'model':'rf','seed':seed,'source_years':'<=2020','target_adaptation_years':'<=2020','validation_year':2021,'selection_source':'BR_validation_2021_only','selected_min_samples_leaf':int(selected),'feature_intersection':common,'target_unit':'t_ha'}
       results.append({**meta,**save(out/f'seed_{seed}'/variant,test,pred,meta,source_path=[CYV/'maize_US_safe.csv.gz',CYV/'maize_BR_safe.csv.gz'],feature_cols=common,config_path=config_path)})
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
def run_m2(out,seeds,config_path):
    df=load_g2f(); full=feature_columns(df,'target_yield'); patterns={'natural':full,'synthetic_no_soil':[c for c in full if not c.startswith('soil_')],'synthetic_no_weather':[c for c in full if not c.startswith('weather_')]}; results=[]
    for year in (2021,2022,2023):
      train=df[df.Year.lt(year)]; test=df[df.Year.eq(year)]
      for seed in seeds:
       modality_groups=[[c for c in full if c.startswith(prefix)] for prefix in ('weather_','soil_','ec_')]
       modality_groups=[group for group in modality_groups if group]
       complete=np.logical_and.reduce([train[group].notna().any(axis=1).to_numpy() for group in modality_groups]); teacher_train=train.loc[complete]
       if len(teacher_train)<20: raise ValueError(f'M2 complete-case teacher unavailable for test_{year}')
       teacher=make_pipeline(teacher_train[full],'rf',seed); teacher.fit(teacher_train[full],teacher_train.target_yield); pseudo=teacher.predict(train[full])
       for pattern,allowed in patterns.items():
        blended=.5*train.target_yield.to_numpy()+.5*pseudo
        student=make_pipeline(train[allowed],'mlp',seed); student.fit(train[allowed],blended); pred=student.predict(test[allowed]); meta={'route':'M2','dataset':'g2f_native','fold':f'test_{year}','pattern':pattern,'missingness_type':'natural' if pattern=='natural' else 'synthetic','variant':'teacher_student','model':'mlp','seed':seed,'distillation_weight':.5,'supervised_weight':.5,'teacher_training':'fold_local_complete_case','test_labels_used_for_training':False}; results.append({**meta,**save(out/pattern/f'test_{year}'/f'seed_{seed}',test,pred,meta,source_path=G2F,feature_cols=allowed,config_path=config_path)})
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
def run_r1(out,seeds,config_path):
    df=load_g2f(); results=[]
    for year in (2021,2022,2023):
      train=df[df.Year.lt(year)].copy(); test=df[df.Year.eq(year)].copy(); X,y,cols=xy(train); Xt=test[cols]
      for seed in seeds:
       expert_names=('ridge','rf'); oof=np.full((len(train),len(expert_names)),np.nan); years=sorted(train.Year.unique())
       for inner_year in years[2:]:
        fit_mask=train.Year.lt(inner_year); val_mask=train.Year.eq(inner_year)
        for j,name in enumerate(expert_names):
         pipe=make_pipeline(train.loc[fit_mask,cols],name,seed); pipe.fit(train.loc[fit_mask,cols],train.loc[fit_mask,'target_yield']); oof[val_mask,j]=pipe.predict(train.loc[val_mask,cols])
       valid=np.isfinite(oof).all(axis=1); labels=np.argmin(np.abs(oof[valid]-y.to_numpy()[valid,None]),axis=1); router=RandomForestClassifier(n_estimators=150,min_samples_leaf=5,random_state=seed,n_jobs=-1); router.fit(oof[valid],labels)
       testp=[]
       for name in expert_names:
        pipe=make_pipeline(X,name,seed); pipe.fit(X,y); testp.append(pipe.predict(Xt))
       testmat=np.column_stack(testp); probabilities=router.predict_proba(testmat); pred,soft_weights=soft_route(testmat,np.log(np.clip(probabilities,1e-12,1))) ; gate=np.argmax(soft_weights,axis=1)
       recent=train.Year.eq(year-1).to_numpy() & valid; recent_mae=np.mean(np.abs(oof[recent]-y.to_numpy()[recent,None]),axis=0); selected=int(np.argmin(recent_mae)); selected_pred=testmat[:,selected]; ensemble_pred=testmat.mean(1); oracle_idx=np.argmin(np.abs(testmat-test.target_yield.to_numpy()[:,None]),axis=1); oracle_pred=testmat[np.arange(len(test)),oracle_idx]
       runout=out/f'test_{year}'/f'seed_{seed}'; meta={'route':'R1','dataset':'g2f_native','fold':f'test_{year}','model':'soft_oof_regret_router','seed':seed,'experts':list(expert_names),'oof_protocol':'expanding_year_cross_fit','selection_source':'most_recent_inner_validation','validation_selected_expert':expert_names[selected],'oracle_role':'diagnostic_only'}; metrics=save(runout,test,pred,meta,source_path=G2F,feature_cols=cols,config_path=config_path,extra_predictions={'validation_selected_pred':selected_pred,'nested_oof_ensemble_pred':ensemble_pred,'oracle_diagnostic_pred':oracle_pred,'router_expert_index':gate,'router_weight_0':soft_weights[:,0],'router_weight_1':soft_weights[:,1]}); pd.DataFrame({'sample_id':train.loc[valid,'sample_id'],'y_true':y.to_numpy()[valid],'ridge_oof':oof[valid,0],'rf_oof':oof[valid,1]}).to_csv(runout/'oof_predictions.csv',index=False); pd.DataFrame({'sample_id':train.loc[valid,'sample_id'],'regret_label':labels}).to_csv(runout/'regret_labels.csv',index=False); (runout/'expert_registry.json').write_text(json.dumps({'experts':expert_names})); results.append({**meta,**metrics,'validation_selected_mae':float(np.mean(np.abs(test.target_yield-selected_pred))),'ensemble_mae':float(np.mean(np.abs(test.target_yield-ensemble_pred))),'oracle_headroom_mae':float(np.mean(np.abs(test.target_yield-oracle_pred)))})
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
def run_u1(out,seeds,config_path):
    df=load_g2f(); results=[]
    for year in (2021,2022,2023):
      fit=df[df.Year.lt(year-2)]; val=df[df.Year.eq(year-2)]; cal=df[df.Year.eq(year-1)]; test=df[df.Year.eq(year)]; cols=feature_columns(fit,'target_yield')
      for seed in seeds:
       val_members=[]
       for k in range(5):
        m=make_pipeline(fit[cols],'rf',seed+k); m.fit(fit[cols],fit.target_yield); val_members.append(m.predict(val[cols]))
       val_mean=np.mean(val_members,0); val_scores={'ensemble_disagreement':np.std(val_members,0)}
       numeric=[c for c in cols if pd.api.types.is_numeric_dtype(fit[c]) and not fit[c].isna().all()]; med=fit[numeric].median(); scale=fit[numeric].std().replace(0,1); val_scores['support_distance']=np.sqrt((((val[numeric].fillna(med)-med)/scale)**2).mean(1)).to_numpy()
       score_aurc={name:deployable_aurc(val.target_yield,pd.Series(val_mean),pd.Series(score),score_name=name)['aurc'] for name,score in val_scores.items()}; selected=min(score_aurc,key=lambda name:(score_aurc[name],name))
       fit_final=pd.concat([fit,val]); preds=[]; calp=[]
       for k in range(5):
        m=make_pipeline(fit_final[cols],'rf',seed+k); m.fit(fit_final[cols],fit_final.target_yield); preds.append(m.predict(test[cols])); calp.append(m.predict(cal[cols]))
       mean=np.mean(preds,0); cmean=np.mean(calp,0); residual=np.abs(cal.target_yield-cmean); q=finite_sample_quantile(residual,.1); weighted_q,ess=weighted_quantile(residual,np.ones(len(residual)),.1)
       if selected=='ensemble_disagreement': uncertainty=np.std(preds,0)
       else:
        med=fit_final[numeric].median(); scale=fit_final[numeric].std().replace(0,1); uncertainty=np.sqrt((((test[numeric].fillna(med)-med)/scale)**2).mean(1)).to_numpy()
       lower=mean-q; upper=mean+q; random_score=np.random.default_rng(seed).permutation(uncertainty)
       runout=out/f'test_{year}'/f'seed_{seed}'; meta={'route':'U1','dataset':'g2f_native','fold':f'test_{year}','model':'rf_ensemble_split_conformal','seed':seed,'validation_year':year-2,'calibration_year':year-1,'quantile':q,'weighted_quantile':weighted_q,'effective_sample_size':ess,'score_selection':'validation_only','selected_score':selected,'oracle_role':'diagnostic_only'}; metrics=save(runout,test,mean,meta,source_path=G2F,feature_cols=cols,config_path=config_path,extra_predictions={'uncertainty_score':uncertainty,'lower':lower,'upper':upper,'random_rejection_score':random_score}); endpoints=uncertainty_endpoints(y_true=test.target_yield,y_pred=mean,lower=lower,upper=upper,uncertainty=uncertainty,random_seed=seed); (runout/'calibration_ledger.json').write_text(json.dumps({'fit_years':sorted(fit_final.Year.unique().tolist()),'calibration_year':year-1,'test_year':year,'finite_sample_quantile':q,'weighted_quantile':weighted_q,'effective_sample_size':ess})); results.append({**meta,**metrics,**endpoints})
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
def run_u2(out,seeds,config_path):
    df=load_g2f(); results=[]; full=feature_columns(df,'target_yield'); soil=[c for c in full if c.startswith('soil_')]; weather=[c for c in full if c.startswith('weather_')]
    def signatures(frame):
     return np.where(~frame[soil].notna().any(axis=1),'no_soil',np.where(~frame[weather].notna().any(axis=1),'no_weather','available'))
    for year in (2021,2022,2023):
     fit=df[df.Year.lt(year-1)]; cal=df[df.Year.eq(year-1)]; test=df[df.Year.eq(year)]
     for seed in seeds:
      for missingness_type,pattern_name,allowed in [('natural','natural',full),('synthetic','synthetic_no_soil',[c for c in full if c not in soil]),('synthetic','synthetic_no_weather',[c for c in full if c not in weather])]:
       cal_patterns=signatures(cal) if missingness_type=='natural' else np.repeat(pattern_name,len(cal)); test_patterns=signatures(test) if missingness_type=='natural' else np.repeat(pattern_name,len(test))
       model=make_pipeline(fit[allowed],'rf',seed); model.fit(fit[allowed],fit.target_yield); cal_pred=model.predict(cal[allowed]); pred=model.predict(test[allowed]); quantiles=conditional_quantiles(np.abs(cal.target_yield-cal_pred),cal_patterns,alpha=.1,min_count=30); q=np.array([quantiles.get(pattern,quantiles['__pooled__']) for pattern in test_patterns]); lower=pred-q; upper=pred+q
       meta={'route':'U2','dataset':'g2f_native','fold':f'test_{year}','model':'mondrian_pattern_conformal','seed':seed,'calibration_year':year-1,'pattern':pattern_name,'pattern_conditioning':'missingness_signature','fallback':'pooled_if_calibration_n_lt_30','missingness_type':missingness_type}; metrics=save(out/pattern_name/f'test_{year}'/f'seed_{seed}',test,pred,meta,source_path=G2F,feature_cols=allowed,config_path=config_path,extra_predictions={'missing_pattern':test_patterns,'lower':lower,'upper':upper,'uncertainty_score':q}); pattern_errors=[]
       for pattern in sorted(set(test_patterns)):
        mask=test_patterns==pattern; cm=conformal_metrics(test.loc[mask,'target_yield'],pd.Series(lower[mask]),pd.Series(upper[mask]),alpha=.1); pattern_errors.append(cm['absolute_coverage_error'])
       endpoints=uncertainty_endpoints(y_true=test.target_yield,y_pred=pred,lower=lower,upper=upper,uncertainty=q,random_seed=seed); runout=out/pattern_name/f'test_{year}'/f'seed_{seed}'; (runout/'pattern_calibration_ledger.json').write_text(json.dumps({'counts':{str(p):int((cal_patterns==p).sum()) for p in set(cal_patterns)},'fallback':meta['fallback']})); pd.DataFrame({'sample_id':test.sample_id,'uncertainty_score':q}).to_csv(runout/'aurc_inputs.csv',index=False); results.append({**meta,**metrics,**endpoints,'max_pattern_coverage_error':max(pattern_errors),'patterns':sorted(set(test_patterns))})
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
def run_baseline(out,seeds,track,config_path):
    specs={
      'G2F':(G2F,'target_yield','Year',2023,'ridge'),
      'G2F_MLP':(G2F,'target_yield','Year',2023,'mlp'),
      'CYBENCH':(CYV/'maize_US_safe.csv.gz','target_yield','season_year',2023,'rf'),
      'REGIONAL':(Path(os.environ.get('AGRITECH_REGIONAL_VIEW', ROOT/'data/external/regional/regional_abs_wheat_weather_apsoil.csv')),'target_value','year',2021,'ridge'),
      'WAITE':(Path(os.environ.get('AGRITECH_WAITE_VIEW', ROOT/'data/external/waite/waite_full_environment_proxy.csv.gz')),'observed_yield_t_ha','year',2020,'ridge')}
    path,target,yearcol,testyear,modelname=specs[track]; df=pd.read_csv(path); df[target]=pd.to_numeric(df[target],errors='coerce'); df=df[df[target].notna()].copy(); years=sorted(pd.to_numeric(df[yearcol],errors='coerce').dropna().astype(int).unique()); testyear=max([y for y in years if y<=testyear],default=max(years)); train=df[pd.to_numeric(df[yearcol],errors='coerce').lt(testyear)].copy(); test=df[pd.to_numeric(df[yearcol],errors='coerce').eq(testyear)].copy(); cols=feature_columns(train,target); results=[]
    for seed in seeds:
      model=make_pipeline(train[cols],modelname,seed); model.fit(train[cols],pd.to_numeric(train[target])); pred=model.predict(test[cols]); renamed=test.rename(columns={target:'target_yield'}) if target!='target_yield' else test
      meta={'route':f'BASELINE_{track}','fold':f'test_{testyear}','model':modelname,'seed':seed}; results.append({**meta,**save(out/f'seed_{seed}',renamed,pred,meta,source_path=path,feature_cols=cols,config_path=config_path)})
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
def main():
 global PROGRESS_PATH,ONLY_FOLD,ONLY_VARIANT,ONLY_PATTERN,ONLY_FRACTION
 p=argparse.ArgumentParser(); choices=['T1','T2','M1','M2','R1','U1','U2','BASELINE_G2F','BASELINE_G2F_MLP','BASELINE_CYBENCH','BASELINE_REGIONAL','BASELINE_WAITE']; p.add_argument('--route',required=True,choices=choices); p.add_argument('--config',type=Path,required=True); p.add_argument('--output-root',type=Path,default=ROOT/'outputs/stage8/formal_v1/reruns_v2/routes'); p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303]); p.add_argument('--resume',action='store_true'); p.add_argument('--only-fold'); p.add_argument('--only-variant'); p.add_argument('--only-pattern'); p.add_argument('--only-fraction',type=float); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
 if a.dry_run:
  print(json.dumps({'mode':'dry_run','route':a.route,'seeds':a.seeds,'output_root':str(a.output_root),'only_fold':a.only_fold,'only_variant':a.only_variant,'only_pattern':a.only_pattern,'only_fraction':a.only_fraction},sort_keys=True)); return 0
 out=a.output_root/a.route; out.mkdir(parents=True,exist_ok=True); PROGRESS_PATH=a.output_root.parent/'progress.jsonl'; ONLY_FOLD=a.only_fold; ONLY_VARIANT=a.only_variant; ONLY_PATTERN=a.only_pattern; ONLY_FRACTION=a.only_fraction
 if a.resume and route_is_fully_accepted(out,a.output_root.parent/'acceptance_v2'/'accepted_predictions_index.json'):
  print(json.dumps({'event':'resume_skip_verified_route','route':a.route,'route_path':str(out)}),flush=True); return 0
 if a.route.startswith('BASELINE_'): run_baseline(out,a.seeds,a.route.removeprefix('BASELINE_'),a.config)
 else: {'T1':run_t1,'T2':run_t2,'M1':run_m1,'M2':run_m2,'R1':run_r1,'U1':run_u1,'U2':run_u2}[a.route](out,a.seeds,a.config)
 return 0
if __name__=='__main__': raise SystemExit(main())
