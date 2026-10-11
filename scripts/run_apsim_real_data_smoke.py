"""Run the unchanged APSIM scenarios on one real region-year and fresh predictions."""
import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--prediction',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True,help='Pinned locally built Models.dll')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('apsim_original_builder',ROOT/'experiments/apsim_comparison/build_experiment.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    builder.CYB=a.raw_root.resolve();builder.OUT=a.output.resolve();builder.GENERATED=a.output.resolve()/'configs';builder.WEATHER_DIR=builder.GENERATED/'weather';builder.WEATHER_DIR.mkdir(parents=True)
    predictions=pd.read_csv(a.prediction);predictions['contract']='ADAPTED_TEMPORAL_SMOKE';predictions['seed']=101
    samples=builder.build_sample_manifest(predictions)
    selected=samples[samples.eligible].sort_values('sample_id').head(1)
    if len(selected)!=1:raise ValueError('No legally available actual-data region-year with complete APSIM inputs')
    selected.to_csv(a.output/'sample_manifest.csv',index=False)
    builder.write_met_files(selected)
    provenance=builder.build_apsim_files(selected)
    provenance.to_csv(a.output/'configuration_provenance.csv',index=False)
    runs=[]
    for config in sorted(builder.GENERATED.glob('*.apsimx')):
        with (a.output/(config.stem+'.log')).open('w') as log:
            execution=subprocess.run(['dotnet',str(a.runtime.resolve()),str(config.resolve())],cwd=builder.GENERATED,stdout=log,stderr=subprocess.STDOUT)
        runs.append({'scenario':config.stem,'returncode':execution.returncode})
        if execution.returncode:raise RuntimeError(f'APSIM execution failed: {config.stem}; inspect log')
    outputs=[]
    for database in sorted(builder.GENERATED.glob('*.db')):
        with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as connection:
            tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'Report' not in tables:raise ValueError(f'No simulation Report table: {database}')
            frame=pd.read_sql_query('SELECT * FROM Report',connection)
            frame['scenario']=database.stem;outputs.append(frame)
    if not outputs:raise ValueError('No fresh simulation database generated')
    combined=pd.concat(outputs,ignore_index=True);combined.to_csv(a.output/'fresh_apsim_report.csv',index=False)
    (a.output/'verification.json').write_text(json.dumps({'status':'SMOKE_EXECUTED','input_kind':'Public CY-Bench daily weather, soil and actual region-year; current-run model predictions','historical_apsim_results_used':False,'scenarios':runs,'simulation_report_rows':len(combined),'sample_ids':selected.sample_id.tolist(),'scope':'One real region-year, unchanged registered scenario definitions; not accepted-paper numeric reproduction'},indent=2)+'\n')


if __name__=='__main__':
    main()
