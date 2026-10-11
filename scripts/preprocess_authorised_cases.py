"""Run recovered original G2F/Roseworthy ETL with external inputs and new outputs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['g2f','roseworthy'],required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.resolve()==a.raw_root.resolve():raise ValueError('Output must be independent of raw root')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['AGRITECH_RAW_ROOT']=str(a.raw_root.resolve());os.environ['AGRITECH_NURSERY_OUTPUT']=str(a.output.resolve())
    path=Path(__file__).with_name('build_data_nursery_v1.py');spec=importlib.util.spec_from_file_location('nursery',path)
    import sys
    m=importlib.util.module_from_spec(spec);sys.modules['nursery']=m;spec.loader.exec_module(m);m.ensure_dirs()
    if a.dataset=='g2f':
        required=[m.TRAIN/(f'{i}_Training_{name}_Data_{years}.csv') for i,name,years in [(1,'Trait','2014_2023'),(2,'Meta','2014_2023'),(3,'Soil','2015_2023'),(6,'EC','2014_2023')]]
        required.append(m.TRAIN/'4_Training_Weather_Data_2014_2023_seasons_only.csv')
    else:required=[m.ROOT/'Agritech_Datasets/academic/figshare_roseworthy'/f'E5_{year}_cleaned_elev_moisture_yield_xy.csv' for year in (2007,2008)]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise FileNotFoundError('Exact authorised/raw input files required: '+str(missing))
    if a.dataset=='g2f':frame=m.build_g2f()
    else:frame=m.build_precision(m.parse_apsoil())
    if frame.empty:raise ValueError('No usable observations; inspect original ETL failure ledger')
    m.write_global_ledgers()
    (a.output/'preprocessing_receipt.json').write_text(json.dumps({'dataset':a.dataset,'rows':len(frame),'raw_inputs':[str(p) for p in required],'historic_results_used':False,'status':'EXECUTED_NOT_NUMERICALLY_VERIFIED'},indent=2)+'\n')


if __name__=='__main__':main()
