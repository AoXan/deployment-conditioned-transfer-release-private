"""Select outcome-blind US regions from genuine raw tables for source-training validation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.distillation_v4.universal_weather_v2.adapters import build_cybench_view


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--regions',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=pd.read_csv(a.raw_root/'yield_maize_US.csv')
    regions=sorted(source.adm_id.astype(str).unique())[:a.regions]
    raw=a.output/'selected_raw';raw.mkdir();receipts={}
    for prefix in ['yield','meteo','crop_calendar','soil','location','crop_mask']:
        path=a.raw_root/f'{prefix}_maize_US.csv';destination=raw/path.name;first=True;count=0;digest=hashlib.sha256()
        for chunk in pd.read_csv(path,chunksize=100000):
            selected=chunk[chunk.adm_id.astype(str).isin(regions)]
            selected.to_csv(destination,mode='w' if first else 'a',header=first,index=False);first=False;count+=len(selected)
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        receipts[path.name]={'source_sha256':digest.hexdigest(),'selected_rows':count}
        print(prefix,count,flush=True)
    view=a.output/'PRIMARY_CYBENCH_MAIZE_US.csv.gz'
    metadata=build_cybench_view(dataset_id='PRIMARY_CYBENCH_MAIZE_US',directory=raw,crop='maize',country='US',output_path=view)
    (a.output/'protocol.json').write_text(json.dumps({'role':'Engineering sample reduction only; full protocol uses all original regions','selection':'First lexical administrative region IDs, no target-dependent selection','regions':regions,'raw_sources':receipts,'adapter':metadata},indent=2,default=str)+'\n')


if __name__=='__main__':main()
