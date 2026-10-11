"""Execute original source training, screening and GROUP transfer on real CY-Bench views."""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.distillation_v4.universal_weather_v2_remediation.planner import build_candidate_matrix
from src.distillation_v4.universal_weather_v2_remediation.remaining_handlers import execute_source_screening,execute_teacher_qualification
from src.distillation_v4.universal_weather_v2_remediation.execution import execute_target_job


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(a.config.read_text())
    cells=[c for c in build_candidate_matrix(cfg) if c['dataset_id']=='CY-Bench_wheat_AU' and c['source_dataset']=='PRIMARY_CYBENCH_MAIZE_US' and c['source_route']=='supervised' and c['transfer_strategy']=='ordinary_transfer' and c['adaptation_mode']=='full_adaptation' and c['deployment_condition']=='complete' and c['split_id']=='GROUP' and c['status']=='EXECUTABLE']
    if len(cells)!=1:raise ValueError('The unchanged candidate contract did not identify one executable GROUP transfer cell')
    cell=cells[0];source={'source_dataset':cell['source_dataset'],'source_route':cell['source_route'],'seed':cell['seed']}
    (a.output/'protocol.json').write_text(json.dumps({'scope':'ENGINEERING_REAL_DATA_ONLY','source_regions_reduced':True,'candidate':cell,'historical_checkpoint_used':False},indent=2)+'\n')
    screening=execute_source_screening(cfg,source,repository_root=ROOT,output_root=a.output,smoke=True)
    qualification=execute_teacher_qualification(cfg,source,repository_root=ROOT,output_root=a.output,smoke=True)
    result=execute_target_job(cfg,cell,repository_root=ROOT,output_root=a.output,smoke=True)
    (a.output/'verification.json').write_text(json.dumps({'status':'PIPELINE_VERIFIED','scope':'Real US maize source training and screening, fresh source checkpoint, AU wheat GROUP target training/prediction','source_screening':screening,'teacher_qualification':qualification,'target_result':result,'full_numeric_reproduced':False},indent=2,default=str)+'\n')


if __name__=='__main__':main()
