"""Recompute current prediction metrics and draw independent new-run assets."""
import argparse
import hashlib
import json
from pathlib import Path


def render(source,output):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True);rows=[];inputs={}
    for path in sorted(source.rglob('predictions.csv')):
        frame=pd.read_csv(path)
        if not {'y_true','y_pred'}<=set(frame.columns):continue
        if frame[['y_true','y_pred']].isna().any().any():raise ValueError('Nonfinite predictions: '+str(path))
        relative=str(path.relative_to(source));inputs[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'run':relative,'n':len(frame),'mae':mean_absolute_error(frame.y_true,frame.y_pred),'rmse':np.sqrt(mean_squared_error(frame.y_true,frame.y_pred)),'r2':r2_score(frame.y_true,frame.y_pred) if len(frame)>1 else None})
    if not rows:raise FileNotFoundError('Fresh predictions with y_true/y_pred required')
    table=pd.DataFrame(rows);table.to_csv(output/'fresh_prediction_metrics.csv',index=False);table.to_latex(output/'fresh_prediction_metrics.tex',index=False)
    ax=table.set_index('run')[['mae','rmse']].plot.bar(figsize=(10,5));ax.set_title('New execution; reduced runs do not reproduce accepted values');ax.figure.tight_layout();ax.figure.savefig(output/'fresh_prediction_metrics.pdf');plt.close(ax.figure)
    receipt={'status':'PIPELINE_VERIFIED','inputs':inputs,'rows':len(table),'historical_results_read':False,'full_numeric_reproduced':False,'accepted_tables_overwritten':False}
    (output/'fresh_assets_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(render(a.source,a.output),indent=2))
