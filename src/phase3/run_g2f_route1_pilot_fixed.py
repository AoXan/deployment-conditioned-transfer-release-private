import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc

def compute_aurc(abs_errors, risk_scores):
    n = len(abs_errors)
    sort_idx = np.argsort(risk_scores)
    sorted_err = abs_errors[sort_idx]
    
    risk_curve = [np.mean(sorted_err[:k]) for k in range(1, n+1)]
    coverage_curve = np.linspace(1/n, 1.0, n)
    return auc(coverage_curve, risk_curve), risk_curve

def run_pilot(data_path: Path, split_path: Path, out_dir: Path, validate_only: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file() or not split_path.is_file():
        missing = [str(path) for path in (data_path, split_path) if not path.is_file()]
        raise FileNotFoundError(f"missing authorised G2F input: {missing}")
    
    print("Loading G2F Full Data...")
    df = pd.read_csv(data_path)
    
    # 1. Load Split and Isolate
    split_df = pd.read_csv(split_path)
    active_ids = set(split_df['sample_id'])
    
    active_df = df[df['sample_id'].isin(active_ids)].copy()
    
    # 3-Layer Isolation based on Year
    test_mask = active_df['Year'] == 2022
    calib_mask = active_df['Year'] == 2021
    train_mask = active_df['Year'] < 2021
    
    test_df = active_df[test_mask].copy()
    calib_df = active_df[calib_mask].copy()
    train_df = active_df[train_mask].copy()

    if validate_only:
        print(json.dumps({"status": "PASS", "train": len(train_df), "calibration": len(calib_df), "test": len(test_df)}, indent=2))
        return
    
    print(f"Data isolated: Train={len(train_df)} (Years {train_df['Year'].min()}-{train_df['Year'].max()}), "
          f"Calib={len(calib_df)} (Year 2021), Test={len(test_df)} (Year 2022)")
          
    feat_cols = [c for c in active_df.columns if c.startswith('weather_') or c.startswith('soil_') or c.startswith('ec_')]
    target = 'target_yield'
    
    # 2. Scaler on Train ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feat_cols].fillna(0))
    y_train = train_df[target].values
    
    X_calib = scaler.transform(calib_df[feat_cols].fillna(0))
    y_calib = calib_df[target].values
    
    X_test = scaler.transform(test_df[feat_cols].fillna(0))
    y_test = test_df[target].values
    
    # 3. Best Fixed Model
    print("Training Best Fixed Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    # 4. Deep Ensemble
    print("Training Deep Ensemble (M=5 MLPs)...")
    ensemble_models = []
    for seed in range(5):
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50, random_state=seed, early_stopping=True, validation_fraction=0.1) 
        mlp.fit(X_train, y_train)
        ensemble_models.append(mlp)
        
    # Calibration
    val_preds_ens = np.array([m.predict(X_calib) for m in ensemble_models])
    val_mean = np.mean(val_preds_ens, axis=0)
    val_residuals = np.abs(y_calib - val_mean)
    
    # Test
    test_preds_ens = np.array([m.predict(X_test) for m in ensemble_models])
    test_mean = np.mean(test_preds_ens, axis=0)
    test_variance = np.var(test_preds_ens, axis=0)
    
    # 5. Conformal
    alpha = 0.1
    n_val = len(val_residuals)
    q_level = min(np.ceil((n_val + 1) * (1 - alpha)) / n_val, 1.0)
    q_hat = np.quantile(val_residuals, q_level)
    
    test_abs_err = np.abs(test_mean - y_test)
    uncert_de = test_variance
    
    # 6. Evaluation
    np.random.seed(101)
    rand_aurcs = [compute_aurc(test_abs_err, np.random.rand(len(test_abs_err)))[0] for _ in range(30)]
    mean_rand_aurc = np.mean(rand_aurcs)
    rand_aurc_std = np.std(rand_aurcs)
    
    de_aurc, _ = compute_aurc(test_abs_err, uncert_de)
    aurc_improvement = mean_rand_aurc - de_aurc
    
    # Catastrophic Recall
    top_10_thresh = np.percentile(test_abs_err, 90)
    is_catastrophic = test_abs_err >= top_10_thresh
    rejected_idx = np.argsort(uncert_de)[-int(0.3 * len(uncert_de)):]
    rejected_mask = np.zeros(len(test_abs_err), dtype=bool)
    rejected_mask[rejected_idx] = True
    
    abs_recall = np.sum(is_catastrophic & rejected_mask) / np.sum(is_catastrophic)
    random_recall_baseline = 0.3 # Randomly rejecting 30% finds 30% of catastrophic errors on average
    rel_multiple = abs_recall / random_recall_baseline if random_recall_baseline > 0 else 0
    delta_recall = abs_recall - random_recall_baseline
    
    cov_std = np.mean(test_abs_err <= q_hat)
    
    metrics = {
        "unit": "temporal_forward_2022",
        "random_aurc_mean": mean_rand_aurc,
        "random_aurc_std": rand_aurc_std,
        "deep_ensemble_aurc": de_aurc,
        "aurc_improvement": aurc_improvement,
        "catastrophic_recall_absolute": abs_recall,
        "catastrophic_recall_relative_multiple": rel_multiple,
        "catastrophic_recall_delta": delta_recall,
        "empirical_coverage": cov_std,
        "coverage_error": cov_std - 0.90,
        "interval_width": 2 * q_hat
    }
    
    print("\n--- METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    with open(out_dir / 'unit_metrics_2022.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Saving artifacts...")
    pd.DataFrame({'sample_id': test_df['sample_id'], 'y_true': y_test, 'y_pred': test_mean, 'variance': test_variance}).to_csv(out_dir / 'predictions.csv', index=False)
    with open(out_dir / 'run_manifest.json', 'w') as f:
        json.dump({"dataset": "G2F_native", "split": "temporal_forward_2022", "models": "RF+5xMLP", "status": "COMPLETED"}, f)

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed G2F 2022 temporal diagnostic.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/publication_repro/cases/g2f"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"data": str(args.data), "split": str(args.split), "output": str(args.output), "executed": False}, indent=2))
        return 0
    run_pilot(args.data, args.split, args.output, args.validate_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
