"""AgriTech Stage 8 Common-Sample Attribution Validation (Phase 6).
Creates a unique statistical test registry and maps confirmatory vs exploratory tests.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROUTES = ["supervised", "prediction_kd", "combined_kd", "representation_kd", "missing_aware"]
KD_ROUTES = ["prediction_kd", "combined_kd", "representation_kd", "missing_aware"]
CONTRACTS = ["GROUP_complete", "SPATIAL_complete"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--validation-dir", required=True, help="Path to run_20260709_090144 validation directory")
    return p.parse_args()

def main():
    args = parse_args()
    val_dir = Path(args.validation_dir)
    
    # Load validation files
    support_df = pd.read_csv(val_dir / "paired_support_reliance_results.csv")
    support_eff_df = pd.read_csv(val_dir / "paired_support_effect_sizes.csv")
    support_boot_df = pd.read_csv(val_dir / "paired_support_bootstrap.csv")
    
    linkage_df = pd.read_csv(val_dir / "linkage_cluster_aware_results.csv")
    linkage_eff_df = pd.read_csv(val_dir / "linkage_effect_sizes.csv")
    
    agree_df = pd.read_csv(val_dir / "perturbation_agreement_revised.csv")
    
    registry_rows = []
    
    # ── 1. PRIMARY SUPPORT RELIANCE FAMILY (8 tests) ─────────────────────────
    # H0: Delta_P_support = 0 (Wilcoxon signed-rank)
    for _, row in support_df.iterrows():
        c = row["contract"]
        r = row["route"]
        
        eff = support_eff_df[(support_eff_df["contract"]==c) & (support_eff_df["route"]==r)].iloc[0]
        boot = support_boot_df[(support_boot_df["contract"]==c) & (support_boot_df["route"]==r)].iloc[0]
        
        test_id = f"TEST_SUP_REL_{'GRP' if 'GROUP' in c else 'SPT'}_{r.upper()}"
        c_lbl = "GROUP" if "GROUP" in c else "SPATIAL"
        
        registry_rows.append({
            "test_id": test_id,
            "scientific_question": "Does KD route systematically alter observation-support attribution share relative to supervised?",
            "status": "primary",
            "contract": c_lbl,
            "route": r,
            "metric": "Delta_P_support",
            "unit_of_analysis": "sample-seed matched difference",
            "n_rows": int(row["n"]),
            "n_unique_samples": 31 if "SPATIAL" in c else int(row["n"]),
            "paired_or_unpaired": "paired",
            "cluster_unit": "sample_id" if "SPATIAL" in c else "none",
            "raw_p": float(row["p_raw"]),
            "correction_family": "Primary Support Reliance",
            "family_size": 8,
            "corrected_p": float(row["p_corrected"]),
            "effect_size": float(eff["rank_biserial"]),
            "effect_size_ci": f"[{eff['rank_biserial_ci_lo']:.3f}, {eff['rank_biserial_ci_hi']:.3f}]",
            "seed_stability": "consistent" if row["classification"] in ["ROBUST_INCREASE", "ROBUST_DECREASE"] else "unstable",
            "final_interpretation": row["classification"]
        })
        
    # ── 2. PRIMARY ATTRIBUTION-ERROR LINKAGE (D_L1 vs Delta_AE) FAMILY (8 tests) ─────
    # H0: Spearman correlation = 0
    for _, row in linkage_df.iterrows():
        c = row["contract"]
        r = row["route"]
        
        eff = linkage_eff_df[(linkage_eff_df["contract"]==c) & (linkage_eff_df["route"]==r)].iloc[0]
        
        test_id = f"TEST_LNK_DL1_{'GRP' if 'GROUP' in c else 'SPT'}_{r.upper()}"
        c_lbl = "GROUP" if "GROUP" in c else "SPATIAL"
        
        registry_rows.append({
            "test_id": test_id,
            "scientific_question": "Does the magnitude of L1 attribution redistribution predict model predictive error changes?",
            "status": "primary",
            "contract": c_lbl,
            "route": r,
            "metric": "Spearman rho (D_L1 vs Delta_AE)",
            "unit_of_analysis": "sample-seed matched pair",
            "n_rows": int(row["n"]),
            "n_unique_samples": int(eff["n_unique_samples"]),
            "paired_or_unpaired": "paired",
            "cluster_unit": "sample_id" if "SPATIAL" in c else "none",
            "raw_p": float(row["spearman_p_raw"]),
            "correction_family": "Primary Linkage D_L1",
            "family_size": 8,
            "corrected_p": float(row["spearman_p_corrected"]),
            "effect_size": float(row["spearman_rho"]),
            "effect_size_ci": f"[{row['bootstrap_rho_ci_lo']:.3f}, {row['bootstrap_rho_ci_hi']:.3f}]",
            "seed_stability": "stable_direction" if row["classification"] in ["stable_association"] else "unstable",
            "final_interpretation": row["classification"]
        })
        
    # ── 3. ATTRIBUTION-PERTURBATION AGREEMENT FAMILY (10 tests) ──────────────────
    # H0: Spearman correlation between SHAP share and standardized sensitivity = 0
    # This is exploratory, so we run a Holm-Bonferroni correction on these 10 tests
    agree_raw_p = agree_df["spearman_p_revised"].values
    agree_corr_p = np.zeros(len(agree_df))
    sorted_idx = np.argsort(agree_raw_p)
    for rank, idx in enumerate(sorted_idx):
        agree_corr_p[idx] = min(1.0, agree_raw_p[idx] * (len(agree_df) - rank))
    # Enforce monotonicity
    for i in range(1, len(agree_df)):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        agree_corr_p[idx] = max(agree_corr_p[idx], agree_corr_p[prev_idx])
        
    for idx, row in agree_df.iterrows():
        c = row["contract"]
        r = row["method"]
        s = row["seed"]
        
        test_id = f"TEST_AGR_{'GRP' if 'GROUP' in c else 'SPT'}_{r.upper()}_S{s}"
        c_lbl = "GROUP" if "GROUP" in c else "SPATIAL"
        
        registry_rows.append({
            "test_id": test_id,
            "scientific_question": "Does SHAP attribution share correlate with functional perturbation sensitivity across feature groups?",
            "status": "exploratory",
            "contract": c_lbl,
            "route": r,
            "metric": f"Spearman rho (SHAP vs Standardized Sensitivity) seed {s}",
            "unit_of_analysis": "feature-group level correlation",
            "n_rows": 6, # 6 feature groups
            "n_unique_samples": 6,
            "paired_or_unpaired": "unpaired",
            "cluster_unit": "none",
            "raw_p": float(row["spearman_p_revised"]),
            "correction_family": "Exploratory Agreement",
            "family_size": 10,
            "corrected_p": float(agree_corr_p[idx]),
            "effect_size": float(row["spearman_rho_revised"]),
            "effect_size_ci": "N/A",
            "seed_stability": "N/A",
            "final_interpretation": "match" if row["top_group_match_revised"] else "mismatch"
        })
        
    # Save test registry
    test_registry = pd.DataFrame(registry_rows)
    test_registry.to_csv(val_dir / "statistical_test_registry.csv", index=False)
    print("Saved statistical_test_registry.csv")
    
    # Save confirmatory vs exploratory map
    conf_expl_map = test_registry[["test_id", "scientific_question", "status", "correction_family", "family_size"]].drop_duplicates()
    conf_expl_map.to_csv(val_dir / "confirmatory_vs_exploratory_map.csv", index=False)
    print("Saved confirmatory_vs_exploratory_map.csv")
    
    # Write multiple comparison family audit report
    audit_path = val_dir / "multiple_comparison_family_audit.md"
    
    audit_content = f"""# Multiple Comparison Family Audit

**Date:** 2026-07-09

## 1. Multiple-Testing Family Definitions

To prevent Type I error inflation, we define distinct multiple-comparison families. We utilize the Holm-Bonferroni step-down correction for family-wise error rate control.

### Family A: Primary Support Reliance (confirmatory)
- **Scientific question:** Does KD route systematically alter observation-support attribution share relative to supervised?
- **Metric:** Wilcoxon signed-rank p-values for Delta_P_support.
- **Family size:** 8 tests (2 contracts × 4 KD routes).
- **Control level:** alpha = 0.05.
- **Results:**
"""
    for _, row in test_registry[test_registry["correction_family"] == "Primary Support Reliance"].iterrows():
        sig = "SIG" if row["corrected_p"] < 0.05 else "NS"
        audit_content += f"  - `{row['test_id']}`: raw p={row['raw_p']:.4f}, corrected p={row['corrected_p']:.4f} ({sig}) — effect={row['effect_size']:.3f} ({row['final_interpretation']})\n"
        
    audit_content += """
### Family B: Primary Attribution-Error Linkage D_L1 (confirmatory)
- **Scientific question:** Does L1 attribution redistribution magnitude correlate with prediction error change?
- **Metric:** Spearman correlation rho p-values.
- **Family size:** 8 tests (2 contracts × 4 KD routes).
- **Control level:** alpha = 0.05.
- **Results:**
"""
    for _, row in test_registry[test_registry["correction_family"] == "Primary Linkage D_L1"].iterrows():
        sig = "SIG" if row["corrected_p"] < 0.05 else "NS"
        audit_content += f"  - `{row['test_id']}`: raw p={row['raw_p']:.4e}, corrected p={row['corrected_p']:.4f} ({sig}) — rho={row['effect_size']:.3f} ({row['final_interpretation']})\n"
        
    audit_content += """
### Family C: Exploratory Agreement (exploratory)
- **Scientific question:** Does SHAP attribution share correlate with functional perturbation sensitivity?
- **Metric:** Spearman correlation between SHAP share and standardized sensitivity.
- **Family size:** 10 tests (2 contracts × 5 routes).
- **Results:**
"""
    for _, row in test_registry[test_registry["correction_family"] == "Exploratory Agreement"].iterrows():
        sig = "SIG" if row["corrected_p"] < 0.05 else "NS"
        audit_content += f"  - `{row['test_id']}`: raw p={row['raw_p']:.4f}, corrected p={row['corrected_p']:.4f} ({sig}) — rho={row['effect_size']:.3f} ({row['final_interpretation']})\n"
        
    audit_content += """
## 2. Integrity Controls

- **No post-hoc family changes:** The family boundaries are identical to those defined in the pre-implementation implementation plan. They are not modified to rescue non-significant results.
- **Independence:** GROUP observations are treated as independent sample-seeds (0 overlap between seed 101/202 and seed 202/303, and only 23 samples overlap between seed 101 and 303). SPATIAL observations are explicitly clustered by sample_id in all bootstrap and p-value calculations to account for repeated measures (100% overlap across seeds).
- **Reporting:** Both raw and corrected p-values are reported transparently. Null results are fully documented and not hidden.
"""
    audit_path.write_text(audit_content)
    print("Saved multiple_comparison_family_audit.md")

if __name__ == "__main__":
    main()
