"""Generate LaTeX table from statistical test registry CSV.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

VAL_DIR = Path("outputs/common_sample_attribution_validation_v1/run_20260709_090144")
OUT_FILE = Path("paper/generated_tables/supplement_statistical_registry.tex")

def main():
    df = pd.read_csv(VAL_DIR / "statistical_test_registry.csv")
    
    # Filter to primary tests for cleaner presentation in LaTeX (16 rows)
    primary_df = df[df["status"] == "primary"].copy()
    
    # Format p-values and numeric columns
    def format_p(p):
        if p < 1e-4:
            return f"{p:.2e}"
        return f"{p:.4f}"
        
    primary_df["raw_p_fmt"] = primary_df["raw_p"].apply(format_p)
    primary_df["corrected_p_fmt"] = primary_df["corrected_p"].apply(format_p)
    primary_df["effect_size_fmt"] = primary_df["effect_size"].apply(lambda x: f"{x:+.3f}")
    
    # Select columns
    # contract, route, metric, n_rows, raw_p, corrected_p, effect_size, interpretation
    cols = ["contract", "route", "metric", "n_rows", "raw_p_fmt", "corrected_p_fmt", "effect_size_fmt", "final_interpretation"]
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Complete Primary Hypothesis Statistical Test Registry. Families (Support Reliance and Linkage) are corrected independently using Holm-Bonferroni step-down correction.}")
    latex_lines.append(r"\label{tab:statistical-registry}")
    latex_lines.append(r"\scriptsize")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{llllllll}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Contract & Route & Metric & N (Pairs) & Raw $p$-value & Corrected $p$-value & Effect Size & Classification \\")
    latex_lines.append(r"\midrule")
    
    # Support family
    latex_lines.append(r"\multicolumn{8}{l}{\textbf{Family A: Primary Support Reliance (Wilcoxon Signed-Rank, F = 8)}} \\")
    sup_df = primary_df[primary_df["correction_family"] == "Primary Support Reliance"].sort_values(["contract", "route"])
    for _, row in sup_df.iterrows():
        contract = "GROUP" if "GROUP" in row["contract"] else "SPATIAL"
        metric = r"$\Delta P_{\mathrm{support}}$"
        route_fmt = row['route'].replace("_", r"\_")
        class_fmt = row['final_interpretation'].replace("_", r"\_")
        latex_lines.append(f"{contract} & {route_fmt} & {metric} & {row['n_rows']} & {row['raw_p_fmt']} & {row['corrected_p_fmt']} & {row['effect_size_fmt']} & {class_fmt} \\\\")
        
    latex_lines.append(r"\midrule")
    # Linkage family
    latex_lines.append(r"\multicolumn{8}{l}{\textbf{Family B: Primary Attribution-Error Linkage (Spearman Correlation, F = 8)}} \\")
    link_df = primary_df[primary_df["correction_family"] == "Primary Linkage D_L1"].sort_values(["contract", "route"])
    for _, row in link_df.iterrows():
        contract = "GROUP" if "GROUP" in row["contract"] else "SPATIAL"
        metric = r"$\rho$ ($D_{L1}$ vs $\Delta_{AE}$)"
        route_fmt = row['route'].replace("_", r"\_")
        class_fmt = row['final_interpretation'].replace("_", r"\_")
        latex_lines.append(f"{contract} & {route_fmt} & {metric} & {row['n_rows']} & {row['raw_p_fmt']} & {row['corrected_p_fmt']} & {row['effect_size_fmt']} & {class_fmt} \\\\")
        
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table}")
    
    OUT_FILE.write_text("\n".join(latex_lines))
    print(f"Generated {OUT_FILE}")

if __name__ == "__main__":
    main()
