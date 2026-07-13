#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


REQUIRED = (
    "docs/final_codex_synthesis/scientific_story_and_novelty.md",
    "docs/final_codex_synthesis/evidence_and_method_registry.csv",
    "docs/final_codex_synthesis/literature_and_citation_audit.csv",
    "docs/final_codex_synthesis/final_acceptance_audit.md",
    "paper/final_ajcai_manuscript.tex",
    "paper/final_ajcai_manuscript.pdf",
    "paper/final_ajcai_supplement.tex",
    "paper/final_ajcai_supplement.pdf",
    "paper/final_references.bib",
    "figures/final_publication/evidence_build_manifest.json",
    "figures/final_publication/provenance_manifest.csv",
    "figures/final_publication/captions.md",
)
FORBIDDEN_MAIN = (
    "Stage 8",
    "formal campaign",
    "candidate_id",
    "job completed",
    "audit revealed",
    "engineering acceptance",
    "max_evals",
    "training-median reference",
    "proves that",
)
REQUIRED_V2 = (
    "docs/final_codex_synthesis_v2/final_revision_summary.md",
    "docs/final_codex_synthesis_v2/final_acceptance_audit.md",
    "paper/final_ajcai_manuscript_v2.tex",
    "paper/final_ajcai_manuscript_v2.pdf",
    "paper/final_ajcai_supplement_v2.tex",
    "paper/final_ajcai_supplement_v2.pdf",
    "paper/final_references_v2.bib",
    "figures/final_publication_v2/evidence_build_manifest.json",
    "figures/final_publication_v2/provenance_manifest.csv",
    "figures/final_publication_v2/captions.md",
)


def extract_citations(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", text):
        keys.update(
            key.strip()
            for key in match.group(1).split(",")
            if key.strip()
        )
    return keys


def extract_bibtex_keys(text: str) -> set[str]:
    return set(
        re.findall(
            r"@\w+\s*\{\s*([^,\s]+)\s*,",
            text,
            flags=re.MULTILINE,
        )
    )


def pdf_page_count(path: Path) -> int:
    """Read a page count without adding a Python PDF dependency to the project."""
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"pdfinfo did not report a page count for {path}")
    return int(match.group(1))


def validate_v2(root: Path) -> dict[str, object]:
    errors: list[str] = []
    for relative in REQUIRED_V2:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"MISSING_OR_EMPTY:{relative}")
    if errors:
        return {
            "errors": errors,
            "citation_count": 0,
            "bib_key_count": 0,
            "figures_complete": False,
        }

    manuscript = (root / "paper/final_ajcai_manuscript_v2.tex").read_text(
        encoding="utf-8"
    )
    supplement = (root / "paper/final_ajcai_supplement_v2.tex").read_text(
        encoding="utf-8"
    )
    combined = manuscript + "\n" + supplement
    bib_text = (root / "paper/final_references_v2.bib").read_text(
        encoding="utf-8"
    )
    citations = extract_citations(combined)
    bib_keys = extract_bibtex_keys(bib_text)
    missing_citations = sorted(citations.difference(bib_keys))
    unused_bib = sorted(bib_keys.difference(citations))
    if missing_citations:
        errors.append(f"UNDEFINED_CITATIONS:{missing_citations}")
    if unused_bib:
        errors.append(f"UNUSED_BIB_KEYS:{unused_bib}")
    all_occurrences = re.findall(
        r"@\w+\s*\{\s*([^,\s]+)\s*,",
        bib_text,
        flags=re.MULTILINE,
    )
    duplicates = sorted(
        key
        for key in set(all_occurrences)
        if all_occurrences.count(key) > 1
    )
    if duplicates:
        errors.append(f"DUPLICATE_BIB_KEYS:{duplicates}")

    required_title = (
        "When Transfer Accuracy Is Not Enough: "
        "Deployment-Conditioned Behavioural Diagnostics for Knowledge Transfer"
    )
    if required_title not in manuscript:
        errors.append("TITLE_MISMATCH")
    forbidden_main = (
        "Stage 8",
        "formal campaign",
        "formal artifact",
        "candidate_id",
        "job completed",
        "audit revealed",
        "engineering acceptance",
        "acceptance gate",
        "registry contains",
        "replay set",
        "395 contract",
        "287 distinct",
        "279 under",
        "overlap 23",
        "calibration",
        "max_evals",
        "proves that",
        "causal importance",
        "shortcut",
    )
    for phrase in forbidden_main:
        if phrase.lower() in manuscript.lower():
            errors.append(f"FORBIDDEN_MAIN_LANGUAGE:{phrase}")
    forbidden_combined = (
        "[?]",
        r"\textbf{TODO}",
        "placeholder",
        "0.275",
        "-0.322",
        "10D Mahalanobis",
        "76/280",
        "global Taylor",
        "global Hessian",
    )
    for phrase in forbidden_combined:
        if phrase.lower() in combined.lower():
            errors.append(f"OBSOLETE_OR_UNRESOLVED_CONTENT:{phrase}")
    required_main = (
        "0.954",
        "1.204",
        "1.157",
        "-0.040",
        "98.9",
        "15.6",
        "seed-level estimates are 0.415, 0.442, and $-0.274$",
        "model--target--deployment contract",
        "behavioural diagnostics",
        "nine-dimensional Ledoit--Wolf",
    )
    for phrase in required_main:
        if phrase.lower() not in manuscript.lower():
            errors.append(f"MISSING_REQUIRED_FACT:{phrase}")
    if "beats both controls in only one seed" not in supplement:
        errors.append("PI_VARIABILITY_NOT_DISCLOSED")
    if "exact hash closure" not in supplement.lower():
        errors.append("ROSEWORTHY_EXACT_FILE_STATUS_MISSING")
    if "A high-error-tail analysis is not reported" not in supplement:
        errors.append("HIGH_ERROR_TAIL_PROVENANCE_DECISION_MISSING")

    manifest = json.loads(
        (
            root
            / "figures/final_publication_v2/evidence_build_manifest.json"
        ).read_text(encoding="utf-8")
    )
    summary = manifest["summary"]
    if summary.get("profile") != "v2":
        errors.append("EVIDENCE_PROFILE_MISMATCH")
    if summary.get("performance_landscape_rows") != 36:
        errors.append("PERFORMANCE_LANDSCAPE_ROW_MISMATCH")
    if summary.get("linkage_rows") != 8:
        errors.append("LINKAGE_FAMILY_MISMATCH")
    if summary.get("agreement_rows") != 30:
        errors.append("PERTURBATION_AGREEMENT_ROW_MISMATCH")
    if summary.get("ood_dimensions") != [9]:
        errors.append("OOD_DIMENSION_MISMATCH")
    if abs(summary.get("ig_pass_rate", 0) - 0.9893670886) > 1e-9:
        errors.append("IG_PASS_RATE_MISMATCH")
    taylor = summary.get("taylor_gate", {})
    if taylor.get("n_pass") != 28 or taylor.get("n_eligible") != 180:
        errors.append("TAYLOR_GATE_MISMATCH")

    figures_complete = True
    for number in range(1, 7):
        for suffix in (".pdf", ".svg", ".png"):
            path = (
                root
                / f"figures/final_publication_v2/figure_{number:02d}{suffix}"
            )
            if not path.is_file() or path.stat().st_size < 1000:
                figures_complete = False
                errors.append(f"MISSING_FIGURE:{path.name}")
    provenance = pd.read_csv(
        root / "figures/final_publication_v2/provenance_manifest.csv"
    )
    if float(provenance["minimum_font_pt"].min()) < 8.0:
        errors.append("FIGURE_FONT_FLOOR_MISMATCH")
    lineage = pd.read_csv(
        root / "figures/final_publication_v2/source_lineage_validation.csv"
    )
    if not lineage["status"].isin(["PASS", "NOT_APPLICABLE"]).all():
        errors.append("LINEAGE_VALIDATION_FAILURE")

    return {
        "errors": errors,
        "citation_count": len(citations),
        "bib_key_count": len(bib_keys),
        "figures_complete": figures_complete,
        "lineage_status": lineage["status"].value_counts().to_dict(),
    }


def validate_v4(root: Path) -> dict[str, object]:
    errors: list[str] = []
    required = (
        "paper/final_ajcai_manuscript_v4.tex",
        "paper/final_ajcai_manuscript_v4.pdf",
        "paper/final_ajcai_supplement_v4.tex",
        "paper/final_ajcai_supplement_v4.pdf",
        "paper/final_references_v4.bib",
        "figures/final_publication_v4/evidence_build_manifest.json",
        "figures/final_publication_v4/provenance_manifest.csv",
        "figures/final_publication_v4/captions.md",
        "figures/final_publication_v4/source_complete_experiment_map.csv",
        "figures/final_publication_v4/supplement_figure_s2.pdf",
        "figures/final_publication_v4/supplement_figure_s4.pdf",
    )
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"MISSING_OR_EMPTY:{relative}")
    if errors:
        return {"errors": errors, "citation_count": 0, "bib_key_count": 0, "figures_complete": False}
    manuscript = (root / "paper/final_ajcai_manuscript_v4.tex").read_text(encoding="utf-8")
    supplement = (root / "paper/final_ajcai_supplement_v4.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    bib_text = (root / "paper/final_references_v4.bib").read_text(encoding="utf-8")
    citations = extract_citations(combined)
    bib_keys = extract_bibtex_keys(bib_text)
    if citations.difference(bib_keys):
        errors.append(f"UNDEFINED_CITATIONS:{sorted(citations.difference(bib_keys))}")
    if bib_keys.difference(citations):
        errors.append(f"UNUSED_BIB_KEYS:{sorted(bib_keys.difference(citations))}")
    occurrences = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text, flags=re.MULTILINE)
    if any(occurrences.count(key) > 1 for key in set(occurrences)):
        errors.append("DUPLICATE_BIB_KEYS")
    required_main = (
        "When Transfer Accuracy Is Not Enough",
        "0.954", "1.204", "1.157", "0.220", "-0.040",
        "98.9", "15.6", "model--target--deployment contract",
        "diagnostic informativeness", "q_{G}^{r,c,s}", "d_{\\phi}",
        r"\Delta p",
    )
    for phrase in required_main:
        if phrase.lower() not in manuscript.lower():
            errors.append(f"MISSING_REQUIRED_FACT:{phrase}")
    if "Complete Experiment Map" not in supplement:
        errors.append("EXPERIMENT_MAP_NOT_DESCRIBED")
    forbidden = (
        "Stage 8", "formal campaign", "candidate_id", "audit revealed",
        "validator", "registry contains", "replay set", "395 contract",
        "287 distinct", "279 under", "overlap 23", "calibration",
        "0.275", "-0.322", "10D Mahalanobis", "76/280", "global Taylor",
        "global Hessian", "causal importance", "shortcut", "[?]",
        "9 of 14", "D_1", "D_{L1}", "D_L1", "Delta AE", "P_g",
    )
    for phrase in forbidden:
        if phrase.lower() in manuscript.lower():
            errors.append(f"FORBIDDEN_V3_LANGUAGE:{phrase}")
    if re.search(r"Supplementary Fig\\.?~S\d+", manuscript):
        errors.append("STALE_HARDCODED_SUPPLEMENT_FIGURE_NUMBER")
    if manuscript.count(r"\FloatBarrier") < 4:
        errors.append("RESULTS_FLOAT_BARRIERS_MISSING")
    if "conceptual motivation" not in manuscript.lower():
        errors.append("EARLY_RANKING_NOT_MARKED_CONCEPTUAL")
    if "beats both controls in only one seed" not in supplement:
        errors.append("PI_VARIABILITY_NOT_DISCLOSED")
    if "exact hash closure" not in supplement.lower():
        errors.append("ROSEWORTHY_EXACT_FILE_STATUS_MISSING")
    if "A high-error-tail analysis is not reported" not in supplement:
        errors.append("HIGH_ERROR_TAIL_PROVENANCE_DECISION_MISSING")
    manifest = json.loads((root / "figures/final_publication_v4/evidence_build_manifest.json").read_text(encoding="utf-8"))
    summary = manifest.get("summary", {})
    if summary.get("profile") != "v4":
        errors.append("EVIDENCE_PROFILE_MISMATCH")
    if summary.get("early_split_rows") != 14 or summary.get("early_split_reversals") != 9:
        errors.append("EARLY_SPLIT_SUMMARY_MISMATCH")
    if summary.get("beeswarm_rows") != 3100 or summary.get("agreement_cell_rows") != 44:
        errors.append("V4_EXPLANATION_SOURCE_MISMATCH")
    if summary.get("experiment_map_rows") != 9:
        errors.append("EXPERIMENT_MAP_SOURCE_MISMATCH")
    if summary.get("performance_landscape_rows") != 36 or summary.get("agreement_rows") != 30:
        errors.append("V2_HEADLINE_SOURCE_MISMATCH")
    if summary.get("ood_dimensions") != [9]:
        errors.append("OOD_DIMENSION_MISMATCH")
    if abs(summary.get("ig_pass_rate", 0) - 0.9893670886) > 1e-9:
        errors.append("IG_PASS_RATE_MISMATCH")
    taylor = summary.get("taylor_gate", {})
    if taylor.get("n_pass") != 28 or taylor.get("n_eligible") != 180:
        errors.append("TAYLOR_GATE_MISMATCH")
    figures_complete = True
    for number in range(1, 7):
        for suffix in (".pdf", ".svg", ".png"):
            path = root / f"figures/final_publication_v4/figure_{number:02d}{suffix}"
            if not path.is_file() or path.stat().st_size < 1000:
                figures_complete = False
                errors.append(f"MISSING_FIGURE:{path.name}")
    provenance = pd.read_csv(root / "figures/final_publication_v4/provenance_manifest.csv")
    if float(provenance["minimum_font_pt"].min()) < 8.0:
        errors.append("FIGURE_FONT_FLOOR_MISMATCH")
    beeswarm = pd.read_csv(root / "figures/final_publication_v4/source_supervised_shap_beeswarm.csv")
    if set(beeswarm["method"]) != {"supervised"}:
        errors.append("BEESWARM_ROUTE_MISMATCH")
    if "feature_value_colour" not in beeswarm or not beeswarm.loc[
        ~beeswarm["constant_within_contract"].astype(bool),
        "feature_value_colour",
    ].between(0.0, 1.0).all():
        errors.append("BEESWARM_COLOUR_SEMANTICS_INVALID")
    agreement = pd.read_csv(root / "figures/final_publication_v4/source_explanation_agreement_by_cell.csv")
    if set(agreement["comparison"]) != {"SHAP--IG", "SHAP--Taylor"}:
        errors.append("AGREEMENT_SOURCE_MISMATCH")
    try:
        main_pages = pdf_page_count(root / "paper/final_ajcai_manuscript_v4.pdf")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        errors.append(f"PDF_PAGE_COUNT_FAILURE:{exc}")
        main_pages = None
    if main_pages is not None and main_pages > 12:
        errors.append(f"PROVISIONAL_MAIN_PAGE_LIMIT_EXCEEDED:{main_pages}")
    return {"errors": errors, "citation_count": len(citations), "bib_key_count": len(bib_keys), "figures_complete": figures_complete, "agreement_cell_rows": int(len(agreement)), "beeswarm_rows": int(len(beeswarm)), "main_pages": main_pages}


def validate_v4_2(root: Path) -> dict[str, object]:
    """Check frozen evidence plus the reader-facing v4.1 boundary."""
    errors: list[str] = []
    required = (
        "paper/final_ajcai_manuscript_v4_2.tex",
        "paper/final_ajcai_manuscript_v4_2.pdf",
        "paper/final_ajcai_supplement_v4_2.tex",
        "paper/final_ajcai_supplement_v4_2.pdf",
        "paper/final_references_v4_2.bib",
        "figures/final_publication_v4_2/evidence_build_manifest.json",
        "figures/final_publication_v4_2/provenance_manifest.csv",
        "figures/final_publication_v4_2/captions.md",
        "figures/final_publication_v4_2/figure_01_layout_audit.json",
        "figures/final_publication_v4_2/source_scratch_baseline_status.csv",
    )
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"MISSING:{relative}")
    if errors:
        return {"errors": errors, "main_pages": None}
    main = (root / "paper/final_ajcai_manuscript_v4_2.tex").read_text(encoding="utf-8")
    supplement = (root / "paper/final_ajcai_supplement_v4_2.tex").read_text(encoding="utf-8")
    combined = main + "\n" + supplement
    bib = (root / "paper/final_references_v4_2.bib").read_text(encoding="utf-8")
    cited, keys = extract_citations(combined), extract_bibtex_keys(bib)
    if cited.difference(keys):
        errors.append(f"UNDEFINED_CITATIONS:{sorted(cited.difference(keys))}")
    if keys.difference(cited):
        errors.append(f"UNUSED_BIB_KEYS:{sorted(keys.difference(cited))}")
    forbidden_public = ("Stage 7", "Stage 8", "Codex", "ARS", "audit", "validator", "registry", "lineage", "hash", "trusted", "rebuilt", "outputs/", "run_")
    for term in forbidden_public:
        if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", combined, flags=re.IGNORECASE):
            errors.append(f"PUBLIC_LEAK:{term}")
    for seed in ("101", "202", "303"):
        if re.search(rf"(?<![0-9]){seed}(?![0-9])", main):
            errors.append(f"MAIN_SEED_ID_LEAK:{seed}")
    if "S1, S2, and S3" not in supplement:
        errors.append("SUPPLEMENT_SEED_MAPPING_MISSING")
    for phrase in ("primary objective", "fixed local reference", "one of the three runs showing the opposite direction"):
        if phrase.lower() not in main.lower():
            errors.append(f"REQUIRED_V42_TEXT_MISSING:{phrase}")
    manifest = json.loads((root / "figures/final_publication_v4_2/evidence_build_manifest.json").read_text())
    summary = manifest["summary"]
    if summary.get("profile") != "v4_2" or summary.get("spatial_scratch_baseline") != "fixed_reference_reused":
        errors.append("FROZEN_BASELINE_OR_PROFILE_MISMATCH")
    scratch = pd.read_csv(root / "figures/final_publication_v4_2/source_scratch_baseline_status.csv")
    if scratch.iloc[0]["classification"] != "fixed_reference_reused":
        errors.append("SPATIAL_SCRATCH_NOT_CLOSED")
    layout = json.loads((root / "figures/final_publication_v4_2/figure_01_layout_audit.json").read_text())
    if layout.get("containment") != "pass" or layout.get("font_min_pt", 0) < 8:
        errors.append("FIGURE1_LAYOUT_AUDIT_FAILED")
    experiment_map = pd.read_csv(root / "figures/final_publication_v4_2/source_complete_experiment_map.csv")
    public_columns = {"study_component", "dataset_and_sample_unit", "task_and_deployment_setting", "methods", "retained_result", "role", "reproducibility_note"}
    if set(experiment_map.columns) != public_columns:
        errors.append("EXPERIMENT_MAP_PUBLIC_FIELDS_MISMATCH")
    try:
        pages = pdf_page_count(root / "paper/final_ajcai_manuscript_v4_2.pdf")
        if pages > 12:
            errors.append(f"PROVISIONAL_PAGE_LIMIT_EXCEEDED:{pages}")
    except Exception as exc:
        pages = None
        errors.append(f"PDF_PAGE_COUNT_FAILURE:{exc}")
    return {"errors": errors, "citation_count": len(cited), "bib_key_count": len(keys), "main_pages": pages}


def validate_v4_4(root: Path) -> dict[str, object]:
    """Validate the frozen v4.4 paper, evidence sources, and public boundary."""
    errors: list[str] = []
    base = "figures/final_publication_v4_4"
    required = (
        "paper/final_ajcai_manuscript_v4_4.tex",
        "paper/final_ajcai_manuscript_v4_4.pdf",
        "paper/final_ajcai_supplement_v4_4.tex",
        "paper/final_ajcai_supplement_v4_4.pdf",
        "paper/final_references_v4_4.bib",
        f"{base}/evidence_build_manifest.json",
        f"{base}/provenance_manifest.csv",
        f"{base}/captions.md",
        f"{base}/figure_01_layout_audit.json",
        f"{base}/figure_01_design.json",
        f"{base}/source_scratch_baseline_status.csv",
        f"{base}/source_protocol_summary.csv",
        f"{base}/source_complete_experiment_map.csv",
    )
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"MISSING_OR_EMPTY:{relative}")
    if errors:
        return {"errors": errors, "main_pages": None}

    main = (root / "paper/final_ajcai_manuscript_v4_4.tex").read_text(encoding="utf-8")
    supplement = (root / "paper/final_ajcai_supplement_v4_4.tex").read_text(encoding="utf-8")
    combined = main + "\n" + supplement
    bib = (root / "paper/final_references_v4_4.bib").read_text(encoding="utf-8")
    cited, keys = extract_citations(combined), extract_bibtex_keys(bib)
    if cited.difference(keys):
        errors.append(f"UNDEFINED_CITATIONS:{sorted(cited.difference(keys))}")
    if keys.difference(cited):
        errors.append(f"UNUSED_BIB_KEYS:{sorted(keys.difference(cited))}")

    public_terms = (
        "Stage", "Codex", "ARS", "audit", "validator", "registry", "lineage",
        "worktree", "hash", "internal provenance", "private seed mapping",
        "outputs/", "run path", "output path", "Agent",
    )
    for term in public_terms:
        if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", combined, flags=re.IGNORECASE):
            errors.append(f"PUBLIC_INTERNAL_LANGUAGE:{term}")
    for seed in ("101", "202", "303"):
        if re.search(rf"(?<![0-9]){seed}(?![0-9])", main):
            errors.append(f"MAIN_SEED_ID_LEAK:{seed}")
    if not all(
        re.search(rf"{run}\$?=\$?{seed}", supplement)
        for run, seed in (("S1", "101"), ("S2", "202"), ("S3", "303"))
    ):
        errors.append("SUPPLEMENT_SEED_MAPPING_MISSING")

    forbidden = (
        "9/14", "9 of 14", "eight primary tests", "D_1", "D_{L1}",
        "D_L1", "Delta AE", "P_g", "outside represented support",
        "every transferred route loses", "governs", "causal importance",
        "explain the reversal", "target-training-only background", "[?]",
    )
    for phrase in forbidden:
        if phrase.lower() in combined.lower():
            errors.append(f"FORBIDDEN_OR_STALE_LANGUAGE:{phrase}")
    if re.search(r"\bprove(?:s|d)?\b", combined, flags=re.IGNORECASE):
        errors.append("FORBIDDEN_OR_STALE_LANGUAGE:prove")
    required_main = (
        "primary objective", "secondary analysis s1", "secondary analysis s2",
        "fixed local reference", "test-disjoint development background",
        "2001--2021", "2022", "2023", "0.954", "1.204", "1.157",
        "0.220", "-0.040", "98.9", "15.6", "d_{\\phi,i}",
        r"\Delta p", "model--target--deployment contract",
    )
    for phrase in required_main:
        if phrase.lower() not in main.lower():
            errors.append(f"REQUIRED_V44_TEXT_MISSING:{phrase}")

    manifest = json.loads((root / base / "evidence_build_manifest.json").read_text())
    summary = manifest.get("summary", {})
    expected_summary = {
        "profile": "v4_4",
        "performance_landscape_rows": 36,
        "agreement_rows": 30,
        "linkage_rows": 8,
        "source_protocol_rows": 9,
        "shap_background_test_rows": 0,
        "spatial_scratch_baseline": "fixed_reference_reused",
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            errors.append(f"EVIDENCE_SUMMARY_MISMATCH:{key}:{summary.get(key)}")
    if abs(summary.get("ig_pass_rate", 0.0) - 0.9893670886075949) > 1e-12:
        errors.append("IG_PASS_RATE_MISMATCH")
    taylor = summary.get("taylor_gate", {})
    if (taylor.get("n_pass"), taylor.get("n_eligible")) != (28, 180):
        errors.append("TAYLOR_GATE_MISMATCH")

    frozen_tables = (
        "source_performance_landscape_seed.csv",
        "source_support_delta_summary.csv",
        "source_linkage_summary.csv",
        "source_perturbation_agreement_disjoint.csv",
        "source_ood_quality_9d.csv",
    )
    for name in frozen_tables:
        left = pd.read_csv(root / "figures/final_publication_v4_3" / name)
        right = pd.read_csv(root / base / name)
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError:
            errors.append(f"FROZEN_SOURCE_MISMATCH:{name}")

    protocol = pd.read_csv(root / base / "source_protocol_summary.csv")
    source_counts = protocol[protocol["contract"].eq("SOURCE")].set_index("item")["background_rows"].to_dict()
    if source_counts != {"source_train_rows": 35282, "source_validation_rows": 1512, "source_test_rows": 1430}:
        errors.append(f"SOURCE_SPLIT_MISMATCH:{source_counts}")
    backgrounds = protocol[protocol["item"].eq("shap_background")]
    if len(backgrounds) != 6 or not backgrounds["background_rows"].eq(32).all() or not backgrounds["test_rows"].eq(0).all():
        errors.append("SHAP_BACKGROUND_PROTOCOL_MISMATCH")

    scratch = pd.read_csv(root / base / "source_scratch_baseline_status.csv")
    if scratch.iloc[0]["classification"] != "fixed_reference_reused":
        errors.append("SPATIAL_SCRATCH_NOT_FIXED_REFERENCE")
    experiment_map = pd.read_csv(root / base / "source_complete_experiment_map.csv")
    map_columns = {"study_component", "dataset_and_unit", "deployment_setting", "methods", "main_observation", "role_in_paper", "limitation"}
    if set(experiment_map.columns) != map_columns or len(experiment_map) != 9:
        errors.append("EXPERIMENT_MAP_READER_FIELDS_MISMATCH")
    if experiment_map.astype(str).apply(lambda col: col.str.contains("Stage|Tier|trusted|rebuilt|provenance", case=False)).any().any():
        errors.append("EXPERIMENT_MAP_INTERNAL_LANGUAGE")

    layout = json.loads((root / base / "figure_01_layout_audit.json").read_text())
    if layout.get("text_containment") != "verified" or layout.get("minimum_font_pt", 0) < 8 or layout.get("arrow_crossings") != 0:
        errors.append("FIGURE1_LAYOUT_FAILURE")
    design = json.loads((root / base / "figure_01_design.json").read_text())
    if design.get("conceptual_panel_contains_results") is not False:
        errors.append("FIGURE1_CONCEPTUAL_RESULT_LEAK")
    provenance = pd.read_csv(root / base / "provenance_manifest.csv")
    if provenance["minimum_font_pt"].min() < 8:
        errors.append("FIGURE_FONT_FLOOR_MISMATCH")
    if provenance.loc[provenance["figure"].eq("figure_01"), "source_key"].tolist() != ["conceptual_design"]:
        errors.append("FIGURE1_PROVENANCE_MISMATCH")
    for number in range(1, 7):
        for suffix in ("pdf", "svg", "png"):
            path = root / base / f"figure_{number:02d}.{suffix}"
            if not path.is_file() or path.stat().st_size < 1000:
                errors.append(f"MISSING_FIGURE:{path.name}")

    try:
        pages = pdf_page_count(root / "paper/final_ajcai_manuscript_v4_4.pdf")
        if pages > 12:
            errors.append(f"PROVISIONAL_PAGE_LIMIT_EXCEEDED:{pages}")
    except Exception as exc:
        pages = None
        errors.append(f"PDF_PAGE_COUNT_FAILURE:{exc}")
    return {
        "errors": errors,
        "citation_count": len(cited),
        "bib_key_count": len(keys),
        "main_pages": pages,
        "experiment_map_rows": int(len(experiment_map)),
        "source_protocol_rows": int(len(protocol)),
    }


def validate(root: Path, profile: str = "v1") -> dict[str, object]:
    if profile == "v2":
        return validate_v2(root)
    if profile == "v3":
        return validate_v3(root)
    if profile == "v4":
        return validate_v4(root)
    if profile == "v4_2":
        return validate_v4_2(root)
    if profile == "v4_4":
        return validate_v4_4(root)
    errors: list[str] = []
    for relative in REQUIRED:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"MISSING_OR_EMPTY:{relative}")
    if errors:
        return {
            "errors": errors,
            "citation_count": 0,
            "figures_complete": False,
        }
    manuscript = (root / "paper/final_ajcai_manuscript.tex").read_text(
        encoding="utf-8"
    )
    supplement = (root / "paper/final_ajcai_supplement.tex").read_text(
        encoding="utf-8"
    )
    bib_text = (root / "paper/final_references.bib").read_text(
        encoding="utf-8"
    )
    citations = extract_citations(manuscript + "\n" + supplement)
    bib_keys = extract_bibtex_keys(bib_text)
    missing_citations = sorted(citations.difference(bib_keys))
    if missing_citations:
        errors.append(f"UNDEFINED_CITATIONS:{missing_citations}")
    all_bib_occurrences = re.findall(
        r"@\w+\s*\{\s*([^,\s]+)\s*,",
        bib_text,
        flags=re.MULTILINE,
    )
    duplicates = sorted(
        key
        for key in set(all_bib_occurrences)
        if all_bib_occurrences.count(key) > 1
    )
    if duplicates:
        errors.append(f"DUPLICATE_BIB_KEYS:{duplicates}")
    for phrase in FORBIDDEN_MAIN:
        if phrase.lower() in manuscript.lower():
            errors.append(f"FORBIDDEN_MAIN_LANGUAGE:{phrase}")
    for token in ("[?]", r"\textbf{TODO}", "placeholder"):
        if token.lower() in (manuscript + supplement).lower():
            errors.append(f"UNRESOLVED_TOKEN:{token}")
    required_main = (
        "287 distinct target samples",
        "395 contract-seed rows",
        r"279 under \group{}",
        r"31 under \spatial{}",
        "23 appearing in both contracts",
        "0.954",
        "1.204",
        "1.157",
        "-0.040",
        "98.9",
        "15.6",
        "64 random feature-order permutations",
        "training scaler mean",
        "target adaptation never uses these kd losses",
        "joint-feature random forest",
    )
    for phrase in required_main:
        if phrase.lower() not in manuscript.lower():
            errors.append(f"MISSING_REQUIRED_FACT:{phrase}")
    if "only one seed" not in supplement.lower() and (
        "one of three seeds" not in supplement.lower()
    ):
        errors.append("PI_VARIABILITY_NOT_DISCLOSED")
    registry = pd.read_csv(
        root / "docs/final_codex_synthesis/evidence_and_method_registry.csv"
    )
    if registry["verification_status"].isin(
        ["SUMMARY_ONLY", "NOT_FOUND", "CONFLICTING"]
    ).any():
        errors.append("UNVERIFIED_FACT_IN_FINAL_REGISTRY")
    literature = pd.read_csv(
        root / "docs/final_codex_synthesis/literature_and_citation_audit.csv"
    )
    missing_literature_keys = sorted(
        set(literature["citation_key"]).difference(bib_keys)
    )
    if missing_literature_keys:
        errors.append(
            f"LITERATURE_KEYS_MISSING_FROM_BIB:{missing_literature_keys}"
        )
    manifest = json.loads(
        (
            root / "figures/final_publication/evidence_build_manifest.json"
        ).read_text(encoding="utf-8")
    )
    expected_counts = {
        "contract_seed_rows": 395,
        "unique_samples": 287,
        "GROUP_complete_unique_samples": 279,
        "SPATIAL_complete_unique_samples": 31,
        "cross_contract_overlap": 23,
    }
    if manifest["summary"]["counts"] != expected_counts:
        errors.append("COMMON_SAMPLE_COUNTS_MISMATCH")
    if manifest["summary"]["ood_dimensions"] != [9]:
        errors.append("OOD_DIMENSION_MISMATCH")
    figures_complete = True
    for number in range(1, 7):
        for suffix in (".pdf", ".svg", ".png"):
            path = root / f"figures/final_publication/figure_{number:02d}{suffix}"
            if not path.is_file() or path.stat().st_size < 1000:
                figures_complete = False
                errors.append(f"MISSING_FIGURE:{path.name}")
    lineage = pd.read_csv(
        root / "figures/final_publication/source_lineage_validation.csv"
    )
    if not lineage["status"].isin(["PASS", "NOT_APPLICABLE"]).all():
        errors.append("LINEAGE_VALIDATION_FAILURE")
    return {
        "errors": errors,
        "citation_count": len(citations),
        "bib_key_count": len(bib_keys),
        "figures_complete": figures_complete,
        "registry_rows": int(len(registry)),
        "lineage_status": lineage["status"].value_counts().to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--profile", choices=("v1", "v2", "v3", "v4", "v4_2", "v4_4"), default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.root.resolve(), profile=args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
