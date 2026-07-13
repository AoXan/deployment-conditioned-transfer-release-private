# Publication Release Scope

`configs/publication_release_manifest.yaml` is an allowlist. A release archive
must be assembled from its `include` entries and filtered by `exclude`; copying
the research worktree wholesale is prohibited.

Included material comprises the model definitions and losses, Stage 8 runners,
behavioural/statistical analyses, APSIM configuration and analysis code, figure
and table builders, clean public configuration templates, tests, schemas, and
documentation.

Excluded material comprises raw/derived restricted datasets, checkpoints,
row-level prediction bundles unless redistribution is separately authorised,
the APSIM source checkout, scratch directories, caches, logs, Git history, and
historical scripts outside the allowlist. APSIM is rebuilt from its pinned
upstream commit by `bootstrap_apsim.py`.

Before packaging, run:

```bash
rg -n '/Users/|OneDrive|Adelaide|AIML' \
  src/agritech_repro src/distillation_v4 src/distillation_program src/stage8 \
  src/stage8_phase3b configs/publication experiments/apsim_comparison \
  docs/reproducibility
```

Ignore the locally created `experiments/apsim_comparison/runtime/` checkout; it
is not part of the release. A clean allowlist archive should be scanned again
for credentials and absolute paths before publication.
