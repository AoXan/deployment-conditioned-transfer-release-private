# Farm Manager Visualisation Suite

Version `1.0.0` is a presentation-oriented visual package derived from the
frozen publication result sources. It separates three evidence classes:

- **Formal evidence**: values reported in the manuscript or Supplement.
- **Exploratory context**: public historical South Australian weather and yield.
- **Illustrative preview**: deterministic product mock-ups that are not farm
  predictions.

## Build

```bash
python -m venv .venv
.venv/bin/pip install -r visualisations/farm_manager_v1/requirements-visuals.txt
.venv/bin/python visualisations/farm_manager_v1/scripts/build_all.py
.venv/bin/python visualisations/farm_manager_v1/scripts/verify_outputs.py
```

All presentation assets are written to `exports/`. The self-contained gallery is
`exports/farm_manager_visual_gallery.html`. The visual source and output SHA256
hashes are recorded in `qa/source_manifest.json` and
`qa/output_manifest.json`.

## Recommended meeting sequence

1. **V02 — Start with the operating reality.** Seasons vary, so an average
   regional model is not enough for a farm decision.
2. **V03 — Show the central scientific result.** The same transfer route changes
   value when the deployment question changes.
3. **V04 — Connect the result to farm data.** Soil availability, missingness,
   learner, and training route jointly affect the preferred model.
4. **V05 — Position process modelling honestly.** Data-driven and APSIM results
   are complementary under constrained inputs; sensitivity prevents a general
   superiority claim.
5. **V06 — Explain trust checks.** Prediction, information reliance, stress
   response, and distance from familiar data answer different questions.
6. **V07/V10 — Show the potential product.** Use these only as clearly labelled
   previews of what local, georeferenced data could enable.
7. **V08 — End with a low-burden partnership.** Propose a staged pilot with
   useful outputs returned at every step.

V01 is a short alternative opening when the audience wants the delivery logic
before the research evidence. V09 can run silently while discussing long-term
season variability.

## Asset index

| ID | Asset | Primary message |
|---|---|---|
| V01 | Data-to-decision chain | Local data are checked before recommendations. |
| V02 | Long-term season context | Local season information changes the decision context. |
| V03 | Contract reversal | Model value depends on the deployment question. |
| V04 | Modality value | Input availability can change the preferred learning route. |
| V05 | APSIM comparison | Process and transferred predictors are complementary under constrained inputs. |
| V06 | Trust and scope | Error, reliance, stress, and unfamiliarity require separate checks. |
| V07 | 2D paddock preview | Prediction and confidence should be displayed separately. |
| V08 | Data partnership | Start with a small pilot and add data only when useful. |
| V09 | Animated context | Historical variability unfolds over time. |
| V10 | Interactive 3D preview | Terrain, a decision layer, and confidence can be explored together. |

## Interpretation boundaries

- Lower MAE under SPATIAL is a relative improvement; the corresponding mean
  absolute predictive quality remains limited.
- APSIM comparisons concern the locked constrained-input configurations.
  Paired intervals include zero and sensitivity settings alter ordering.
- Attribution and finite stress response are complementary model diagnostics,
  not causal agronomic effects.
- Paddock maps and the 3D scene are labelled illustrative. They show what local
  data could enable and must not be presented as Seabrook predictions.
