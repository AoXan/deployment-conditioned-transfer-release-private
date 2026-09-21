# Data Access and Licence Boundaries

The public code does not bundle raw agricultural data or frozen checkpoints.
`data_manifest/publication_inputs.yaml` is the machine-readable source list.

## Expected layout

```text
data/external/
  cybench/
    raw/
      yield_maize_US.csv
      meteo_maize_US.csv
      crop_calendar_maize_US.csv
      soil_maize_US.csv
      location_maize_US.csv
      crop_mask_maize_US.csv
      yield_wheat_AU.csv
      meteo_wheat_AU.csv
      crop_calendar_wheat_AU.csv
      soil_wheat_AU.csv
      location_wheat_AU.csv
      crop_mask_wheat_AU.csv
    PRIMARY_CYBENCH_MAIZE_US.csv.gz
    CY-Bench_wheat_AU.csv.gz
  g2f/
    g2f_native_hybrid_env.csv.gz
    g2f_native_hybrid_env__temporal_forward__test_2022.csv
  roseworthy/
    roseworthy_e5_point_yield.csv.gz
  waite/
    Waite_Trial_Data.xls
    waite_full_environment_proxy.csv.gz
```

CY-Bench is pinned to Zenodo DOI `10.5281/zenodo.17279151`. Its metadata can be
checked without downloading data:

```bash
python3 scripts/acquire_publication_data.py --dataset cybench --validate-only
```

Add `--download` only after checking the record terms. The command does not
download G2F or Roseworthy. Those sources require file-specific access or
provenance review; it exits non-zero when asked to download them.

The acquisition command verifies the checksum published in the Zenodo record
and extracts the archive under `cybench/raw`. The adapter accepts the archive's
own single-directory nesting when the six tables for each crop-country pair
share a directory. Build the two harmonised views used by the formal campaign
with:

```bash
python -m agritech_repro preprocess --data-root /path/to/authorised/data
```

The command calls the publication adapter in
`scripts/prepare_publication_views.py`, writes the two compressed views shown
above, and materialises a path-resolved copy of the exact campaign config under
`outputs/publication_repro/configs/`. Use `--validate-only` to check all 12 raw
files and the resolved config without writing views.

The G2F citation is DOI `10.1186/s13104-020-4922-8`. The Roseworthy public record
is DOI `10.25909/19158419.v2`, but neither of the two cleaned point files used by
the accepted study appears in the official v1 or v2 file inventory. They are
therefore not redistributed. An authorised copy must match SHA-256
`eb0613052d02510ef81093f2c21751f24be00e726b9d856516f78794091201a9` and
`53c24da8d42655c23da2ee5afc67de523cc93b48446d4bac30dba563a5c7e513`.

The Waite source is DOI `10.4225/08/55E5165EC0D29`. The official CSIRO version
4 API reports `Waite_Trial_Data.xls` as publicly accessible under the Creative
Commons Attribution 4.0 International Licence. Acquire and checksum it with:

```bash
python3 scripts/acquire_publication_data.py --dataset waite --validate-only
python3 scripts/acquire_publication_data.py --dataset waite --download
```

The accepted workbook SHA-256 is
`a5b1b7f4c943a6533917e3b7c4fe51eaa030c77b9777cd34c0864ca3c8961c29`.
The downloader fails if the DOI, licence, filename, or checksum changes.

For exact checkpoint replay, unpack the authorised bundle under a directory and
set:

```bash
export AGRITECH_CHECKPOINT_ROOT=/path/to/checkpoints
export AGRITECH_PREDICTION_ROOT=/path/to/prediction_bundle
```

The prediction bundle convention is either
`$AGRITECH_PREDICTION_ROOT/<candidate_id>/predictions.csv` or
`$AGRITECH_PREDICTION_ROOT/jobs/targets/<candidate_id>/predictions.csv`.
Checksums can be verified without exposing paths:

```bash
python3 scripts/verify_input_checksums.py --manifest authorised.sha256 --root /path/to/bundle
```

`tests/fixtures/publication/primary_fixture.csv` is synthetic and verifies the
public interface only. It is not a scientific substitute for any paper dataset.
