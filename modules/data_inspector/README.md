# VEC Data Inspector

VEC Data Inspector is a small, local, Virtual Embryo-oriented QC reporter for `.h5ad` files. It answers the practical first question before a workflow: what is stored in this file, and how does its structure compare with a gene-panel file I supplied?

It is aimed at Virtual Embryo Challenge participants who use Python or Jupyter and need a safe structural summary without loading an expression matrix into memory. It uses only `h5py` and NumPy; it does not depend on `anndata`, `scanpy`, or `veckit`.

It does not score predictions, decide whether a file can be submitted, authenticate provenance, fetch data, interpret biology, or change the input file.

## Five-minute quickstart

Create an environment and install the tiny dependency set:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Make entirely synthetic demonstration inputs (these are not challenge data):

```bash
python examples/make_synthetic_fixtures.py --output-dir examples/generated
```

Inspect the dense demonstration file and compare it with a deliberately reordered panel. Reports are written away from the input file:

```bash
vec-inspect examples/generated/synthetic_dense_t1_like.h5ad \
  --panel examples/generated/synthetic_panel_reordered.txt \
  --report-dir reports/dense-demo --write-reorder-plan
```

Expected terminal summary (the fingerprint will differ if the fixture differs):

```text
VEC Data Inspector: synthetic_dense_t1_like.h5ad
  local SHA-256 fingerprint: <local fingerprint>
  shape: 3 observations × 4 variables
  X: dense_dataset, dtype=float32, finite=True, negative=False
  celltype: present; spatial_3D: not found
  This is a read-only structural report, not a scorer or an authorization decision.
  Structural QC cannot establish the required expression normalization.
  attention:
    - Panel mismatch: missing=0, extra=0, wrong_order=True.
```

Open `reports/dense-demo/inspection_report.md`. The optional `reorder_plan.csv` is a source-index plan only; the inspector never rewrites the `.h5ad` file.
The writer also refuses to replace existing report artifacts; use a new report directory for each run.

To inspect your own file, replace the input and use a new report directory:

```bash
vec-inspect /path/to/file.h5ad --report-dir /path/to/inspection-report
```

Reports made from real challenge files may contain unpublished or data-derived details, absolute local paths, fingerprints, label counts, or spatial summaries. Keep those reports private unless the current terms permit the intended sharing and any required approval and credit have been obtained. Review/redact local paths before any permitted sharing. Use only the bundled synthetic fixtures for public screenshots and examples.

For sparse shape-coverage demonstrations, generate two additional modest files on demand. They contain fabricated feature names, values, labels, and coordinates only; they are generated locally and are ignored by version control:

```bash
python examples/make_synthetic_fixtures.py --output-dir examples/generated --extended
```

This adds `synthetic_sparse_2048_genes_no_spatial.h5ad` (8 × 2,048) and `synthetic_sparse_500_genes_spatial.h5ad` (10 × 500, with `spatial_3D`).

For a compact, shareable output preview without generating a local report, see [`examples/synthetic_inspection_report.md`](examples/synthetic_inspection_report.md) and the matching JSON. They were generated only from a bundled synthetic dense fixture and use package-relative paths; they contain no challenge or private local data.

## What it reports

- Local identity: absolute path, name, byte size, UTC timestamp, and a streaming SHA-256 **local fingerprint**. A local fingerprint is not an authenticated checksum.
- AnnData HDF5 structure: `X` shape, storage representation/dtype, `obs`, `var`, `obsm`, `layers`, and `uns` keys; axis-index and spatial-row consistency; and a bounded obvious-missingness scan for common one-dimensional fields.
- Expression QC without full matrix densification: finite-value status, negative values, finite min/max, and sparsity. Dense arrays are read in row blocks. CSR/CSC stored values are read in blocks; reported extrema include a guaranteed implicit zero, and the logical zero fraction is labeled as a lower bound because explicit zeros or duplicate indices can increase it.
- Optional gene-panel comparison: exact ordered match, same unique set in a different order, missing genes, extra genes, and duplicate names. It can produce a separate source-index plan but never reorders the source.
- Cell-type counts and fractions when a readable `obs/celltype` field exists. Common categorical encodings are validated before counting: one-dimensional code/category arrays, integer codes, expected code length, `-1` as the only missing sentinel, and in-range non-missing codes. Invalid labels are reported as invalid/not assessed; they are never truncated, relabeled, or repaired.
- `obsm/spatial_3D` shape, finite status, coordinate bounds, centroid, variance, and extent when present.
- A concise terminal summary plus JSON and Markdown reports.

## Challenge-facing use, carefully scoped

The challenge documents publish board-specific panel files and machine-readable metadata. Give this tool a panel file you have obtained and checked yourself; it will compare the supplied file's `/var` index against that exact local text file and record a local panel fingerprint. It intentionally does not download a panel or label it as authenticated.

In the combined kit, Contribution A fetches and caches the current `T1:val` panel over HTTPS, recording the timestamp, source URLs, and the normalized panel hash prefix published in the fetched index. After reviewing its `provenance.json`, hand that exact file to the inspector. This is a transport/integrity check, not organizer authentication, certificate pinning, or provenance attestation:

```bash
vec-t1-walkthrough fetch-contract --cache official_contract_cache
vec-inspect /path/to/file.h5ad \
  --panel official_contract_cache/T1__val.genes.txt \
  --report-dir reports/t1-current-contract
```

For other boards, obtain the named panel from the current official panel index and preserve its source/date separately. The inspector does not infer a board from dimensions.

Useful primary references (accessed 2026-09-20):

- [Challenge data and panel guidance](https://virtualembryo.ai/challenge/data)
- [Panel metadata index](https://virtualembryo.ai/challenge/panels/index.json)
- [Submission-format guidance](https://virtualembryo.ai/challenge/account/submissions)
- [Rules](https://virtualembryo.ai/challenge/rules)
- [FAQ and dataset-use restrictions](https://virtualembryo.ai/challenge/faq)

The report does not say a file is acceptable for a board. Current requirements can change, format requirements are only one part of the workflow, and any question about submission, data use, presentation, or redistribution belongs with the challenge documentation and organizers.

## Related tools and differentiation

[`cellgeni/h5ad-cli`](https://github.com/cellgeni/h5ad-cli) already provides general streaming exploration of large `.h5ad` and `.zarr` stores, including read-only layout inspection and dense/sparse chunking. [`vals/anndata-design-inspector`](https://github.com/vals/anndata-design-inspector) extracts categorical experimental-design metadata and deliberately adds biological-context reasoning. VEC Data Inspector does not claim either general capability as novel and deliberately avoids biological inference. Its narrower purpose is a Virtual Embryo-oriented, non-mutating inspection record: expression finiteness/negativity/sparsity QC, optional exact ordered-panel comparison and reorder planning, a SHA-256 explicitly labeled as a local fingerprint, stage-safe cell-label and spatial warnings, Markdown/JSON reports, and tests that verify source bytes remain unchanged. If you need general store navigation, export, import, subsetting, or experimental-design inference, use the broader tools; if you need this VEC-specific descriptive preflight, use this one.

## Important interpretation limits

When a cell-type column is present, the report states only its stored labels and counts. Label vocabularies may not be harmonized across developmental stages. Absence of a label is not proof of biological absence.

When `spatial_3D` is present, the inspector accepts `n_obs × 3` or more columns and summarizes only the first three, matching the current upload guidance; any extra columns are reported as ignored. Coordinates are per-embryo local and are not registered across time points. Do not interpret raw frame differences as developmental displacement or growth without an appropriate model.

For a submission, `n_obs` is a sample-size choice subject to the selected board's current limits. It is not a measurement or prediction of embryo size, and the inspector makes no developmental inference from it.

Finite and non-negative expression values do not establish the normalization scale. Raw counts or another non-negative scale can pass these QC checks and still be unsuitable for scoring; confirm the current official preprocessing contract yourself.

For sparse matrices, sparsity is an estimate based on stored entries: explicit zeros or duplicate sparse indices can change logical sparsity. For unusual HDF5 encodings, the tool reports the layout and declines numeric QC rather than materializing it.

`X` must be a readable two-dimensional dense dataset or a common CSR/CSC group. A readable file with a missing, scalar, one-dimensional, higher-dimensional, or unsupported `X` still receives a report with expression QC marked invalid or not assessed; it is not reported as clean. CSR/CSC storage is checked in bounded slices for the required one-dimensional arrays, integer pointers/indices, shape, pointer length/start/order/final value, matching stored lengths, and index bounds. If that validation fails, numeric and logical-matrix QC are not assessed. The inspector does not repair, sort, or coalesce sparse entries.

Complex-valued expression values and spatial coordinates are explicitly declined: the report does not discard imaginary components to manufacture real-valued summaries. For structurally valid sparse layouts, duplicate coordinates are not coalesced. Sparse minima/maxima are therefore labeled as stored-entry extrema (with a guaranteed implicit zero when applicable), not logical-matrix extrema. The implicit-zero fraction remains a conservative lower bound.

## Malformed-input CLI policy

For an automation pipeline, opt into `--fail-on-attention`:

```bash
vec-inspect /path/to/file.h5ad --report-dir /path/to/new-report --fail-on-attention
```

The default behavior below is unchanged. With this flag, generated reports are
preserved, but the command exits `1` when the same material `attention` findings
already shown in the terminal/Markdown report are present. An unreadable input
or report-writing error still exits `2`. Exit `0` with the flag means only that
the tool reported no attention findings; it does not establish normalization,
biological quality, board eligibility, or official submission acceptance. It
must not be used to automatically authorize an upload.

For a readable HDF5 file with a malformed but inspectable layout, `vec-inspect` writes the normal Markdown/JSON report and exits `0`; inspect the terminal `attention` section and report fields marked `invalid` or `not assessed`. **Exit `0` means only that the report was generated, not that the file or any component is valid.** For an unreadable/non-HDF5 input, inaccessible file, invalid command arguments, a mid-run source change, or report-output collision, it prints a concise `vec-inspect:` diagnostic to stderr, writes no new report artifacts, and exits `2`. Expected malformed inputs do not produce a traceback.

## Read-only behavior

This process opens the input in HDF5 mode `r`. Before and after inspection it checks the source file's size and nanosecond mtime; if another process changes it mid-run, it stops before writing reports. This does not provide OS-level immutability or control other processes. Generated artifacts are only the requested Markdown/JSON files and, when requested, a CSV plan in the separate report directory.

## Notebook

[`notebooks/vec_data_inspector_demo.ipynb`](notebooks/vec_data_inspector_demo.ipynb) follows the same synthetic quickstart. It is deliberately small, and executes the command-line tool instead of embedding a second implementation.

Install and launch the optional notebook runtime from this project directory:

```bash
pip install -e '.[notebook]'
jupyter lab
```

Local Jupyter after this local install is the only supported notebook path. Google Colab is not assessed or supported. No Colab bootstrap is included.

For a non-interactive reproducibility run from the project directory:

```bash
jupyter nbconvert --to notebook --execute notebooks/vec_data_inspector_demo.ipynb \
  --output /tmp/vec_data_inspector_demo.executed.ipynb --ExecutePreprocessor.timeout=180
```

## Development

From this module directory, with your virtual environment activated, install
the test dependencies before running the suite:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

See [TESTING.md](TESTING.md), [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), [CHANGELOG.md](CHANGELOG.md), and the repository-level [license](../../LICENSE) and [attribution](../../ATTRIBUTION.md).
