# Testing

The optional `--fail-on-attention` CLI path has seven synthetic HDF5 checks in
`tests/test_cli_attention.py`: clean output with and without the option,
unchanged default exit `0` for negative values, opt-in exit `1` with reports
preserved, non-finite and unsupported complex values, axis mismatch, wrong panel
order, and the existing unreadable-input exit `2`. Every invocation checks that
the input bytes are unchanged. These checks do not establish board eligibility,
normalization, or official acceptance.

Run from this directory after installing requirements:

```bash
pip install -e '.[test]'
PYTHONPATH=src python -m unittest discover -s tests -v
python examples/make_synthetic_fixtures.py --output-dir examples/generated
PYTHONPATH=src python -m vec_data_inspector.cli examples/generated/synthetic_sparse_spatial_t2_like.h5ad --report-dir reports/synthetic
```

The closeout G2/G3 check on 2026-09-11 used the available local Python runtime and ran 31 tests successfully. It includes actual h5py malformed categorical fixtures and CLI report checks. This is a historical result, not a substitute for running the public tests above.

Notebook users can install the separate Jupyter runtime with `pip install -e '.[notebook]'`; the CLI and unit suite do not require it.

The unit suite builds temporary synthetic HDF5 files and checks:

- dense non-finite and negative-value detection;
- sparse CSR summary with one-value chunks, without matrix expansion;
- logical sparse extrema/zero lower bounds for one-stored-value and all-zero matrices;
- preservation of source bytes, file size, and mtime through inspection;
- axis-index/spatial-row consistency and bounded obvious missingness;
- exact set/different order, missing, extra, and duplicate gene-name cases;
- a separate reorder-plan CSV;
- refusal to overwrite existing report artifacts;
- absent `celltype`, absent `spatial_3D`, and malformed spatial shape handling.
- readable malformed `X` dimensions through both API and CLI, plus clean CLI handling of an unreadable input;
- AnnData-written dense/CSR/CSC fixtures and h5py-crafted corrupt sparse layouts, with bounded CSR/CSC structure validation before numeric QC;
- complex expression/coordinate rejection without real coercion and duplicate-entry sparse-statistics labeling;
- terminal attention for non-finite spatial coordinates, unsupported/invalid `X`, sparse structure failures, and duplicate panel names;
- source SHA-256, size, and mtime invariance across successful inspection, malformed-input failure, and report-collision failure.
- the checked-in synthetic-only example report's absence of private absolute paths.
- valid string and categorical `celltype` encodings (including the documented `-1` missing-code sentinel), plus fractional/scalar/out-of-range/invalid-negative/wrong-length/rank-malformed categorical encodings without truncation or repair;
- celltype issue/status propagation across JSON, terminal attention, and Markdown while retaining safe string-label counts; and
- the documented policy that CLI exit `0` means a report was generated, not that its contents are structurally valid.

The local Jupyter notebook remains the only supported notebook path. Colab is not assessed or supported in this release.

The suite does not benchmark a production-scale file, test every historical AnnData encoding, authenticate a panel source, run a challenge scorer, or test a submission service. The synthetic files are deterministic and contain no challenge data.
