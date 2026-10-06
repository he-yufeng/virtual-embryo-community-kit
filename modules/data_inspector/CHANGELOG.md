# Changelog

## Unreleased

- Add optional `--fail-on-attention`: preserve generated reports, return `1` for
  existing material attention findings, and retain default `0`/error `2` behavior.
- Verify the option with seven synthetic HDF5 CLI cases and unchanged input bytes.

## Closeout G2/G3 — 2026-09-11

- Validate categorical celltype codes/categories before counting; reject malformed rank, dtype, length, and bounds without coercion or repair.
- Treat only categorical code `-1` as missing and distinguish invalid negative codes.
- Carry expression, axis, panel, spatial, and celltype attention findings into Markdown as well as JSON and terminal output.

## 0.1.1 — repair candidate

- Made malformed/missing/unsupported `X` layouts reportable instead of allowing dimension indexing failures.
- Added bounded CSR/CSC structural validation before numeric QC; invalid sparse storage is explicitly not assessed and never repaired.
- Declined complex expression and coordinate arrays rather than silently discarding imaginary components.
- Labeled sparse extrema as non-coalesced stored-entry statistics and retained conservative implicit-zero lower bounds.
- Surfaced skipped/invalid expression components, sparse failures, non-finite coordinates, and panel duplicates in terminal attention output.

## Post-review hardening — 2026-09-10

- Corrected sparse implicit-zero statistics; added axis/spatial consistency, bounded missingness, normalization/privacy/count caveats, local panel fingerprints, report collision protection, and concise terminal attention findings.
- Added explicit differentiation from existing general AnnData tools and expanded the unit suite with adversarial regression coverage.

## 0.1.0 — 2026-09-10

- First private review candidate.
- Added h5py-first, read-only inspection for common dense, CSR, and CSC AnnData HDF5 layouts.
- Added streaming local SHA-256 fingerprinting, chunked numeric QC, panel comparison, cell-type counts, spatial summaries, Markdown/JSON reports, and an optional reorder plan.
- Added deterministic synthetic fixture generator, notebook, and unit tests.
