# Known limitations

- This is a structural report, not a scorer, upload client, authorization decision, or provenance authenticator.
- The optional panel file is trusted only as a local input. Its origin, freshness, and board association are outside this tool.
- Common dense, CSR, and CSC HDF5 layouts are handled. Other AnnData encodings are described where possible and left unmaterialized.
- CSR/CSC validation is bounded and checks storage shape, arrays/dtypes, pointer consistency, and index bounds. It does not sort, repair, or coalesce duplicate coordinates.
- Sparse logical zero fraction is a lower bound derived from stored-entry count; extrema are stored-entry extrema augmented with zero when an implicit zero is guaranteed, not coalesced logical-matrix extrema.
- The bounded missingness scan covers directly stored one-dimensional fields and common categorical codes, up to 1,000,000 values per field. It is not a semantic missing-data audit.
- The source stability check detects size/mtime changes across the run. It cannot stop a different process from changing the file after the check completes.
- Report artifacts are created exclusively, so an existing output (including a dangling symlink) is never opened for writing. A collision after preflight fails the run; report creation is not a transaction across all artifacts and may leave files created earlier in that run. Use a fresh report directory rather than deleting or overwriting another writer's outputs.
- Cell-type summaries require a readable one-dimensional `obs/celltype` dataset or a common categorical layout with one-dimensional categories/codes, integer codes, matching expected length, legal in-range codes, and only `-1` as missing. Unsupported or malformed labels are reported but not repaired, coalesced, or counted.
- Spatial summaries require a real-valued numeric `n_obs × 3`-or-more dataset and summarize only the first three columns; they do not register coordinate frames or perform biological interpretation.
- Finite/non-negative QC cannot prove the challenge-required normalization scale.
- Reports made from challenge files can themselves contain unpublished or identifying data-derived details; keep them private unless current terms permit the intended sharing.
- JSON report values are intended for local automation, not as a stable long-term API.
