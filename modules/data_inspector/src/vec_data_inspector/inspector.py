"""Read-only, h5py-first inspection of common AnnData-on-disk layouts.

This module deliberately does not import anndata or a scoring package.  It only
opens the input file with h5py in read mode and processes numeric arrays in
bounded row/data chunks.  It is a structural report, not a challenge checker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


TOOL_NAME = "VEC Data Inspector"
REPORT_SCHEMA_VERSION = "1.1"
MISSINGNESS_SCAN_LIMIT_PER_FIELD = 1_000_000


def _plain(value: Any) -> Any:
    """Convert NumPy and h5py-friendly values into JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _attr_text(obj: h5py.Group | h5py.Dataset, name: str) -> str | None:
    value = obj.attrs.get(name)
    return _text(value) if value is not None else None


def _sha256_stream(path: Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _require_unchanged(path: Path, before: dict[str, Any]) -> None:
    after = _file_snapshot(path)
    if after != before:
        raise RuntimeError(
            "The source file changed while it was being inspected. No report was written; "
            "run again when the source is stable."
        )


def _member_kind(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    if isinstance(node, h5py.Dataset):
        return {
            "kind": "dataset",
            "shape": list(node.shape),
            "dtype": str(node.dtype),
            "encoding_type": _attr_text(node, "encoding-type"),
        }
    return {
        "kind": "group",
        "keys": sorted(node.keys()),
        "encoding_type": _attr_text(node, "encoding-type"),
    }


def _group_inventory(handle: h5py.File, name: str) -> dict[str, Any]:
    node = handle.get(name)
    if node is None:
        return {"present": False, "keys": []}
    if not isinstance(node, h5py.Group):
        return {"present": True, "kind": "dataset", "keys": []}
    return {
        "present": True,
        "keys": sorted(node.keys()),
        "members": {key: _member_kind(node[key]) for key in sorted(node.keys())},
    }


def _axis_index_name(axis: h5py.Group) -> str | None:
    declared = axis.attrs.get("_index")
    if declared is not None:
        candidate = _text(declared)
        if candidate in axis:
            return candidate
    for candidate in ("_index", "index"):
        if candidate in axis:
            return candidate
    return None


def _axis_length(axis: h5py.Group | None, index_name: str | None) -> int | None:
    if not isinstance(axis, h5py.Group) or index_name is None:
        return None
    node = axis.get(index_name)
    return int(node.shape[0]) if isinstance(node, h5py.Dataset) and node.ndim == 1 else None


def _missing_count(dataset: h5py.Dataset, chunk_items: int) -> dict[str, Any]:
    total = int(dataset.shape[0])
    scanned = min(total, MISSINGNESS_SCAN_LIMIT_PER_FIELD)
    missing = 0
    for start in range(0, scanned, chunk_items):
        values = np.asarray(dataset[start : min(scanned, start + chunk_items)]).reshape(-1)
        if np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.complexfloating):
            missing += int(np.count_nonzero(~np.isfinite(values)))
        elif values.dtype.kind in {"O", "S", "U"}:
            missing += sum(_text(value).strip() == "" for value in values)
    return {"missing_in_scanned": missing, "scanned": scanned, "total": total, "complete_scan": scanned == total}


def _axis_missingness(axis: h5py.Group | None, chunk_items: int) -> dict[str, Any]:
    if not isinstance(axis, h5py.Group):
        return {"present": False, "fields": {}}
    fields: dict[str, Any] = {}
    for name in sorted(axis.keys()):
        node = axis[name]
        if isinstance(node, h5py.Dataset) and node.ndim == 1:
            fields[name] = _missing_count(node, chunk_items)
        elif isinstance(node, h5py.Group) and isinstance(node.get("codes"), h5py.Dataset):
            codes = node["codes"]
            if codes.ndim == 1 and np.issubdtype(codes.dtype, np.integer):
                scanned = min(int(codes.shape[0]), MISSINGNESS_SCAN_LIMIT_PER_FIELD)
                missing = 0
                invalid_negative = 0
                for start in range(0, scanned, chunk_items):
                    values = codes[start : min(scanned, start + chunk_items)]
                    missing += int(np.count_nonzero(values == -1))
                    invalid_negative += int(np.count_nonzero(values < -1))
                fields[name] = {
                    "missing_in_scanned": missing,
                    "invalid_negative_codes_in_scanned": invalid_negative,
                    "scanned": scanned,
                    "total": int(codes.shape[0]),
                    "complete_scan": scanned == int(codes.shape[0]),
                    "encoding": "categorical_codes",
                }
    return {"present": True, "scan_limit_per_field": MISSINGNESS_SCAN_LIMIT_PER_FIELD, "fields": fields}


def _read_strings(node: h5py.Dataset) -> list[str]:
    raw = node[()]
    values = np.asarray(raw).reshape(-1)
    return [_text(value) for value in values]


def _var_names(handle: h5py.File) -> tuple[list[str], str | None, str | None]:
    var = handle.get("var")
    if not isinstance(var, h5py.Group):
        return [], None, "Missing /var group; gene-name comparison was skipped."
    index_name = _axis_index_name(var)
    if index_name is None or not isinstance(var[index_name], h5py.Dataset):
        return [], index_name, "Could not find a readable /var index dataset; gene-name comparison was skipped."
    return _read_strings(var[index_name]), index_name, None


def _numeric_summary_chunks(chunks: Iterator[np.ndarray], total_elements: int) -> dict[str, Any]:
    finite = True
    negative = False
    finite_count = 0
    zero_count = 0
    min_value: float | None = None
    max_value: float | None = None
    issue: str | None = None
    for block in chunks:
        array = np.asarray(block)
        if not np.issubdtype(array.dtype, np.number):
            issue = f"Expression values have non-numeric dtype {array.dtype}; numeric QC was skipped."
            break
        if np.issubdtype(array.dtype, np.complexfloating):
            issue = (
                f"Expression values have non-real complex dtype {array.dtype}; numeric QC was skipped "
                "rather than discarding imaginary components."
            )
            break
        block_finite = np.isfinite(array)
        finite = finite and bool(np.all(block_finite))
        finite_values = array[block_finite]
        finite_count += int(finite_values.size)
        if finite_values.size:
            block_min = float(np.min(finite_values))
            block_max = float(np.max(finite_values))
            min_value = block_min if min_value is None else min(min_value, block_min)
            max_value = block_max if max_value is None else max(max_value, block_max)
            negative = negative or bool(np.any(finite_values < 0))
            zero_count += int(np.count_nonzero(finite_values == 0))
    summary: dict[str, Any] = {
        "numeric_qc_performed": issue is None,
        "finite": finite if issue is None else None,
        "negative_values_detected": negative if issue is None else None,
        "min_finite": min_value,
        "max_finite": max_value,
        "finite_values_seen": finite_count,
        "total_elements": total_elements,
        "issue": issue,
    }
    if issue is None and total_elements:
        summary["zero_fraction"] = zero_count / total_elements
        summary["sparsity_estimate"] = zero_count / total_elements
    return summary


def _slice_chunks(dataset: h5py.Dataset, rows: int) -> Iterator[np.ndarray]:
    if dataset.ndim != 2:
        return
    for start in range(0, dataset.shape[0], rows):
        yield dataset[start : min(dataset.shape[0], start + rows), :]


def _data_chunks(dataset: h5py.Dataset, items: int) -> Iterator[np.ndarray]:
    for start in range(0, dataset.shape[0], items):
        yield dataset[start : min(dataset.shape[0], start + items)]


def _not_assessed_expression(issue: str, **details: Any) -> dict[str, Any]:
    """Return a uniform, explicitly non-reassuring skipped-QC result."""
    return {
        **details,
        "numeric_qc_performed": False,
        "finite": None,
        "negative_values_detected": None,
        "min_finite": None,
        "max_finite": None,
        "issue": issue,
    }


def _append_issue(existing: str | None, addition: str) -> str:
    """Preserve earlier independent findings when a component has several faults."""
    return addition if not existing else f"{existing} {addition}"


def _sparse_shape(group: h5py.Group) -> tuple[list[int] | None, str | None]:
    """Read only a well-formed two-element non-negative integer shape attribute."""
    raw_shape = group.attrs.get("shape")
    if raw_shape is None:
        return None, "Sparse /X lacks a shape attribute."
    values = np.asarray(raw_shape)
    if values.ndim != 1 or values.size != 2:
        return None, "Sparse /X shape must be a one-dimensional two-element attribute."
    if not np.issubdtype(values.dtype, np.integer):
        return None, "Sparse /X shape must use an integer dtype."
    shape = [int(value) for value in values]
    if any(value < 0 for value in shape):
        return None, "Sparse /X shape dimensions must be non-negative."
    return shape, None


def _sparse_structure_summary(
    *,
    data: h5py.Dataset | h5py.Group,
    indices: h5py.Dataset | h5py.Group,
    indptr: h5py.Dataset | h5py.Group,
    shape: list[int] | None,
    encoding: str,
    chunk_items: int,
) -> dict[str, Any]:
    """Check CSR/CSC array invariants in one-dimensional bounded slices.

    This deliberately validates storage only.  It neither sorts nor coalesces
    sparse entries, so corrupt data is never reinterpreted as a matrix.
    """
    issues: list[str] = []
    checks = [
        "data/indices/indptr are datasets",
        "data/indices/indptr are one-dimensional",
        "indices and indptr use integer dtypes",
        "data and indices lengths agree",
        "indptr length/start/monotonicity/final value",
        "indices are in bounds",
    ]
    if not isinstance(data, h5py.Dataset) or not isinstance(indices, h5py.Dataset) or not isinstance(indptr, h5py.Dataset):
        return {"performed": True, "valid": False, "checks": checks, "issues": ["Sparse /X data, indices, and indptr must each be datasets."]}
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        issues.append("Sparse /X data, indices, and indptr must each be one-dimensional.")
    if not np.issubdtype(indices.dtype, np.integer):
        issues.append(f"Sparse /X indices dtype must be integer, not {indices.dtype}.")
    if not np.issubdtype(indptr.dtype, np.integer):
        issues.append(f"Sparse /X indptr dtype must be integer, not {indptr.dtype}.")
    if issues or shape is None:
        return {"performed": True, "valid": False, "checks": checks, "issues": issues or ["Sparse /X shape is invalid."]}

    stored_entries = int(data.shape[0])
    if int(indices.shape[0]) != stored_entries:
        issues.append(
            f"Sparse /X data has length {stored_entries} but indices has length {int(indices.shape[0])}."
        )
    major_dimension = shape[0] if encoding.startswith("csr") else shape[1]
    minor_dimension = shape[1] if encoding.startswith("csr") else shape[0]
    expected_indptr_length = major_dimension + 1
    if int(indptr.shape[0]) != expected_indptr_length:
        issues.append(
            f"Sparse /X indptr has length {int(indptr.shape[0])}; {encoding} shape {shape} requires {expected_indptr_length}."
        )
    # Do not index an empty or malformed pointer array.  The remaining checks
    # use bounded slices and retain only a single prior value between slices.
    if int(indptr.shape[0]):
        first = int(indptr[0])
        last = int(indptr[-1])
        if first != 0:
            issues.append(f"Sparse /X indptr must start at 0, not {first}.")
        if last != stored_entries:
            issues.append(f"Sparse /X indptr final value {last} does not match stored entry count {stored_entries}.")
        previous: int | None = None
        for start in range(0, int(indptr.shape[0]), chunk_items):
            values = np.asarray(indptr[start : min(int(indptr.shape[0]), start + chunk_items)])
            if values.size and np.any(values < 0):
                issues.append("Sparse /X indptr contains a negative pointer.")
                break
            if previous is not None and values.size and int(values[0]) < previous:
                issues.append("Sparse /X indptr is not monotonic non-decreasing.")
                break
            if values.size > 1 and np.any(values[1:] < values[:-1]):
                issues.append("Sparse /X indptr is not monotonic non-decreasing.")
                break
            if values.size:
                previous = int(values[-1])
    if not issues and minor_dimension >= 0:
        for start in range(0, int(indices.shape[0]), chunk_items):
            values = np.asarray(indices[start : min(int(indices.shape[0]), start + chunk_items)])
            if values.size and (np.any(values < 0) or np.any(values >= minor_dimension)):
                issues.append(
                    f"Sparse /X {encoding} indices must be in [0, {minor_dimension}); an out-of-bounds index was found."
                )
                break
    return {"performed": True, "valid": not issues, "checks": checks, "issues": issues}


def _expression_summary(handle: h5py.File, chunk_rows: int, chunk_items: int) -> dict[str, Any]:
    x = handle.get("X")
    if x is None:
        return _not_assessed_expression("Missing /X.", present=False, assessment="invalid")
    if isinstance(x, h5py.Dataset):
        result = {
            "present": True,
            "storage": "dense_dataset",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "chunks": list(x.chunks) if x.chunks else None,
        }
        if x.ndim != 2:
            return _not_assessed_expression(
                f"/X is {x.ndim}-dimensional; this inspector expects a 2D expression matrix.",
                **result,
                assessment="invalid",
            )
        result.update(_numeric_summary_chunks(_slice_chunks(x, chunk_rows), int(np.prod(x.shape))))
        result["chunking"] = f"Read dense rows in blocks of at most {chunk_rows}."
        result["assessment"] = "assessed" if result.get("numeric_qc_performed") else "not_assessed"
        return result
    if not isinstance(x, h5py.Group):
        return _not_assessed_expression("Unsupported /X object.", present=True, assessment="invalid")
    encoding = _attr_text(x, "encoding-type")
    if encoding not in {"csr_matrix", "csc_matrix", "csr_array", "csc_array"}:
        return _not_assessed_expression(
            "Unsupported /X group encoding; numeric QC was skipped without materializing it.",
            present=True,
            storage="group",
            encoding_type=encoding,
            assessment="not_assessed",
        )
    if not all(key in x for key in ("data", "indices", "indptr")):
        return _not_assessed_expression(
            "Sparse /X group lacks data, indices, or indptr.",
            present=True,
            storage="sparse_group",
            encoding_type=encoding,
            assessment="invalid",
        )
    shape, shape_issue = _sparse_shape(x)
    data = x["data"]
    indices = x["indices"]
    indptr = x["indptr"]
    if shape is None:
        return _not_assessed_expression(
            shape_issue or "Sparse /X lacks a valid 2D shape.",
            present=True,
            storage="sparse_group",
            encoding_type=encoding,
            assessment="invalid",
        )
    stored_entries = int(data.shape[0]) if isinstance(data, h5py.Dataset) and data.ndim == 1 else None
    result = {
        "present": True,
        "storage": "sparse_group",
        "encoding_type": encoding,
        "shape": shape,
        "dtype": str(data.dtype) if isinstance(data, h5py.Dataset) else "unknown",
        "stored_entries": stored_entries,
        "chunking": f"Read sparse data values in blocks of at most {chunk_items}; indices were not expanded.",
    }
    structural = _sparse_structure_summary(
        data=data, indices=indices, indptr=indptr, shape=shape, encoding=encoding, chunk_items=chunk_items
    )
    result["sparse_structure"] = structural
    if not structural["valid"] or not isinstance(data, h5py.Dataset) or stored_entries is None:
        return _not_assessed_expression(
            "Sparse /X structural validation failed; numeric and logical-matrix QC were not assessed. "
            + " ".join(structural["issues"]),
            **result,
            assessment="invalid",
        )
    result.update(_numeric_summary_chunks(_data_chunks(data, chunk_items), stored_entries))
    result["assessment"] = "assessed" if result.get("numeric_qc_performed") else "not_assessed"
    matrix_elements = int(shape[0]) * int(shape[1])
    result["matrix_elements"] = matrix_elements
    if matrix_elements and result.get("numeric_qc_performed"):
        result["stored_value_zero_fraction"] = result.pop("zero_fraction", None)
        guaranteed_implicit_zeros = max(0, matrix_elements - stored_entries)
        result["logical_zero_fraction_lower_bound"] = guaranteed_implicit_zeros / matrix_elements
        result["sparsity_estimate"] = guaranteed_implicit_zeros / matrix_elements
        if guaranteed_implicit_zeros:
            current_min, current_max = result.get("min_finite"), result.get("max_finite")
            result["min_finite"] = 0.0 if current_min is None else min(0.0, current_min)
            result["max_finite"] = 0.0 if current_max is None else max(0.0, current_max)
        result["sparsity_note"] = (
            "Logical zero fraction is a lower bound derived from unstored positions; explicit stored zeros or "
            "duplicate sparse indices can make the true zero fraction larger. Min/max are stored-entry extrema "
            "augmented with a guaranteed implicit zero when one exists; duplicate coordinates are not coalesced, "
            "so these are not logical-matrix extrema."
        )
        result["sparse_numeric_semantics"] = "stored_entry_statistics_not_coalesced_logical_matrix_statistics"
    return result


def _invalid_celltype_summary(
    issue: str,
    expected: int | None,
    *,
    encoding: str,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe corrupt cell labels without fabricating a label summary."""
    return {
        "present": True,
        "status": "invalid",
        "assessment": "invalid",
        "encoding": encoding,
        "counted_observations": None,
        "expected_observations": expected,
        "matches_n_obs": None,
        "labels": [],
        "issue": issue,
        "validation": validation or {"performed": True, "valid": False},
        "caveat": "No cell-type counts were produced from an invalid encoding. Label vocabularies may not be harmonized across developmental stages.",
    }


def _count_celltypes(obs: h5py.Group, expected: int | None, chunk_items: int) -> dict[str, Any]:
    if "celltype" not in obs:
        return {"present": False, "status": "not_present", "warning": "No /obs/celltype field was found."}
    node = obs["celltype"]
    counts: Counter[str] = Counter()
    if isinstance(node, h5py.Dataset):
        if node.ndim != 1:
            return _invalid_celltype_summary(
                "/obs/celltype is not one-dimensional; cell-type counts were not assessed.",
                expected,
                encoding="dataset",
                validation={"performed": True, "valid": False, "dataset_rank": int(node.ndim)},
            )
        for values in _data_chunks(node, chunk_items):
            counts.update(_text(value) for value in np.asarray(values).reshape(-1))
    elif isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = node["codes"]
        categories = node["categories"]
        if not isinstance(codes, h5py.Dataset) or not isinstance(categories, h5py.Dataset):
            return _invalid_celltype_summary(
                "Categorical /obs/celltype codes and categories must be datasets; cell-type counts were not assessed.",
                expected,
                encoding="categorical_codes",
            )
        validation: dict[str, Any] = {
            "performed": True,
            "codes_rank": int(codes.ndim),
            "categories_rank": int(categories.ndim),
            "codes_dtype": str(codes.dtype),
            "categories_dtype": str(categories.dtype),
            "expected_observations": expected,
        }
        validation_issues: list[str] = []
        if codes.ndim != 1:
            validation_issues.append(f"codes must be one-dimensional, not {codes.ndim}-dimensional")
        if categories.ndim != 1:
            validation_issues.append(f"categories must be one-dimensional, not {categories.ndim}-dimensional")
        if not np.issubdtype(codes.dtype, np.integer):
            validation_issues.append(f"codes dtype must be integer, not {codes.dtype}")
        if codes.ndim == 1:
            validation["code_count"] = int(codes.shape[0])
            if expected is not None and int(codes.shape[0]) != expected:
                validation_issues.append(
                    f"codes has {int(codes.shape[0])} values but X has n_obs={expected}"
                )
        if validation_issues:
            validation["valid"] = False
            return _invalid_celltype_summary(
                "Categorical /obs/celltype encoding is invalid; cell-type counts were not assessed: "
                + "; ".join(validation_issues)
                + ".",
                expected,
                encoding="categorical_codes",
                validation=validation,
            )
        labels = _read_strings(categories)
        missing_codes = 0
        invalid_negative = 0
        invalid_positive = 0
        for values in _data_chunks(codes, chunk_items):
            block = np.asarray(values)
            missing_codes += int(np.count_nonzero(block == -1))
            invalid_negative += int(np.count_nonzero(block < -1))
            invalid_positive += int(np.count_nonzero(block >= len(labels)))
        validation.update(
            {
                "category_count": len(labels),
                "missing_code_count": missing_codes,
                "invalid_negative_code_count": invalid_negative,
                "invalid_positive_code_count": invalid_positive,
            }
        )
        if invalid_negative or invalid_positive:
            validation["valid"] = False
            return _invalid_celltype_summary(
                "Categorical /obs/celltype encoding is invalid; cell-type counts were not assessed: "
                f"{invalid_negative} code(s) below the only supported missing sentinel -1 and "
                f"{invalid_positive} code(s) outside category bounds [0, {len(labels)}).",
                expected,
                encoding="categorical_codes",
                validation=validation,
            )
        validation["valid"] = True
        for values in _data_chunks(codes, chunk_items):
            for code in np.asarray(values).reshape(-1):
                code_int = int(code)
                if code_int == -1:
                    counts["<missing>"] += 1
                else:
                    counts[labels[code_int]] += 1
    else:
        return _invalid_celltype_summary(
            "Unsupported /obs/celltype encoding; cell-type counts were not assessed.",
            expected,
            encoding="unsupported",
        )
    total = sum(counts.values())
    result = {
        "present": True,
        "status": "assessed",
        "assessment": "assessed",
        "encoding": "dataset" if isinstance(node, h5py.Dataset) else "categorical_codes",
        "counted_observations": total,
        "expected_observations": expected,
        "matches_n_obs": total == expected if expected is not None else None,
        "labels": [
            {"label": label, "count": count, "fraction": count / total if total else 0.0}
            for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "caveat": "Label vocabularies may not be harmonized across developmental stages. Absence of a label is not proof of biological absence.",
    }
    if isinstance(node, h5py.Group):
        result["validation"] = validation
        result["missing_code_count"] = validation["missing_code_count"]
    if expected is not None and total != expected:
        result["status"] = "invalid"
        result["assessment"] = "invalid"
        result["issue"] = f"celltype field has {total} values but X has n_obs={expected}."
    elif expected is None:
        result["warning"] = "celltype values were counted, but X does not provide a valid n_obs for comparison."
    return result


def _spatial_summary(handle: h5py.File, chunk_rows: int, expected_n_obs: int | None) -> dict[str, Any]:
    obsm = handle.get("obsm")
    if not isinstance(obsm, h5py.Group) or "spatial_3D" not in obsm:
        return {"present": False, "warning": "No /obsm/spatial_3D field was found."}
    node = obsm["spatial_3D"]
    if not isinstance(node, h5py.Dataset):
        return {"present": True, "issue": "/obsm/spatial_3D is not a dataset."}
    result: dict[str, Any] = {"present": True, "shape": list(node.shape), "dtype": str(node.dtype)}
    if node.ndim != 2 or node.shape[1] < 3:
        result["issue"] = "spatial_3D should have shape n_obs × 3 or more."
        return result
    result["coordinate_columns_summarized"] = 3
    result["extra_columns_ignored"] = max(0, int(node.shape[1]) - 3)
    result["row_count_matches_n_obs"] = int(node.shape[0]) == expected_n_obs if expected_n_obs is not None else None
    if expected_n_obs is not None and int(node.shape[0]) != expected_n_obs:
        result["issue"] = f"spatial_3D has {node.shape[0]} rows but X has n_obs={expected_n_obs}."
    if not np.issubdtype(node.dtype, np.number):
        result["issue"] = _append_issue(result.get("issue"), "spatial_3D has a non-numeric dtype.")
        return result
    if np.issubdtype(node.dtype, np.complexfloating):
        result["issue"] = _append_issue(
            result.get("issue"),
            "spatial_3D has a non-real complex dtype; coordinate statistics were skipped "
            "rather than discarding imaginary components.",
        )
        result["finite"] = None
        return result
    minimum = np.full(3, np.inf, dtype=float)
    maximum = np.full(3, -np.inf, dtype=float)
    sums = np.zeros(3, dtype=float)
    sumsquares = np.zeros(3, dtype=float)
    count = 0
    finite = True
    for block in _slice_chunks(node, chunk_rows):
        array = np.asarray(block[:, :3], dtype=float)
        block_finite = np.isfinite(array)
        finite = finite and bool(np.all(block_finite))
        fully_finite_rows = np.all(block_finite, axis=1)
        if not np.any(fully_finite_rows):
            continue
        usable = array[fully_finite_rows]
        minimum = np.minimum(minimum, np.min(usable, axis=0))
        maximum = np.maximum(maximum, np.max(usable, axis=0))
        sums += np.sum(usable, axis=0)
        sumsquares += np.sum(usable * usable, axis=0)
        count += usable.shape[0]
    result["finite"] = finite
    if count:
        centroid = sums / count
        variance = sumsquares / count - centroid * centroid
        result.update(
            {
                "min": minimum.tolist(),
                "max": maximum.tolist(),
                "centroid": centroid.tolist(),
                "variance": np.maximum(variance, 0).tolist(),
                "extent": (maximum - minimum).tolist(),
            }
        )
    else:
        result["issue"] = "No fully finite coordinate rows were available for summary statistics."
    result["caveat"] = (
        "Coordinates are per-embryo local and are not registered across time points. "
        "Do not interpret raw frame differences as developmental displacement or growth without an appropriate model."
    )
    return result


def _panel_summary(var_names: list[str], panel_path: Path | None) -> dict[str, Any] | None:
    if panel_path is None:
        return None
    panel_bytes = panel_path.read_bytes()
    panel = [line.strip() for line in panel_bytes.decode("utf-8").splitlines() if line.strip()]
    var_duplicates = sorted(name for name, count in Counter(var_names).items() if count > 1)
    panel_duplicates = sorted(name for name, count in Counter(panel).items() if count > 1)
    var_set, panel_set = set(var_names), set(panel)
    missing = [name for name in panel if name not in var_set]
    extra = [name for name in var_names if name not in panel_set]
    result: dict[str, Any] = {
        "panel_path": str(panel_path),
        "panel_sha256_local_fingerprint": hashlib.sha256(panel_bytes).hexdigest(),
        "panel_fingerprint_note": "Local fingerprint of the supplied panel file; its origin and board association are not authenticated by this tool.",
        "panel_gene_count": len(panel),
        "file_gene_count": len(var_names),
        "duplicates_in_file": var_duplicates,
        "duplicates_in_panel": panel_duplicates,
        "missing_from_file": missing,
        "extra_in_file": extra,
        "exact_order_match": bool(var_names == panel and not var_duplicates and not panel_duplicates),
        "same_unique_set_wrong_order": False,
    }
    if not var_duplicates and not panel_duplicates and var_set == panel_set and var_names != panel:
        result["same_unique_set_wrong_order"] = True
        result["reorder_plan_available"] = True
        result["reorder_plan_note"] = "A source-index plan can be written on request. It never changes the inspected file."
    else:
        result["reorder_plan_available"] = False
    return result


def _write_reorder_plan(report: dict[str, Any], output_dir: Path) -> str | None:
    panel_info = report.get("gene_panel")
    if not panel_info or not panel_info.get("reorder_plan_available"):
        return None
    var_names = report.get("_internal_var_names", [])
    panel_path = Path(panel_info["panel_path"])
    panel = [line.strip() for line in panel_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lookup = {gene: index for index, gene in enumerate(var_names)}
    destination = output_dir / "reorder_plan.csv"
    with destination.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_panel_position", "gene", "source_var_position"])
        writer.writerows((position, gene, lookup[gene]) for position, gene in enumerate(panel))
    return destination.name


def inspect_h5ad(
    source: str | Path,
    panel: str | Path | None = None,
    *,
    chunk_rows: int = 1024,
    chunk_items: int = 1_000_000,
) -> dict[str, Any]:
    """Inspect *source* without writing to it or loading an expression matrix whole."""
    path = Path(source).expanduser().resolve()
    panel_path = Path(panel).expanduser().resolve() if panel else None
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if panel_path and not panel_path.is_file():
        raise FileNotFoundError(f"Panel file does not exist: {panel_path}")
    if chunk_rows < 1 or chunk_items < 1:
        raise ValueError("chunk_rows and chunk_items must both be positive.")
    before = _file_snapshot(path)
    generated_at = datetime.now(timezone.utc).isoformat()
    fingerprint = _sha256_stream(path)
    _require_unchanged(path, before)
    with h5py.File(path, "r") as handle:
        var_names, var_index, var_warning = _var_names(handle)
        expression = _expression_summary(handle, chunk_rows, chunk_items)
        x_shape = expression.get("shape")
        n_obs = int(x_shape[0]) if isinstance(x_shape, list) and len(x_shape) == 2 else None
        n_vars = int(x_shape[1]) if isinstance(x_shape, list) and len(x_shape) == 2 else None
        obs = handle.get("obs")
        var = handle.get("var")
        obs_index = _axis_index_name(obs) if isinstance(obs, h5py.Group) else None
        obs_length = _axis_length(obs if isinstance(obs, h5py.Group) else None, obs_index)
        var_length = _axis_length(var if isinstance(var, h5py.Group) else None, var_index)
        axis_issues = []
        axis_matches_x: bool | None = None
        if n_obs is None or n_vars is None:
            axis_issues.append("X does not provide a valid two-dimensional shape; axis consistency with X was not assessed.")
        else:
            axis_matches_x = True
        if n_obs is not None and obs_length != n_obs:
            axis_issues.append(f"obs index length {obs_length!r} does not match X n_obs={n_obs}.")
            axis_matches_x = False
        if n_vars is not None and var_length != n_vars:
            axis_issues.append(f"var index length {var_length!r} does not match X n_vars={n_vars}.")
            axis_matches_x = False
        celltypes = _count_celltypes(obs, n_obs, chunk_items) if isinstance(obs, h5py.Group) else {"present": False, "warning": "No /obs group was found."}
        report: dict[str, Any] = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "generated_at_utc": generated_at,
            "read_only_behavior": "This process opens the source with h5py mode 'r' and writes only to the requested report directory; it does not provide OS-level immutability or control other processes.",
            "file_identity": {
                "path": str(path),
                "filename": path.name,
                "size_bytes": before["size_bytes"],
                "mtime_utc": datetime.fromtimestamp(before["mtime_ns"] / 1_000_000_000, tz=timezone.utc).isoformat(),
                "sha256_local_fingerprint": fingerprint,
                "fingerprint_note": "This is a local fingerprint, not an authenticated checksum.",
            },
            "root_keys": sorted(handle.keys()),
            "anndata_structure": {
                "n_obs": n_obs,
                "n_vars": n_vars,
                "x": expression,
                "obs": _group_inventory(handle, "obs"),
                "var": _group_inventory(handle, "var"),
                "obsm": _group_inventory(handle, "obsm"),
                "layers": _group_inventory(handle, "layers"),
                "uns": _group_inventory(handle, "uns"),
                "var_index_name": var_index,
                "obs_index_name": obs_index,
                "var_name_warning": var_warning,
                "axis_consistency": {
                    "obs_index_length": obs_length,
                    "var_index_length": var_length,
                    "matches_x": axis_matches_x,
                    "issues": axis_issues,
                },
                "obvious_missingness": {
                    "obs": _axis_missingness(obs if isinstance(obs, h5py.Group) else None, chunk_items),
                    "var": _axis_missingness(var if isinstance(var, h5py.Group) else None, chunk_items),
                    "note": "Bounded scan of directly stored one-dimensional fields and common categorical codes; not an exhaustive semantic missingness audit.",
                },
            },
            "celltype_summary": celltypes,
            "spatial_3D_summary": _spatial_summary(handle, chunk_rows, n_obs),
            "gene_panel": _panel_summary(var_names, panel_path),
            "limitations": [
                "This is a read-only structural and QC summary, not a scorer or an authorization decision.",
                "It handles common dense, CSR, and CSC AnnData HDF5 encodings. Unrecognized encodings are reported without materializing them.",
                "Finite, non-negative values do not establish the required challenge normalization; raw counts or another scale may pass these checks and still be unsuitable for scoring.",
                "For submissions, n_obs is a sample-size choice under the selected board limits, not a measurement or prediction of embryo size.",
                "Reports from challenge files may contain unpublished or data-derived details and local paths. Keep them private unless current terms permit sharing; use synthetic fixtures for public examples.",
            ],
            # Kept only for optional reorder-plan generation; stripped before any public report is written.
            "_internal_var_names": var_names,
        }
    _require_unchanged(path, before)
    return report


def _attention_findings(report: dict[str, Any]) -> list[str]:
    """Return every material warning shared by terminal and Markdown reports."""
    structure = report["anndata_structure"]
    x = structure["x"]
    attention: list[str] = []
    if structure["axis_consistency"]["matches_x"] is not True:
        attention.extend(structure["axis_consistency"]["issues"])
    if x.get("issue"):
        attention.append(x["issue"])
    if x.get("numeric_qc_performed") is False and not x.get("issue"):
        attention.append("X numeric QC was not assessed.")
    if x.get("finite") is False:
        attention.append("X contains non-finite values.")
    if x.get("negative_values_detected"):
        attention.append("X contains negative values.")
    spatial = report["spatial_3D_summary"]
    if spatial.get("finite") is False:
        attention.append("spatial_3D contains non-finite coordinate values.")
    if spatial.get("issue"):
        attention.append(spatial["issue"])
    celltypes = report["celltype_summary"]
    if celltypes.get("issue"):
        attention.append(celltypes["issue"])
    if celltypes.get("warning") and celltypes.get("present"):
        attention.append(celltypes["warning"])
    if structure.get("var_name_warning"):
        attention.append(structure["var_name_warning"])
    missingness = structure["obvious_missingness"]
    missing_total = sum(
        finding.get("missing_in_scanned", 0)
        for axis_name in ("obs", "var")
        for finding in missingness[axis_name].get("fields", {}).values()
    )
    if missing_total:
        attention.append(f"Bounded metadata scan found {missing_total} obvious missing value(s).")
    panel = report.get("gene_panel")
    if panel and not panel.get("exact_order_match"):
        attention.append(
            f"Panel mismatch: missing={len(panel['missing_from_file'])}, extra={len(panel['extra_in_file'])}, "
            f"wrong_order={panel['same_unique_set_wrong_order']}."
        )
        if panel.get("duplicates_in_file") or panel.get("duplicates_in_panel"):
            attention.append(
                f"Panel duplicate names: file={len(panel['duplicates_in_file'])}, "
                f"panel={len(panel['duplicates_in_panel'])}."
            )
    return attention


def _markdown(report: dict[str, Any]) -> str:
    identity = report["file_identity"]
    structure = report["anndata_structure"]
    x = structure["x"]
    obs_columns = [key for key in structure["obs"].get("keys", []) if key != structure.get("obs_index_name")]
    var_columns = [key for key in structure["var"].get("keys", []) if key != structure.get("var_index_name")]
    lines = [
        f"# {TOOL_NAME} report",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## File identity",
        "",
        f"- Path: `{identity['path']}`",
        f"- Size: {identity['size_bytes']} bytes",
        f"- SHA-256: `{identity['sha256_local_fingerprint']}`",
        f"- Note: {identity['fingerprint_note']}",
        "",
        "## AnnData structure",
        "",
        f"- n_obs: {structure['n_obs']}",
        f"- n_vars: {structure['n_vars']}",
        f"- X storage: {x.get('storage', 'unknown')}",
        f"- X dtype: {x.get('dtype', 'unknown')}",
        f"- obs columns: {', '.join(obs_columns) or '(none)'}",
        f"- var columns: {', '.join(var_columns) or '(none)'}",
        f"- obsm keys: {', '.join(structure['obsm'].get('keys', [])) or '(none)'}",
        f"- layers: {', '.join(structure['layers'].get('keys', [])) or '(none)'}",
        f"- uns keys: {', '.join(structure['uns'].get('keys', [])) or '(none)'}",
        f"- Axis lengths match X: {structure['axis_consistency']['matches_x']}",
    ]
    for issue in structure["axis_consistency"]["issues"]:
        lines.append(f"- Axis issue: {issue}")
    lines.extend([
        "",
        "## Expression QC",
        "",
        f"- Finite: {x.get('finite', 'not assessed')}",
        f"- Negative values detected: {x.get('negative_values_detected', 'not assessed')}",
        f"- Finite min / max: {x.get('min_finite', 'n/a')} / {x.get('max_finite', 'n/a')}",
        f"- Sparsity estimate: {x.get('sparsity_estimate', 'n/a')}",
    ])
    if x.get("issue"):
        lines.append(f"- Note: {x['issue']}")
    if x.get("sparsity_note"):
        lines.append(f"- Sparse note: {x['sparsity_note']}")
    missingness = structure["obvious_missingness"]
    missing_findings = []
    for axis_name in ("obs", "var"):
        for field, finding in missingness[axis_name].get("fields", {}).items():
            if finding.get("missing_in_scanned", 0):
                missing_findings.append(
                    f"{axis_name}/{field}: {finding['missing_in_scanned']} obvious missing value(s) "
                    f"in {finding['scanned']} scanned"
                )
    lines.extend(["", "## Obvious missingness (bounded scan)", ""])
    if missing_findings:
        lines.extend(f"- {finding}" for finding in missing_findings)
    else:
        lines.append("- No obvious missing values found in the scanned common one-dimensional fields.")
    lines.append(f"- Note: {missingness['note']}")
    celltypes = report["celltype_summary"]
    lines.extend(["", "## Cell-type summary", ""])
    if celltypes.get("present"):
        lines.append(f"- Status: {celltypes.get('status', 'unknown')}")
        lines.append(f"- Encoding: {celltypes.get('encoding', 'unknown')}")
        lines.append(f"- Counted observations: {celltypes.get('counted_observations', 'not assessed')}")
        lines.append(f"- Expected observations: {celltypes.get('expected_observations', 'not assessed')}")
        lines.append(f"- Matches n_obs: {celltypes.get('matches_n_obs', 'not assessed')}")
        if celltypes.get("missing_code_count") is not None:
            lines.append(f"- Missing categorical codes (-1): {celltypes['missing_code_count']}")
        if celltypes.get("issue"):
            lines.append(f"- Issue: {celltypes['issue']}")
        if celltypes.get("warning"):
            lines.append(f"- Warning: {celltypes['warning']}")
        if celltypes.get("labels"):
            lines.extend(["", "| Label | Count | Fraction |", "| --- | ---: | ---: |"])
            lines.extend(f"| {row['label']} | {row['count']} | {row['fraction']:.4f} |" for row in celltypes["labels"])
        else:
            lines.append("- Label counts: not assessed.")
        if celltypes.get("caveat"):
            lines.extend(["", f"> {celltypes['caveat']}"])
    else:
        lines.append(f"- {celltypes.get('warning', celltypes.get('issue', 'Not available.'))}")
    spatial = report["spatial_3D_summary"]
    lines.extend(["", "## Spatial summary", ""])
    if spatial.get("present"):
        lines.extend([
            f"- Shape: {spatial.get('shape')}",
            f"- Finite: {spatial.get('finite', 'not assessed')}",
            f"- Minimum: {spatial.get('min', 'n/a')}",
            f"- Maximum: {spatial.get('max', 'n/a')}",
            f"- Centroid: {spatial.get('centroid', 'n/a')}",
            f"- Variance: {spatial.get('variance', 'n/a')}",
            f"- Extent: {spatial.get('extent', 'n/a')}",
        ])
        if spatial.get("caveat"):
            lines.extend(["", f"> {spatial['caveat']}"])
        if spatial.get("issue"):
            lines.append(f"- Note: {spatial['issue']}")
    else:
        lines.append(f"- {spatial.get('warning', 'Not available.')}")
    panel = report.get("gene_panel")
    if panel:
        lines.extend(["", "## Gene-panel comparison", ""])
        lines.extend([
            f"- Panel path: `{panel['panel_path']}`",
            f"- Panel SHA-256 local fingerprint: `{panel['panel_sha256_local_fingerprint']}`",
            f"- Exact ordered match: {panel['exact_order_match']}",
            f"- Same unique set, different order: {panel['same_unique_set_wrong_order']}",
            f"- Missing from file: {len(panel['missing_from_file'])}",
            f"- Extra in file: {len(panel['extra_in_file'])}",
            f"- Duplicate file names: {len(panel['duplicates_in_file'])}",
            f"- Duplicate panel names: {len(panel['duplicates_in_panel'])}",
        ])
        if panel.get("reorder_plan_file"):
            lines.append(f"- Reorder plan: `{panel['reorder_plan_file']}` (a plan only; source unchanged)")
    attention = _attention_findings(report)
    if attention:
        lines.extend(["", "## Attention", ""])
        lines.extend(f"- {item}" for item in attention)
    lines.extend(["", "## Scope", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def terminal_summary(report: dict[str, Any]) -> str:
    identity = report["file_identity"]
    structure = report["anndata_structure"]
    x = structure["x"]
    lines = [
        f"{TOOL_NAME}: {identity['filename']}",
        f"  local SHA-256 fingerprint: {identity['sha256_local_fingerprint']}",
        f"  shape: {structure['n_obs']} observations × {structure['n_vars']} variables",
        f"  X: {x.get('storage', 'unknown')}, dtype={x.get('dtype', 'unknown')}, finite={x.get('finite', 'not assessed')}, negative={x.get('negative_values_detected', 'not assessed')}",
        f"  celltype: {'present' if report['celltype_summary'].get('present') else 'not found'}; spatial_3D: {'present' if report['spatial_3D_summary'].get('present') else 'not found'}",
        "  This is a read-only structural report, not a scorer or an authorization decision.",
        "  Structural QC cannot establish the required expression normalization.",
    ]
    attention = _attention_findings(report)
    if attention:
        lines.append("  attention:")
        lines.extend(f"    - {item}" for item in attention)
    return "\n".join(lines)


def write_reports(report: dict[str, Any], report_dir: str | Path, *, write_reorder_plan: bool = False) -> dict[str, Path]:
    """Write Markdown/JSON (and optionally a source-index plan) outside the source file."""
    destination = Path(report_dir).expanduser().resolve()
    source = Path(report["file_identity"]["path"]).resolve()
    if destination == source or destination.is_file():
        raise ValueError("Report destination must be a separate directory, not the input file.")
    if destination == source.parent and destination == source.parent.resolve():
        # Writing alongside the source is allowed, but prevent accidental filename collision below.
        pass
    json_path = destination / "inspection_report.json"
    markdown_path = destination / "inspection_report.md"
    intended = [json_path, markdown_path]
    if write_reorder_plan:
        intended.append(destination / "reorder_plan.csv")
    collisions = [path for path in intended if path.exists() or path.is_symlink()]
    if collisions:
        names = ", ".join(path.name for path in collisions)
        raise ValueError(f"Refusing to overwrite existing report artifact(s): {names}. Choose a new report directory.")
    destination.mkdir(parents=True, exist_ok=True)
    if write_reorder_plan:
        reorder = _write_reorder_plan(report, destination)
        if report.get("gene_panel") and reorder:
            report["gene_panel"]["reorder_plan_file"] = reorder
    public_report = {key: value for key, value in report.items() if key != "_internal_var_names"}
    json_text = json.dumps(_plain(public_report), indent=2, sort_keys=True) + "\n"
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(json_text)
    markdown_text = _markdown(public_report)
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(markdown_text)
    results = {"json": json_path, "markdown": markdown_path}
    if write_reorder_plan and (destination / "reorder_plan.csv").exists():
        results["reorder_plan"] = destination / "reorder_plan.csv"
    return results
