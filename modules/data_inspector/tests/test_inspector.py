from __future__ import annotations

import os
import json
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
from scipy import sparse as scipy_sparse

# Keep the checked-in synthetic fixture generator importable both when this
# suite is run from the module directory and via the documented root command.
_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from examples.make_synthetic_fixtures import extended_spatial_500_fixture, extended_wide_sparse_fixture
from vec_data_inspector.cli import main as cli_main
from vec_data_inspector.inspector import inspect_h5ad, terminal_summary, write_reports


def make_file(path: Path, *, sparse: bool = False, spatial: bool = False, values: np.ndarray | None = None, genes: list[str] | None = None) -> None:
    values = np.asarray(values if values is not None else [[0.0, 2.0], [3.0, 0.0]], dtype=np.float32 if values is None else None)
    genes = genes or ["A", "B"]
    with h5py.File(path, "w") as handle:
        if sparse:
            x = handle.create_group("X")
            x.attrs["encoding-type"] = "csr_matrix"
            x.attrs["shape"] = np.array(values.shape, dtype=np.int64)
            rows, cols = np.nonzero(values)
            data = values[rows, cols]
            indptr = [0]
            for row in range(values.shape[0]):
                indptr.append(indptr[-1] + int(np.count_nonzero(rows == row)))
            x.create_dataset("data", data=data)
            x.create_dataset("indices", data=cols.astype(np.int32))
            x.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int32))
        else:
            handle.create_dataset("X", data=values)
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.asarray([f"c{i}" for i in range(values.shape[0])], dtype=h5py.string_dtype("utf-8")))
        obs.create_dataset("celltype", data=np.asarray(["x"] * values.shape[0], dtype=h5py.string_dtype("utf-8")))
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.asarray(genes, dtype=h5py.string_dtype("utf-8")))
        obsm = handle.create_group("obsm")
        if spatial:
            obsm.create_dataset("spatial_3D", data=np.arange(values.shape[0] * 3, dtype=np.float32).reshape(values.shape[0], 3))
        handle.create_group("layers")
        handle.create_group("uns")


def source_snapshot(path: Path) -> tuple[str, int, int]:
    """The public read-only contract includes bytes, size, and nanosecond mtime."""
    stat = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), stat.st_size, stat.st_mtime_ns


def make_anndata_file(path: Path, matrix: np.ndarray | scipy_sparse.spmatrix) -> None:
    """Write a normal on-disk AnnData file; used for positive encoding fixtures."""
    adata = ad.AnnData(X=matrix)
    adata.obs_names = [f"cell_{index}" for index in range(adata.n_obs)]
    adata.var_names = [f"gene_{index}" for index in range(adata.n_vars)]
    adata.obs["celltype"] = ["type_a"] * adata.n_obs
    adata.obsm["spatial_3D"] = np.arange(adata.n_obs * 3, dtype=np.float32).reshape(adata.n_obs, 3)
    adata.write_h5ad(path)


def make_sparse_layout(
    path: Path,
    *,
    encoding: str = "csr_matrix",
    data: np.ndarray | list[float] = (1.0, 2.0),
    indices: np.ndarray | list[int] = (0, 1),
    indptr: np.ndarray | list[int] = (0, 1, 2),
    shape: np.ndarray | list[int] = (2, 2),
) -> None:
    """Construct deliberately malformed sparse HDF5 layouts without a repair path."""
    make_file(path)
    with h5py.File(path, "r+") as handle:
        del handle["X"]
        x = handle.create_group("X")
        x.attrs["encoding-type"] = encoding
        x.attrs["shape"] = np.asarray(shape)
        x.create_dataset("data", data=np.asarray(data))
        x.create_dataset("indices", data=np.asarray(indices))
        x.create_dataset("indptr", data=np.asarray(indptr))


def replace_with_categorical_celltype(
    path: Path,
    codes: np.ndarray,
    categories: np.ndarray | None = None,
) -> None:
    """Make a common categorical /obs/celltype group, including corrupt cases."""
    with h5py.File(path, "r+") as handle:
        del handle["obs/celltype"]
        celltype = handle["obs"].create_group("celltype")
        celltype.attrs["encoding-type"] = "categorical"
        celltype.create_dataset("codes", data=np.asarray(codes))
        celltype.create_dataset(
            "categories",
            data=np.asarray(["A", "B"] if categories is None else categories, dtype=h5py.string_dtype("utf-8")),
        )


class InspectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_dense_qc_and_read_only_source(self) -> None:
        source = self.root / "dense.h5ad"
        make_file(source, values=np.array([[0, np.nan], [-1, np.inf]], dtype=np.float32))
        before = (os.stat(source).st_size, os.stat(source).st_mtime_ns, source.read_bytes())
        report = inspect_h5ad(source, chunk_rows=1)
        after = (os.stat(source).st_size, os.stat(source).st_mtime_ns, source.read_bytes())
        self.assertEqual(before, after)
        self.assertFalse(report["anndata_structure"]["x"]["finite"])
        self.assertTrue(report["anndata_structure"]["x"]["negative_values_detected"])
        self.assertEqual(report["anndata_structure"]["n_obs"], 2)

    def test_sparse_qc_does_not_read_matrix_as_dense(self) -> None:
        source = self.root / "sparse.h5ad"
        make_file(source, sparse=True, spatial=True)
        report = inspect_h5ad(source, chunk_items=1)
        x = report["anndata_structure"]["x"]
        self.assertEqual(x["storage"], "sparse_group")
        self.assertEqual(x["stored_entries"], 2)
        self.assertTrue(x["finite"])
        self.assertTrue(report["spatial_3D_summary"]["present"])

    def test_panel_exact_wrong_order_missing_extra_and_duplicates(self) -> None:
        source = self.root / "genes.h5ad"
        make_file(source, genes=["B", "A"])
        panel = self.root / "panel.txt"
        panel.write_text("A\nB\n", encoding="utf-8")
        report = inspect_h5ad(source, panel)
        comparison = report["gene_panel"]
        self.assertFalse(comparison["exact_order_match"])
        self.assertTrue(comparison["same_unique_set_wrong_order"])
        outputs = write_reports(report, self.root / "reports", write_reorder_plan=True)
        self.assertTrue(outputs["reorder_plan"].is_file())
        self.assertIn("A", outputs["reorder_plan"].read_text(encoding="utf-8"))
        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertIn("obs columns: celltype", markdown)
        self.assertNotIn("obs columns: _index", markdown)
        make_file(source, genes=["A", "A", "C"])
        report = inspect_h5ad(source, panel)
        comparison = report["gene_panel"]
        self.assertEqual(comparison["missing_from_file"], ["B"])
        self.assertEqual(comparison["extra_in_file"], ["C"])
        self.assertEqual(comparison["duplicates_in_file"], ["A"])

    def test_missing_spatial_and_absent_celltype_are_reported(self) -> None:
        source = self.root / "minimal.h5ad"
        make_file(source)
        with h5py.File(source, "r+") as handle:
            del handle["obs/celltype"]
        report = inspect_h5ad(source)
        self.assertFalse(report["spatial_3D_summary"]["present"])
        self.assertFalse(report["celltype_summary"]["present"])

    def test_bad_spatial_shape_is_described(self) -> None:
        source = self.root / "bad_spatial.h5ad"
        make_file(source, spatial=True)
        with h5py.File(source, "r+") as handle:
            del handle["obsm/spatial_3D"]
            handle["obsm"].create_dataset("spatial_3D", data=np.ones((2, 2), dtype=np.float32))
        report = inspect_h5ad(source)
        self.assertIn("3 or more", report["spatial_3D_summary"]["issue"])

    def test_spatial_extra_columns_are_allowed_but_only_first_three_are_summarized(self) -> None:
        source = self.root / "spatial_four_columns.h5ad"
        make_file(source, spatial=True)
        with h5py.File(source, "r+") as handle:
            del handle["obsm/spatial_3D"]
            handle["obsm"].create_dataset("spatial_3D", data=np.arange(8, dtype=np.float32).reshape(2, 4))
        spatial = inspect_h5ad(source)["spatial_3D_summary"]
        self.assertNotIn("issue", spatial)
        self.assertEqual(spatial["coordinate_columns_summarized"], 3)
        self.assertEqual(spatial["extra_columns_ignored"], 1)
        self.assertEqual(len(spatial["centroid"]), 3)

    def test_unusual_expression_dtype_is_not_coerced(self) -> None:
        source = self.root / "strings.h5ad"
        make_file(source)
        with h5py.File(source, "r+") as handle:
            del handle["X"]
            handle.create_dataset("X", data=np.asarray([["zero", "one"], ["two", "three"]], dtype=h5py.string_dtype("utf-8")))
        report = inspect_h5ad(source)
        x = report["anndata_structure"]["x"]
        self.assertFalse(x["numeric_qc_performed"])
        self.assertIn("non-numeric", x["issue"])

    def test_inventory_does_not_expand_large_looking_uns_metadata(self) -> None:
        source = self.root / "metadata.h5ad"
        make_file(source)
        with h5py.File(source, "r+") as handle:
            handle["uns"].create_dataset("large_synthetic_note", data=np.zeros(1_000_000, dtype=np.uint8))
        report = inspect_h5ad(source)
        self.assertIn("large_synthetic_note", report["anndata_structure"]["uns"]["keys"])

    def test_spatial_statistics_ignore_only_nonfinite_rows(self) -> None:
        source = self.root / "partially_bad_spatial.h5ad"
        make_file(source, spatial=True)
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]],
            dtype=np.float32,
        )
        with h5py.File(source, "r+") as handle:
            del handle["obsm/spatial_3D"]
            handle["obsm"].create_dataset("spatial_3D", data=coordinates)
        rowwise = inspect_h5ad(source, chunk_rows=1)["spatial_3D_summary"]
        whole_chunk = inspect_h5ad(source, chunk_rows=4)["spatial_3D_summary"]
        self.assertFalse(rowwise["finite"])
        self.assertEqual(rowwise["min"], [0.0, 0.0, 0.0])
        self.assertEqual(rowwise["max"], [4.0, 8.0, 12.0])
        self.assertEqual(rowwise["centroid"], [2.0, 4.0, 6.0])
        for key in ("min", "max", "centroid", "variance", "extent"):
            self.assertEqual(rowwise[key], whole_chunk[key])

    def test_notebook_has_deterministic_cell_ids(self) -> None:
        notebook = Path(__file__).parents[1] / "notebooks" / "vec_data_inspector_demo.ipynb"
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        ids = [cell.get("id") for cell in payload["cells"]]
        self.assertTrue(all(isinstance(cell_id, str) and cell_id for cell_id in ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_notebook_documents_the_supported_local_jupyter_path(self) -> None:
        notebook = Path(__file__).parents[1] / "notebooks" / "vec_data_inspector_demo.ipynb"
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        sources = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        self.assertIn("supports local Jupyter only", sources)
        self.assertIn("Google Colab is not assessed or supported", sources)

    def test_checked_in_synthetic_example_report_has_no_private_paths(self) -> None:
        report = (Path(__file__).parents[1] / "examples" / "synthetic_inspection_report.md").read_text(encoding="utf-8")
        self.assertIn("Synthetic-only example", report)
        self.assertIn("synthetic_dense_t1_like.h5ad", report)
        self.assertNotIn("/" + "Users/", report)
        self.assertNotIn("/" + "private/", report)

    def test_extended_generated_fixture_shapes_and_inspection(self) -> None:
        wide = self.root / "wide.h5ad"
        spatial = self.root / "spatial_500.h5ad"
        extended_wide_sparse_fixture(wide)
        extended_spatial_500_fixture(spatial)
        wide_report = inspect_h5ad(wide)
        spatial_report = inspect_h5ad(spatial)
        self.assertEqual(wide_report["anndata_structure"]["x"]["shape"], [8, 2048])
        self.assertEqual(wide_report["anndata_structure"]["x"]["storage"], "sparse_group")
        self.assertTrue(wide_report["celltype_summary"]["present"])
        self.assertFalse(wide_report["spatial_3D_summary"]["present"])
        self.assertEqual(spatial_report["anndata_structure"]["x"]["shape"], [10, 500])
        self.assertTrue(spatial_report["celltype_summary"]["present"])
        self.assertEqual(spatial_report["spatial_3D_summary"]["shape"], [10, 3])

    def test_report_writer_refuses_to_overwrite_existing_artifacts(self) -> None:
        source = self.root / "source.h5ad"
        make_file(source)
        report = inspect_h5ad(source)
        destination = self.root / "reports"
        destination.mkdir()
        sentinel = destination / "inspection_report.md"
        sentinel.write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            write_reports(report, destination)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")
        self.assertFalse((destination / "inspection_report.json").exists())

    def test_report_writer_preserves_files_created_after_collision_check(self) -> None:
        source = self.root / "race_source.h5ad"
        make_file(source)
        before = source_snapshot(source)
        panel = self.root / "race_panel.txt"
        panel.write_text("B\nA\n", encoding="utf-8")
        real_mkdir = Path.mkdir
        for filename in ("inspection_report.json", "inspection_report.md", "reorder_plan.csv"):
            with self.subTest(filename=filename):
                report = inspect_h5ad(source, panel)
                destination = (self.root / ("race_" + filename)).resolve()
                collision = destination / filename
                sentinel = b"report created by another writer"
                injected = []

                def create_late_file(path, *args, **kwargs):
                    real_mkdir(path, *args, **kwargs)
                    if path == destination:
                        collision.write_bytes(sentinel)
                        injected.append(True)

                error = None
                with patch.object(Path, "mkdir", new=create_late_file):
                    try:
                        write_reports(report, destination, write_reorder_plan=True)
                    except FileExistsError as caught:
                        error = caught
                self.assertEqual(injected, [True], "The deterministic collision must actually be injected.")
                self.assertTrue(collision.read_bytes() == sentinel,
                                "A report artifact created after preflight was overwritten.")
                self.assertIsInstance(error, FileExistsError)
                self.assertEqual(source_snapshot(source), before)

    def test_report_writer_refuses_dangling_output_symlinks(self) -> None:
        source = self.root / "symlink_source.h5ad"
        make_file(source)
        before = source_snapshot(source)
        panel = self.root / "symlink_panel.txt"
        panel.write_text("B\nA\n", encoding="utf-8")
        for filename in ("inspection_report.json", "inspection_report.md", "reorder_plan.csv"):
            with self.subTest(filename=filename):
                report = inspect_h5ad(source, panel)
                destination = self.root / ("symlink_" + filename)
                destination.mkdir()
                outside = self.root / ("outside_" + filename)
                output = destination / filename
                try:
                    output.symlink_to(outside)
                except OSError as error:
                    self.skipTest(f"Symlinks unavailable: {error}")
                error = None
                try:
                    write_reports(report, destination, write_reorder_plan=True)
                except (ValueError, FileExistsError) as caught:
                    error = caught
                self.assertFalse(outside.exists(), "A dangling output link was followed outside the report directory.")
                self.assertIsNotNone(error)
                self.assertTrue(output.is_symlink())
                self.assertEqual(source_snapshot(source), before)

    def test_cli_returns_error_for_late_report_collision(self) -> None:
        source = self.root / "cli_race_source.h5ad"
        make_file(source)
        before = source_snapshot(source)
        destination = (self.root / "cli_race_reports").resolve()
        collision = destination / "inspection_report.md"
        sentinel = b"concurrent Markdown report"
        real_mkdir = Path.mkdir
        injected = []

        def create_late_file(path, *args, **kwargs):
            real_mkdir(path, *args, **kwargs)
            if path == destination:
                collision.write_bytes(sentinel)
                injected.append(True)

        stderr = io.StringIO()
        with patch.object(Path, "mkdir", new=create_late_file), redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            code = cli_main([str(source), "--report-dir", str(destination)])
        self.assertEqual(injected, [True], "The deterministic collision must actually be injected.")
        self.assertEqual(code, 2)
        self.assertTrue(collision.read_bytes() == sentinel)
        self.assertIn("vec-inspect:", stderr.getvalue())
        self.assertEqual(source_snapshot(source), before)

    def test_sparse_statistics_include_implicit_zeros(self) -> None:
        one_value = self.root / "one_value.h5ad"
        make_file(one_value, sparse=True, values=np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float32))
        x = inspect_h5ad(one_value)["anndata_structure"]["x"]
        self.assertEqual(x["min_finite"], 0.0)
        self.assertEqual(x["max_finite"], 5.0)
        self.assertEqual(x["logical_zero_fraction_lower_bound"], 0.75)
        self.assertEqual(x["sparsity_estimate"], 0.75)
        self.assertEqual(x["stored_value_zero_fraction"], 0.0)

        all_zero = self.root / "all_zero.h5ad"
        make_file(all_zero, sparse=True, values=np.zeros((2, 2), dtype=np.float32))
        zero_x = inspect_h5ad(all_zero)["anndata_structure"]["x"]
        self.assertEqual(zero_x["min_finite"], 0.0)
        self.assertEqual(zero_x["max_finite"], 0.0)
        self.assertEqual(zero_x["sparsity_estimate"], 1.0)

    def test_axis_spatial_consistency_and_bounded_missingness(self) -> None:
        source = self.root / "mismatch.h5ad"
        make_file(source, spatial=True)
        with h5py.File(source, "r+") as handle:
            del handle["obs/_index"]
            handle["obs"].create_dataset(
                "_index", data=np.asarray(["c0"], dtype=h5py.string_dtype("utf-8"))
            )
            handle["obs"].create_dataset(
                "note", data=np.asarray(["", "present"], dtype=h5py.string_dtype("utf-8"))
            )
            del handle["obsm/spatial_3D"]
            handle["obsm"].create_dataset("spatial_3D", data=np.ones((1, 3), dtype=np.float32))
        report = inspect_h5ad(source)
        consistency = report["anndata_structure"]["axis_consistency"]
        self.assertFalse(consistency["matches_x"])
        self.assertIn("obs index length", consistency["issues"][0])
        self.assertIn("rows but X", report["spatial_3D_summary"]["issue"])
        missing = report["anndata_structure"]["obvious_missingness"]["obs"]["fields"]["note"]
        self.assertEqual(missing["missing_in_scanned"], 1)

    def test_invalid_x_dimensions_are_reported_by_api_and_cli_without_source_change(self) -> None:
        source = self.root / "one_dimensional_x.h5ad"
        make_file(source)
        with h5py.File(source, "r+") as handle:
            del handle["X"]
            handle.create_dataset("X", data=np.asarray([1.0, 2.0], dtype=np.float32))
        before = source_snapshot(source)
        report = inspect_h5ad(source)
        self.assertEqual(before, source_snapshot(source))
        x = report["anndata_structure"]["x"]
        self.assertEqual(x["assessment"], "invalid")
        self.assertFalse(x["numeric_qc_performed"])
        self.assertIn("1-dimensional", x["issue"])
        self.assertIsNone(report["anndata_structure"]["n_vars"])
        self.assertIn("valid two-dimensional shape", terminal_summary(report))

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main([str(source), "--report-dir", str(self.root / "one_dimensional_report")])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn("Traceback", stdout.getvalue())
        self.assertIn("1-dimensional", stdout.getvalue())
        self.assertTrue((self.root / "one_dimensional_report" / "inspection_report.json").is_file())
        self.assertEqual(before, source_snapshot(source))

    def test_missing_scalar_and_higher_dimensional_x_are_explicitly_not_assessed(self) -> None:
        cases: dict[str, np.ndarray | None] = {
            "missing": None,
            "scalar": np.asarray(1.0, dtype=np.float32),
            "three_dimensional": np.ones((1, 2, 3), dtype=np.float32),
        }
        for name, replacement in cases.items():
            with self.subTest(name=name):
                source = self.root / f"{name}_x.h5ad"
                make_file(source)
                with h5py.File(source, "r+") as handle:
                    del handle["X"]
                    if replacement is not None:
                        handle.create_dataset("X", data=replacement)
                report = inspect_h5ad(source)
                x = report["anndata_structure"]["x"]
                self.assertFalse(x["numeric_qc_performed"])
                self.assertIn(x["assessment"], {"invalid", "not_assessed"})
                self.assertIn("not assessed", terminal_summary(report))

    def test_unreadable_input_has_clean_cli_exit_and_does_not_change_source(self) -> None:
        source = self.root / "not_hdf5.h5ad"
        source.write_bytes(b"not an HDF5 file")
        before = source_snapshot(source)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main([str(source), "--report-dir", str(self.root / "never_written")])
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("vec-inspect:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse((self.root / "never_written").exists())
        self.assertEqual(before, source_snapshot(source))

    def test_anndata_written_csr_and_csc_pass_bounded_structure_checks(self) -> None:
        values = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
        for name, matrix, encoding in (
            ("csr", scipy_sparse.csr_matrix(values), "csr_matrix"),
            ("csc", scipy_sparse.csc_matrix(values), "csc_matrix"),
        ):
            with self.subTest(encoding=encoding):
                source = self.root / f"anndata_{name}.h5ad"
                make_anndata_file(source, matrix)
                report = inspect_h5ad(source, chunk_items=1)
                x = report["anndata_structure"]["x"]
                self.assertEqual(x["encoding_type"], encoding)
                self.assertTrue(x["sparse_structure"]["performed"])
                self.assertTrue(x["sparse_structure"]["valid"])
                self.assertTrue(x["numeric_qc_performed"])
                self.assertEqual(x["logical_zero_fraction_lower_bound"], 0.5)

    def test_corrupt_sparse_structure_is_not_given_numeric_or_logical_statistics(self) -> None:
        corrupt_cases = {
            "out_of_bounds_indices": {"indices": (77, 88), "indptr": (0, 2, 2)},
            "bad_final_pointer": {"indices": (0, 1), "indptr": (0, 2, 999)},
            "mismatched_lengths": {"data": (1.0,), "indices": (0, 1), "indptr": (0, 1, 1)},
        }
        for name, kwargs in corrupt_cases.items():
            with self.subTest(name=name):
                source = self.root / f"{name}.h5ad"
                make_sparse_layout(source, **kwargs)
                report = inspect_h5ad(source, chunk_items=1)
                x = report["anndata_structure"]["x"]
                self.assertFalse(x["sparse_structure"]["valid"])
                self.assertFalse(x["numeric_qc_performed"])
                self.assertEqual(x["assessment"], "invalid")
                self.assertNotIn("logical_zero_fraction_lower_bound", x)
                self.assertIn("structural validation failed", x["issue"])
                self.assertIn("structural validation failed", terminal_summary(report))

    def test_complex_expression_and_coordinates_are_declined_without_coercion(self) -> None:
        source = self.root / "complex.h5ad"
        make_file(source, values=np.asarray([[1 + 2j, 0], [0, 3 + 4j]], dtype=np.complex64), spatial=True)
        with h5py.File(source, "r+") as handle:
            del handle["obsm/spatial_3D"]
            handle["obsm"].create_dataset(
                "spatial_3D", data=np.asarray([[1 + 1j, 0, 0], [0, 1, 1]], dtype=np.complex64)
            )
        report = inspect_h5ad(source)
        x = report["anndata_structure"]["x"]
        spatial = report["spatial_3D_summary"]
        self.assertFalse(x["numeric_qc_performed"])
        self.assertIn("non-real complex", x["issue"])
        self.assertIsNone(x["min_finite"])
        self.assertIn("non-real complex", spatial["issue"])
        self.assertIsNone(spatial["finite"])
        terminal = terminal_summary(report)
        self.assertIn("non-real complex", terminal)

    def test_duplicate_sparse_entries_are_not_presented_as_logical_matrix_extrema(self) -> None:
        source = self.root / "duplicate_entries.h5ad"
        # Values 1 and 2 occupy the same logical (0, 0) coordinate.  Its
        # coalesced value would be 3, so a maximum of 2 must not be labeled a
        # logical-matrix maximum.
        make_sparse_layout(source, data=(1.0, 2.0), indices=(0, 0), indptr=(0, 2, 2))
        x = inspect_h5ad(source)["anndata_structure"]["x"]
        self.assertTrue(x["sparse_structure"]["valid"])
        self.assertEqual(x["max_finite"], 2.0)
        self.assertEqual(x["logical_zero_fraction_lower_bound"], 0.5)
        self.assertEqual(x["sparse_numeric_semantics"], "stored_entry_statistics_not_coalesced_logical_matrix_statistics")
        self.assertIn("not logical-matrix extrema", x["sparsity_note"])

    def test_terminal_surfaces_spatial_nonfinite_unsupported_x_and_panel_duplicates(self) -> None:
        spatial_source = self.root / "spatial_nan.h5ad"
        make_file(spatial_source, spatial=True)
        with h5py.File(spatial_source, "r+") as handle:
            handle["obsm/spatial_3D"][0, 0] = np.nan
        self.assertIn("non-finite coordinate", terminal_summary(inspect_h5ad(spatial_source)))

        unsupported_source = self.root / "unsupported_x.h5ad"
        make_file(unsupported_source)
        with h5py.File(unsupported_source, "r+") as handle:
            del handle["X"]
            x = handle.create_group("X")
            x.attrs["encoding-type"] = "unsupported_layout"
        self.assertIn("Unsupported /X group encoding", terminal_summary(inspect_h5ad(unsupported_source)))

        panel_source = self.root / "panel_duplicates.h5ad"
        make_file(panel_source)
        panel = self.root / "duplicates.txt"
        panel.write_text("A\nA\nB\n", encoding="utf-8")
        self.assertIn("Panel duplicate names", terminal_summary(inspect_h5ad(panel_source, panel)))

    def test_cli_collision_preserves_source_hash_size_and_mtime(self) -> None:
        source = self.root / "collision_source.h5ad"
        make_anndata_file(source, scipy_sparse.csr_matrix(np.eye(2, dtype=np.float32)))
        before = source_snapshot(source)
        report_dir = self.root / "collision_reports"
        report_dir.mkdir()
        (report_dir / "inspection_report.md").write_text("sentinel", encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main([str(source), "--report-dir", str(report_dir)])
        self.assertEqual(code, 2)
        self.assertIn("Refusing to overwrite", stderr.getvalue())
        self.assertEqual((report_dir / "inspection_report.md").read_text(encoding="utf-8"), "sentinel")
        self.assertFalse((report_dir / "inspection_report.json").exists())
        self.assertEqual(before, source_snapshot(source))

    def test_valid_string_and_categorical_celltypes_are_assessed_without_repair(self) -> None:
        string_source = self.root / "valid_string_celltype.h5ad"
        make_file(string_source)
        string_summary = inspect_h5ad(string_source, chunk_items=1)["celltype_summary"]
        self.assertEqual(string_summary["status"], "assessed")
        self.assertEqual(string_summary["encoding"], "dataset")
        self.assertNotIn("issue", string_summary)

        categorical_source = self.root / "valid_categorical_celltype.h5ad"
        make_file(categorical_source, values=np.ones((7, 2), dtype=np.float32))
        replace_with_categorical_celltype(
            categorical_source,
            np.asarray([0, 1, -1, 0, 1, -1, 0], dtype=np.int32),
        )
        before = source_snapshot(categorical_source)
        report = inspect_h5ad(categorical_source, chunk_items=1)
        self.assertEqual(before, source_snapshot(categorical_source))
        summary = report["celltype_summary"]
        self.assertEqual(summary["status"], "assessed")
        self.assertEqual(summary["missing_code_count"], 2)
        self.assertEqual(summary["validation"]["invalid_negative_code_count"], 0)
        self.assertEqual(summary["validation"]["invalid_positive_code_count"], 0)
        self.assertEqual(
            {row["label"]: row["count"] for row in summary["labels"]},
            {"A": 3, "B": 2, "<missing>": 2},
        )

    def test_malformed_categorical_celltype_encodings_are_invalid_not_coerced(self) -> None:
        cases: dict[str, tuple[np.ndarray, np.ndarray | None, str]] = {
            "fractional": (np.asarray([0.9, 1.9, 0.1], dtype=np.float64), None, "codes dtype must be integer"),
            "scalar": (np.asarray(0, dtype=np.int32), None, "codes must be one-dimensional"),
            "positive_out_of_range": (np.asarray([0, 2, 1], dtype=np.int32), None, "outside category bounds"),
            "negative_not_missing": (np.asarray([0, -2, 1], dtype=np.int32), None, "below the only supported missing sentinel -1"),
            "wrong_length": (np.asarray([0, 1], dtype=np.int32), None, "codes has 2 values"),
            "scalar_categories": (np.asarray([0, 0, 0], dtype=np.int32), np.asarray("A"), "categories must be one-dimensional"),
        }
        for name, (codes, categories, expected_issue) in cases.items():
            with self.subTest(name=name):
                source = self.root / f"categorical_{name}.h5ad"
                make_file(source, values=np.ones((3, 2), dtype=np.float32))
                replace_with_categorical_celltype(source, codes, categories)
                before = source_snapshot(source)
                report = inspect_h5ad(source, chunk_items=1)
                self.assertEqual(before, source_snapshot(source))
                summary = report["celltype_summary"]
                self.assertEqual(summary["status"], "invalid")
                self.assertEqual(summary["assessment"], "invalid")
                self.assertEqual(summary["labels"], [])
                self.assertIsNone(summary["counted_observations"])
                self.assertIn(expected_issue, summary["issue"])
                self.assertIn(summary["issue"], terminal_summary(report))
                if name == "negative_not_missing":
                    missingness = report["anndata_structure"]["obvious_missingness"]["obs"]["fields"]["celltype"]
                    self.assertEqual(missingness["missing_in_scanned"], 0)
                    self.assertEqual(missingness["invalid_negative_codes_in_scanned"], 1)

    def test_invalid_categorical_cli_writes_report_and_propagates_warning_everywhere(self) -> None:
        source = self.root / "categorical_fractional_cli.h5ad"
        make_file(source)
        replace_with_categorical_celltype(source, np.asarray([0.9, 1.9], dtype=np.float64))
        before = source_snapshot(source)
        report_dir = self.root / "categorical_cli_report"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main([str(source), "--report-dir", str(report_dir), "--chunk-items", "1"])
        self.assertEqual(code, 0, "exit 0 means a report was generated, not that metadata is valid")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(before, source_snapshot(source))
        payload = json.loads((report_dir / "inspection_report.json").read_text(encoding="utf-8"))
        markdown = (report_dir / "inspection_report.md").read_text(encoding="utf-8")
        issue = payload["celltype_summary"]["issue"]
        self.assertEqual(payload["celltype_summary"]["status"], "invalid")
        self.assertIn("attention:", stdout.getvalue())
        self.assertIn(issue, stdout.getvalue())
        self.assertIn(issue, markdown)
        self.assertIn("## Attention", markdown)

    def test_markdown_preserves_safe_string_celltype_counts_and_length_issue(self) -> None:
        source = self.root / "string_celltype_wrong_length.h5ad"
        make_file(source, values=np.ones((3, 2), dtype=np.float32))
        with h5py.File(source, "r+") as handle:
            del handle["obs/celltype"]
            handle["obs"].create_dataset(
                "celltype", data=np.asarray(["A", "B"], dtype=h5py.string_dtype("utf-8"))
            )
        report = inspect_h5ad(source)
        issue = report["celltype_summary"]["issue"]
        outputs = write_reports(report, self.root / "string_length_report")
        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertEqual(report["celltype_summary"]["status"], "invalid")
        self.assertEqual(len(report["celltype_summary"]["labels"]), 2)
        self.assertIn(issue, terminal_summary(report))
        self.assertIn(issue, markdown)
        self.assertIn("| A | 1 | 0.5000 |", markdown)

    def test_attention_is_shared_by_json_terminal_and_markdown_for_critical_findings(self) -> None:
        source = self.root / "critical_propagation.h5ad"
        make_file(source, spatial=True, genes=["A", "A"])
        replace_with_categorical_celltype(source, np.asarray([0], dtype=np.int32))
        with h5py.File(source, "r+") as handle:
            handle["obsm/spatial_3D"][0, 0] = np.nan
        panel = self.root / "critical_panel.txt"
        panel.write_text("A\nA\n", encoding="utf-8")
        report = inspect_h5ad(source, panel)
        markdown = (self.root / "critical_markdown")
        outputs = write_reports(report, markdown)
        text = outputs["markdown"].read_text(encoding="utf-8")
        terminal = terminal_summary(report)
        self.assertIn(report["celltype_summary"]["issue"], terminal)
        self.assertIn(report["celltype_summary"]["issue"], text)
        self.assertIn("spatial_3D contains non-finite coordinate values.", terminal)
        self.assertIn("spatial_3D contains non-finite coordinate values.", text)
        self.assertIn("Panel duplicate names", terminal)
        self.assertIn("Panel duplicate names", text)

    def test_expression_axis_spatial_and_panel_findings_reach_json_terminal_and_markdown(self) -> None:
        expression_source = self.root / "expression_warning.h5ad"
        make_file(expression_source)
        with h5py.File(expression_source, "r+") as handle:
            del handle["X"]
            unsupported = handle.create_group("X")
            unsupported.attrs["encoding-type"] = "unsupported_layout"

        axis_source = self.root / "axis_warning.h5ad"
        make_file(axis_source, values=np.ones((3, 2), dtype=np.float32))
        with h5py.File(axis_source, "r+") as handle:
            del handle["obs/_index"]
            handle["obs"].create_dataset("_index", data=np.asarray(["c0"], dtype=h5py.string_dtype("utf-8")))

        spatial_source = self.root / "spatial_warning.h5ad"
        make_file(spatial_source, spatial=True, values=np.ones((3, 2), dtype=np.float32))
        with h5py.File(spatial_source, "r+") as handle:
            del handle["obsm/spatial_3D"]
            handle["obsm"].create_dataset("spatial_3D", data=np.ones((2, 3), dtype=np.float32))

        panel_source = self.root / "panel_warning.h5ad"
        make_file(panel_source)
        panel = self.root / "panel_warning.txt"
        panel.write_text("A\nA\nB\n", encoding="utf-8")

        scenarios = {
            "expression": (expression_source, None, "Unsupported /X group encoding"),
            "axis": (axis_source, None, "obs index length"),
            "spatial": (spatial_source, None, "spatial_3D has 2 rows"),
            "panel": (panel_source, panel, "Panel duplicate names"),
        }
        for name, (source, optional_panel, expected_text) in scenarios.items():
            with self.subTest(name=name):
                report = inspect_h5ad(source, optional_panel)
                destination = self.root / f"propagation_{name}"
                outputs = write_reports(report, destination)
                payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
                markdown = outputs["markdown"].read_text(encoding="utf-8")
                terminal = terminal_summary(report)
                if name == "expression":
                    self.assertIn(expected_text, payload["anndata_structure"]["x"]["issue"])
                elif name == "axis":
                    self.assertIn(expected_text, payload["anndata_structure"]["axis_consistency"]["issues"][0])
                elif name == "spatial":
                    self.assertIn(expected_text, payload["spatial_3D_summary"]["issue"])
                else:
                    self.assertEqual(payload["gene_panel"]["duplicates_in_panel"], ["A"])
                self.assertIn(expected_text, terminal)
                self.assertIn(expected_text, markdown)


if __name__ == "__main__":
    unittest.main()
