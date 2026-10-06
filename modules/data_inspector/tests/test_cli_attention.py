"""Real synthetic HDF5 CLI checks, not official-board or biological validation."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import h5py
import numpy as np
from vec_data_inspector.cli import main


class AttentionExitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='vec_attention_synthetic_')
        self.root = Path(self.directory.name)
        self.input = self.root / 'synthetic.h5ad'

    def tearDown(self):
        self.directory.cleanup()

    def write(self, values=None, *, obs_length=2):
        if values is None:
            values = np.asarray([[1., 0.], [0., 2.]], dtype=np.float32)
        with h5py.File(self.input, 'w') as handle:
            handle.create_dataset('X', data=values)
            obs = handle.create_group('obs')
            obs.attrs['_index'] = '_index'
            obs.create_dataset('_index', data=[f'fake_cell_{i}' for i in range(obs_length)], dtype=h5py.string_dtype())
            var = handle.create_group('var')
            var.attrs['_index'] = '_index'
            var.create_dataset('_index', data=['fake_gene_A', 'fake_gene_B'], dtype=h5py.string_dtype())

    def run_cli(self, *, flag=True, suffix='report', panel=None):
        before = hashlib.sha256(self.input.read_bytes()).hexdigest()
        arguments = [str(self.input), '--report-dir', str(self.root / suffix)]
        if flag:
            arguments.append('--fail-on-attention')
        if panel is not None:
            arguments.extend(['--panel', str(panel)])
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(arguments)
        self.assertEqual(before, hashlib.sha256(self.input.read_bytes()).hexdigest())
        return status, stdout.getvalue(), stderr.getvalue()

    def assert_report(self, suffix='report'):
        path = self.root / suffix / 'inspection_report.json'
        self.assertTrue(path.is_file())
        return json.loads(path.read_text())

    def test_clean_structural_report_returns_zero_with_or_without_flag(self):
        self.write()
        self.assertEqual(self.run_cli(flag=False, suffix='default')[0], 0)
        self.assertEqual(self.run_cli(suffix='flag')[0], 0)
        self.assert_report('default')
        self.assert_report('flag')

    def test_negative_values_default_zero_flag_one_reports_preserved(self):
        self.write(np.asarray([[-1., 0.], [0., 2.]], dtype=np.float32))
        self.assertEqual(self.run_cli(flag=False, suffix='default')[0], 0)
        status, stdout, stderr = self.run_cli(suffix='flag')
        self.assertEqual(status, 1)
        self.assertIn('reports preserved', stderr)
        self.assertIn('attention', stdout)
        self.assertTrue(self.assert_report('flag')['anndata_structure']['x']['negative_values_detected'])

    def test_nonfinite_values_return_one_without_dropping_report(self):
        self.write(np.asarray([[np.nan, 0.], [0., 2.]], dtype=np.float32))
        self.assertEqual(self.run_cli()[0], 1)
        self.assertFalse(self.assert_report()['anndata_structure']['x']['finite'])

    def test_not_assessed_complex_values_return_one(self):
        self.write(np.asarray([[1+2j, 0], [0, 2]], dtype=np.complex64))
        self.assertEqual(self.run_cli()[0], 1)
        self.assertFalse(self.assert_report()['anndata_structure']['x']['numeric_qc_performed'])

    def test_axis_mismatch_returns_one(self):
        self.write(obs_length=1)
        self.assertEqual(self.run_cli()[0], 1)
        self.assertFalse(self.assert_report()['anndata_structure']['axis_consistency']['matches_x'])

    def test_wrong_panel_order_returns_one_and_preserves_source(self):
        self.write()
        panel = self.root / 'fake_panel.txt'
        panel.write_text('fake_gene_B\nfake_gene_A\n')
        self.assertEqual(self.run_cli(panel=panel)[0], 1)
        self.assertTrue(self.assert_report()['gene_panel']['same_unique_set_wrong_order'])

    def test_unreadable_file_keeps_error_two_and_no_report(self):
        self.input.write_text('synthetic non-HDF5 fixture')
        status, _, _ = self.run_cli()
        self.assertEqual(status, 2)
        self.assertFalse((self.root / 'report' / 'inspection_report.json').exists())


if __name__ == '__main__':
    unittest.main()
