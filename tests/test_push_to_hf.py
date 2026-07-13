import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError
from scripts import push_to_hf


class RestoreTests(unittest.TestCase):
    def test_restore_downloads_state_with_huggingface_hub(self):
        with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as data_dir:
            cached_file = Path(cache_dir) / "doc_registry.parquet"
            cached_file.write_bytes(b"restored state")

            with (
                patch.object(push_to_hf, "STATE_FILES", ("gold/doc_registry.parquet",)),
                patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
                patch.object(
                    push_to_hf, "hf_hub_download", return_value=str(cached_file)
                ) as download,
            ):
                push_to_hf.restore("owner/dataset", Path(data_dir))

            download.assert_called_once_with(
                repo_id="owner/dataset",
                filename="gold/doc_registry.parquet",
                repo_type="dataset",
                token="test-token",
            )
            self.assertEqual(
                (Path(data_dir) / "gold/doc_registry.parquet").read_bytes(),
                b"restored state",
            )

    def test_restore_skips_missing_remote_state(self):
        missing = RemoteEntryNotFoundError("missing", response=Mock(status_code=404))

        with tempfile.TemporaryDirectory() as data_dir:
            with (
                patch.object(push_to_hf, "STATE_FILES", ("observatory.db",)),
                patch.object(push_to_hf, "hf_hub_download", side_effect=missing),
                redirect_stdout(StringIO()) as output,
            ):
                push_to_hf.restore("owner/dataset", Path(data_dir))

            self.assertIn("[HF] No prior observatory.db", output.getvalue())
            self.assertFalse((Path(data_dir) / "observatory.db").exists())

    def test_restore_transient_failure_uses_cached_local_state(self):
        transient = HfHubHTTPError("service unavailable", response=Mock(status_code=503))

        with tempfile.TemporaryDirectory() as data_dir:
            destination = Path(data_dir) / "observatory.db"
            destination.write_bytes(b"cached state")

            with (
                patch.object(push_to_hf, "STATE_FILES", ("observatory.db",)),
                patch.object(
                    push_to_hf, "hf_hub_download", side_effect=transient
                ) as download,
                patch.object(push_to_hf.time, "sleep"),
                redirect_stdout(StringIO()) as output,
            ):
                push_to_hf.restore("owner/dataset", Path(data_dir))

            self.assertEqual(download.call_count, 3)
            self.assertEqual(destination.read_bytes(), b"cached state")
            self.assertIn("using cached local state", output.getvalue())


if __name__ == "__main__":
    unittest.main()
