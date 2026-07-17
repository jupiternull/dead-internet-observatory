import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


streamlit = ModuleType("streamlit")


def cache_resource(**_kwargs):
    def decorate(function):
        function.clear = Mock()
        return function

    return decorate


streamlit.cache_resource = cache_resource
streamlit.cache_data = cache_resource

with patch.dict("sys.modules", {"streamlit": streamlit}):
    from app import app


class DatabasePathTests(unittest.TestCase):
    def setUp(self):
        app._last_valid_database_path = None

    def _database(self, directory):
        path = Path(directory) / "observatory.db"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE composite_index "
                "(date TEXT, aliveness_index REAL, smoothed_index REAL, n_docs INTEGER, "
                "anomaly_flag INTEGER, anomaly_reason TEXT)"
            )
            conn.execute(
                "CREATE TABLE daily_index "
                "(date TEXT, source TEXT, mean_score REAL, aliveness_index REAL, n_docs INTEGER)"
            )
            conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        return str(path)

    def test_database_path_downloads_revision_with_huggingface_hub(self):
        app._database_path.clear()
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with patch.object(
                app, "hf_hub_download", return_value=database
            ) as download:
                path = app._database_path("dataset-revision")

            download.assert_called_once_with(
                repo_id=app.DATASET_REPO,
                repo_type="dataset",
                filename="observatory.db",
                revision="dataset-revision",
            )
            self.assertEqual(path, database)

    def test_api_failure_uses_last_valid_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            app._last_valid_database_path = database
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                self.assertEqual(app._resolve_database_path(), database)

    def test_download_failure_uses_last_valid_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            app._last_valid_database_path = database
            with (
                patch.object(app, "_dataset_revision", return_value="new-revision"),
                patch.object(app, "_database_path", side_effect=RuntimeError("CAS down")),
            ):
                self.assertEqual(app._resolve_database_path(), database)

    def test_invalid_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "observatory.db"
            invalid.write_bytes(b"not sqlite")
            app._last_valid_database_path = str(invalid)
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                with self.assertRaises(sqlite3.DatabaseError):
                    app._resolve_database_path()

    def test_wrong_schema_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "observatory.db"
            with sqlite3.connect(invalid) as conn:
                conn.execute("CREATE TABLE unrelated (value TEXT)")
            app._last_valid_database_path = str(invalid)
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                with self.assertRaises(sqlite3.DatabaseError):
                    app._resolve_database_path()

    def test_query_loaders_keep_the_resolved_snapshot(self):
        snapshot = "/cache/snapshots/revision/observatory.db"
        responses = [
            [{"smoothed_index": 58.1}],
            [{"value": "3440000"}],
            [],
            [],
        ]
        with (
            patch.object(app, "_query", side_effect=responses) as query,
            patch.object(app.pd, "DataFrame", return_value=Mock(empty=True)),
        ):
            app.load_score(snapshot)
            app.load_total_docs(snapshot)
            app.load_timeline(snapshot)
            app.load_sources(snapshot)

        self.assertEqual([call.args[0] for call in query.call_args_list], [snapshot] * 4)


if __name__ == "__main__":
    unittest.main()
