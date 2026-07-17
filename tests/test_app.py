import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


streamlit = ModuleType("streamlit")
_resource_cache = {}


def cache_resource(**_kwargs):
    def decorate(function):
        key = (function.__module__, function.__qualname__)

        def cached(*args, **kwargs):
            cache_key = (key, args, tuple(sorted(kwargs.items())))
            if cache_key not in _resource_cache:
                _resource_cache[cache_key] = function(*args, **kwargs)
            return _resource_cache[cache_key]

        def clear():
            for cache_key in list(_resource_cache):
                if cache_key[0] == key:
                    del _resource_cache[cache_key]

        cached.clear = Mock(side_effect=clear)
        return cached

    return decorate


streamlit.cache_resource = cache_resource
streamlit.cache_data = cache_resource

with patch.dict("sys.modules", {"streamlit": streamlit}):
    from app import app


class DatabasePathTests(unittest.TestCase):
    def setUp(self):
        state = app._database_path_state()
        with state["lock"]:
            state["last_valid_path"] = None

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
            state = app._database_path_state()
            state["last_valid_path"] = database
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                self.assertEqual(app._resolve_database_path(), database)

    def test_download_failure_uses_last_valid_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            state = app._database_path_state()
            state["last_valid_path"] = database
            with (
                patch.object(app, "_dataset_revision", return_value="new-revision"),
                patch.object(app, "_database_path", side_effect=RuntimeError("CAS down")),
            ):
                self.assertEqual(app._resolve_database_path(), database)

    def test_invalid_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "observatory.db"
            invalid.write_bytes(b"not sqlite")
            state = app._database_path_state()
            state["last_valid_path"] = str(invalid)
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                with self.assertRaises(sqlite3.DatabaseError):
                    app._resolve_database_path()

    def test_wrong_schema_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "observatory.db"
            with sqlite3.connect(invalid) as conn:
                conn.execute("CREATE TABLE unrelated (value TEXT)")
            state = app._database_path_state()
            state["last_valid_path"] = str(invalid)
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                with self.assertRaises(sqlite3.DatabaseError):
                    app._resolve_database_path()

    def test_last_valid_database_survives_streamlit_script_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with patch.object(app, "hf_hub_download", return_value=database):
                app._database_path("rerun-regression-revision")

            def rerun_database_path_state():
                return {"last_valid_path": None, "lock": object()}

            rerun_database_path_state.__module__ = app.__name__
            rerun_database_path_state.__qualname__ = "_database_path_state"
            app._database_path_state = cache_resource()(rerun_database_path_state)

            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                self.assertEqual(app._resolve_database_path(), database)

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
