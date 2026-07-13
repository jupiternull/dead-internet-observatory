import unittest
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
    def test_database_path_downloads_revision_with_huggingface_hub(self):
        app._database_path.clear()
        session = Mock()
        session.get.return_value = Mock(content=b"database")

        with (
            patch.object(app, "_http_session", return_value=session),
            patch.object(
                app,
                "hf_hub_download",
                create=True,
                return_value="/cache/observatory.db",
            ) as download,
        ):
            path = app._database_path("dataset-revision")

        download.assert_called_once_with(
            repo_id=app.DATASET_REPO,
            repo_type="dataset",
            filename="observatory.db",
            revision="dataset-revision",
        )
        self.assertEqual(path, "/cache/observatory.db")


if __name__ == "__main__":
    unittest.main()
