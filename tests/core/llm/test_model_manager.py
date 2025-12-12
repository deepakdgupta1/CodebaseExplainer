"""
Tests for ModelManager class.

This module tests the LM Studio CLI integration including:
- Path resolution and bootstrap
- Server health checks
- Model loading lifecycle
"""

import pytest
from unittest.mock import patch, MagicMock
from codehierarchy.core.llm.model_manager import ModelManager
from codehierarchy.config.schema import LLMConfig
import subprocess


@pytest.fixture
def mock_config():
    """Create a mock LLMConfig for testing."""
    config = MagicMock(spec=LLMConfig)
    config.model_name = "test-model-v1"
    config.gpu_offload_ratio = 0.5
    config.context_window = 2048
    return config


@pytest.fixture
def mock_lms_path():
    """Mock shutil.which to return a valid lms path."""
    with patch("shutil.which") as mock:
        mock.return_value = "/usr/bin/lms"
        yield mock


@pytest.fixture
def mock_bootstrap():
    """Mock the _bootstrap_lms method to avoid subprocess calls during init."""
    with patch.object(ModelManager, "_bootstrap_lms"):
        yield


class TestModelManager:
    """Test suite for ModelManager class."""

    @patch("subprocess.run")
    def test_init_checks_lms_version(
        self, mock_run, mock_config, mock_lms_path
    ):
        """Test that init runs bootstrap and verifies lms CLI."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ModelManager(mock_config)

        assert manager.lms_path == "/usr/bin/lms"
        # Verify bootstrap was called (which calls 'lms bootstrap')
        mock_run.assert_any_call(
            ["/usr/bin/lms", "bootstrap"],
            check=True, capture_output=True, text=True
        )
        # Verify version check was called (uses 'version' not '--version')
        mock_run.assert_any_call(
            ["/usr/bin/lms", "version"], check=True, capture_output=True
        )

    def test_init_lms_not_found(self, mock_config):
        """Test initialization exits when lms is not installed."""
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.exists", return_value=False), \
             pytest.raises(SystemExit):
            ModelManager(mock_config)

    @patch("codehierarchy.core.llm.model_manager.requests.get")
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_load_model_success(
        self, mock_sleep, mock_run, mock_requests, mock_config, mock_lms_path
    ):
        """Test successful model loading flow."""
        # Mock bootstrap calls
        mock_run.return_value = MagicMock(returncode=0)

        # Mock server health check (returns True - server running)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_requests.return_value = mock_response

        manager = ModelManager(mock_config)

        # Now mock for load_model
        # First call: _get_loaded_model_ids returns empty
        # After load: returns the model
        mock_requests.side_effect = [
            MagicMock(status_code=200, json=lambda: {"data": []}),  # health check
            MagicMock(status_code=200, json=lambda: {"data": []}),  # get loaded models (empty)
            MagicMock(status_code=200, json=lambda: {"data": [{"id": "test-model-v1"}]}),  # after load
        ]

        result = manager.load_model()

        assert result == "test-model-v1"
        # Verify load command was called
        load_calls = [c for c in mock_run.call_args_list if "load" in str(c)]
        assert len(load_calls) >= 1

    @patch("codehierarchy.core.llm.model_manager.requests.get")
    @patch("subprocess.Popen")
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_auto_start_server(
        self, mock_sleep, mock_run, mock_popen, mock_requests, mock_config, mock_lms_path
    ):
        """Test that server is started if not running."""
        # Mock bootstrap
        mock_run.return_value = MagicMock(returncode=0)

        manager = ModelManager(mock_config)

        # Server not healthy initially, then healthy after start
        from requests.exceptions import ConnectionError
        mock_requests.side_effect = [
            ConnectionError(),  # First health check fails
            MagicMock(status_code=200, json=lambda: {"data": []}),  # After server starts
            MagicMock(status_code=200, json=lambda: {"data": []}),  # get_loaded_models
            MagicMock(status_code=200, json=lambda: {"data": [{"id": "test-model-v1"}]}),  # verify
        ]

        result = manager.load_model()

        # Verify Popen was called to start server
        mock_popen.assert_called()

    @patch("codehierarchy.core.llm.model_manager.requests.get")
    @patch("subprocess.run")
    def test_load_model_subprocess_error(
        self, mock_run, mock_requests, mock_config, mock_lms_path
    ):
        """Test handling of subprocess failure during load."""
        # Bootstrap succeeds
        mock_run.side_effect = [
            MagicMock(returncode=0),  # bootstrap
            MagicMock(returncode=0),  # version check
            subprocess.CalledProcessError(1, ["lms", "load"], stderr="Insufficient memory")
        ]

        # Server is healthy
        mock_requests.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": []}
        )

        manager = ModelManager(mock_config)
        result = manager.load_model()

        assert result is None

    @patch("codehierarchy.core.llm.model_manager.requests.get")
    @patch("subprocess.run")
    def test_is_server_healthy_true(self, mock_run, mock_requests, mock_config, mock_lms_path):
        """Test health check returns True when server responds 200."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ModelManager(mock_config)

        mock_requests.return_value = MagicMock(status_code=200)
        assert manager._is_server_healthy() is True

    @patch("codehierarchy.core.llm.model_manager.requests.get")
    @patch("subprocess.run")
    def test_is_server_healthy_false(self, mock_run, mock_requests, mock_config, mock_lms_path):
        """Test health check returns False when server not responding."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ModelManager(mock_config)

        from requests.exceptions import ConnectionError
        mock_requests.side_effect = ConnectionError()
        assert manager._is_server_healthy() is False

    @patch("codehierarchy.core.llm.model_manager.requests.get")
    @patch("subprocess.run")
    def test_get_loaded_model_ids(self, mock_run, mock_requests, mock_config, mock_lms_path):
        """Test fetching loaded model IDs from API."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ModelManager(mock_config)

        mock_requests.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "model1"}, {"id": "model2"}]}
        )

        ids = manager._get_loaded_model_ids()
        assert ids == ["model1", "model2"]

    @patch("codehierarchy.core.llm.model_manager.requests.get")
    @patch("subprocess.run")
    def test_model_already_loaded_skips_load(
        self, mock_run, mock_requests, mock_config, mock_lms_path
    ):
        """Test that load_model skips loading if model is already loaded."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ModelManager(mock_config)

        # Server healthy and model already loaded
        mock_requests.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "test-model-v1"}]}
        )

        result = manager.load_model()

        assert result == "test-model-v1"
        # Ensure 'lms load' was NOT called (model already loaded)
        load_calls = [c for c in mock_run.call_args_list
                      if len(c[0]) > 0 and "load" in str(c[0][0])]
        # Only bootstrap and version calls, no load
        assert all("load" not in str(c[0][0]) for c in mock_run.call_args_list
                   if len(c[0]) > 0 and isinstance(c[0][0], list))
