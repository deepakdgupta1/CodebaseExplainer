
import pytest
from unittest.mock import patch, MagicMock
from codehierarchy.core.llm.model_manager import ModelManager
from codehierarchy.config.schema import LLMConfig
import subprocess


@pytest.fixture
def mock_config():
    config = MagicMock(spec=LLMConfig)
    config.model_name = "test-model-v1"
    config.gpu_offload_ratio = 0.5
    config.context_window = 2048
    config.use_mmap = True
    return config


@pytest.fixture
def mock_shutil_which():
    with patch("shutil.which") as mock:
        mock.return_value = "/usr/bin/lms"
        yield mock


class TestModelManager:

    @patch("subprocess.run")
    def test_init_checks_lms_version(
        self, mock_run, mock_config, mock_shutil_which
    ):
        """Test that init verifies lms version."""
        mock_run.return_value.returncode = 0

        manager = ModelManager(mock_config)

        assert manager.lms_path == "/usr/bin/lms"
        mock_run.assert_called_with(
            ["/usr/bin/lms", "--version"], check=True, capture_output=True
        )

    def test_init_lms_not_found(self, mock_config):
        """Test initialization when lms is not installed."""
        with patch("shutil.which", return_value=None):
            manager = ModelManager(mock_config)
            assert manager.lms_path is None

    @patch("codehierarchy.core.llm.model_manager.ModelManager._is_server_running")
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_load_model_success(
        self, mock_sleep, mock_run, mock_is_running, mock_config, mock_shutil_which
    ):
        """Test successful model loading flow."""
        mock_is_running.return_value = True

        # Setup subprocess mocks
        # 1. lms --version (in init)
        # 2. lms load
        # 3. lms ps
        mock_run.side_effect = [
            MagicMock(returncode=0),  # init
            MagicMock(returncode=0, stdout="Model loaded"),  # load
            MagicMock(returncode=0, stdout="test-model-v1")  # ps
        ]

        manager = ModelManager(mock_config)
        result = manager.load_model()

        assert result is True
        # Verify load command structure
        call_args = mock_run.call_args_list[1][0][0]
        assert call_args[:3] == ["/usr/bin/lms", "load", "test-model-v1"]
        assert "--mmap" not in call_args  # confirm bug fix

    @patch("codehierarchy.core.llm.model_manager.ModelManager._is_server_running")
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_auto_start_server(
        self, mock_sleep, mock_run, mock_is_running, mock_config, mock_shutil_which
    ):
        """Test that server is started if not running."""
        # First check False, subsequent True
        mock_is_running.side_effect = [False, True]

        # Setup subprocess mocks
        # 1. init
        # 2. load
        # 3. server start
        # 4. ps
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0),
            MagicMock(returncode=0),
            MagicMock(returncode=0, stdout="test-model-v1")
        ]

        manager = ModelManager(mock_config)
        manager.load_model()

        # Verify server start command
        start_cmd = ["/usr/bin/lms", "server", "start", "--port", "1234"]
        mock_run.assert_any_call(
            start_cmd, check=True, capture_output=True, text=True
        )

    @patch("subprocess.run")
    def test_load_model_subprocess_error(
        self, mock_run, mock_config, mock_shutil_which
    ):
        """Test handling of subprocess failure during load."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # init
            subprocess.CalledProcessError(
                1, ["lms", "load"], stderr="Insufficient memory"
            )
        ]

        manager = ModelManager(mock_config)
        result = manager.load_model()

        assert result is False

    def test_is_server_running_true(self, mock_config, mock_shutil_which):
        """Test socket check returns True when port open."""
        with patch("subprocess.run"):
            manager = ModelManager(mock_config)

        with patch("socket.socket") as mock_socket:
            mock_ctx = mock_socket.return_value.__enter__.return_value
            mock_ctx.connect_ex.return_value = 0
            assert manager._is_server_running() is True

    def test_is_server_running_false(self, mock_config, mock_shutil_which):
        """Test socket check returns False when port closed."""
        with patch("subprocess.run"):
            manager = ModelManager(mock_config)

        with patch("socket.socket") as mock_socket:
            mock_ctx = mock_socket.return_value.__enter__.return_value
            mock_ctx.connect_ex.return_value = 111  # Connection refused
            assert manager._is_server_running() is False
