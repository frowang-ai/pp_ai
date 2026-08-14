import pytest
from pi_ai.utils.provider_env import get_provider_env_value


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PI_TEST_PROVIDER_ENV_VALUE", raising=False)


def test_returns_none_when_not_set_anywhere():
    assert get_provider_env_value("PI_TEST_PROVIDER_ENV_VALUE") is None


def test_returns_scoped_env_value_when_present(monkeypatch):
    monkeypatch.delenv("PI_TEST_PROVIDER_ENV_VALUE", raising=False)
    assert get_provider_env_value("PI_TEST_PROVIDER_ENV_VALUE", {"PI_TEST_PROVIDER_ENV_VALUE": "scoped"}) == "scoped"


def test_falls_back_to_os_environ_when_scoped_env_missing_key(monkeypatch):
    monkeypatch.setenv("PI_TEST_PROVIDER_ENV_VALUE", "from-os")
    assert get_provider_env_value("PI_TEST_PROVIDER_ENV_VALUE", {}) == "from-os"


def test_falls_back_to_os_environ_when_scoped_value_is_empty_string(monkeypatch):
    monkeypatch.setenv("PI_TEST_PROVIDER_ENV_VALUE", "from-os")
    assert get_provider_env_value("PI_TEST_PROVIDER_ENV_VALUE", {"PI_TEST_PROVIDER_ENV_VALUE": ""}) == "from-os"


def test_scoped_env_takes_priority_over_os_environ(monkeypatch):
    monkeypatch.setenv("PI_TEST_PROVIDER_ENV_VALUE", "from-os")
    assert (
        get_provider_env_value("PI_TEST_PROVIDER_ENV_VALUE", {"PI_TEST_PROVIDER_ENV_VALUE": "from-scope"})
        == "from-scope"
    )


def test_reads_os_environ_when_env_argument_omitted(monkeypatch):
    monkeypatch.setenv("PI_TEST_PROVIDER_ENV_VALUE", "direct")
    assert get_provider_env_value("PI_TEST_PROVIDER_ENV_VALUE") == "direct"


def test_returns_none_for_empty_os_environ_value(monkeypatch):
    monkeypatch.setenv("PI_TEST_PROVIDER_ENV_VALUE", "")
    assert get_provider_env_value("PI_TEST_PROVIDER_ENV_VALUE") is None
