from thesis.config import load_settings


def test_load_settings():
    settings = load_settings()
    assert settings.app.name == "msc-thesis"