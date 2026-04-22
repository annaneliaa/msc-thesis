from thesis.mining.mining_dummy_job import run_dummy_mining_job


def test_dummy_job_creates_file():
    path = run_dummy_mining_job("pytest_run")
    assert path.exists()
