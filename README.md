# MSc Thesis NeSy System for FP reduction

## Install
```bash
pip install -e ".[dev]"
```

## Run CLI
Right now still a dummy
```bash
thesis-cli init
thesis-cli show-config
thesis-cli mine --run-name debug
```

## Run API
```bash
uvicorn thesis.api.main:app --reload
```

## Run tests
```bash
pytest
```
