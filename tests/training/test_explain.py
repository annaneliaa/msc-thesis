from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from thesis.training.explain import (
    compute_lime_signed_importances,
    compute_shap_signed_importances,
)


@pytest.fixture
def linear_dataset():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=["a", "b", "c", "d"])
    # `a` drives the label; `d` is pure noise -- used below to check that a
    # signal feature outranks a noise feature in both explainers.
    y = (X["a"] > 0).astype(int).values
    return X, y


def test_shap_signed_importances_ranks_signal_feature_above_noise(linear_dataset):
    X, y = linear_dataset
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X.iloc[:150], y[:150])

    result = compute_shap_signed_importances(
        model,
        X_background=X.iloc[:150].sample(50, random_state=0),
        X_explain=X.iloc[150:],
        feature_names=list(X.columns),
        top_n=4,
    )
    ranked = sorted(result, key=lambda f: abs(result[f]), reverse=True)
    assert ranked[0] == "a"
    assert set(result.keys()) <= set(X.columns)


def test_shap_signed_importances_respects_top_n(linear_dataset):
    X, y = linear_dataset
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X.iloc[:150], y[:150])

    result = compute_shap_signed_importances(
        model,
        X_background=X.iloc[:150].sample(50, random_state=0),
        X_explain=X.iloc[150:170],
        feature_names=list(X.columns),
        top_n=2,
    )
    assert len(result) == 2


def test_lime_signed_importances_ranks_signal_feature_above_noise(linear_dataset):
    X, y = linear_dataset
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X.iloc[:150], y[:150])

    result = compute_lime_signed_importances(
        model,
        X_background=X.iloc[:150].sample(50, random_state=0),
        X_explain=X.iloc[150:165],
        feature_names=list(X.columns),
        top_n=4,
        num_samples=200,
        random_state=0,
    )
    ranked = sorted(
        result.importances, key=lambda f: abs(result.importances[f]), reverse=True
    )
    assert ranked[0] == "a"
    assert set(result.importances.keys()) == {"a", "b", "c", "d"}


def test_lime_signed_importances_reports_local_fidelity(linear_dataset):
    X, y = linear_dataset
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X.iloc[:150], y[:150])

    result = compute_lime_signed_importances(
        model,
        X_background=X.iloc[:150].sample(50, random_state=0),
        X_explain=X.iloc[150:165],
        feature_names=list(X.columns),
        top_n=4,
        num_samples=200,
        random_state=0,
    )
    # A linear model explained by a linear surrogate should fit near-perfectly.
    assert 0.0 <= result.mean_fidelity <= 1.0 + 1e-9
    assert result.mean_fidelity > 0.8


def test_lime_signed_importances_averages_over_every_explained_row(linear_dataset):
    X, y = linear_dataset
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X.iloc[:150], y[:150])

    single = compute_lime_signed_importances(
        model,
        X_background=X.iloc[:150].sample(50, random_state=0),
        X_explain=X.iloc[150:151],
        feature_names=list(X.columns),
        top_n=4,
        num_samples=200,
        random_state=0,
    )
    batch = compute_lime_signed_importances(
        model,
        X_background=X.iloc[:150].sample(50, random_state=0),
        X_explain=X.iloc[150:160],
        feature_names=list(X.columns),
        top_n=4,
        num_samples=200,
        random_state=0,
    )
    # Not asserting exact equality (different explained sets) -- just that
    # averaging over more rows doesn't blow up or drop features.
    assert set(single.importances.keys()) == set(batch.importances.keys())
