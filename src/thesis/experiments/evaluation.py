import numpy as np
from sklearn.metrics import roc_auc_score


def eval_subset_metrics(X_full, y, res_model, threshold=0.5, subset_col=None):
    split = res_model["test_idx_start"]

    y_test = np.asarray(y)[split:]
    p = np.asarray(res_model["proba_test"])
    pred = (p >= threshold).astype(int)

    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    alerts = int(pred.sum())

    auc = float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else np.nan

    subset_size = None
    if subset_col is not None:
        subset_size = int(X_full[subset_col].iloc[split:].sum())

    return {
        "auc": auc,
        "fp": fp,
        "fn": fn,
        "alerts": alerts,
        "subset_size": subset_size,
    }
