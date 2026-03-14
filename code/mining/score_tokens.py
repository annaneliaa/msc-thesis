import pandas as pd
import numpy as np

# -----------------------------------
# Scorers
# -----------------------------------
def benign_prev_scorer():
    def _benign_prevalence_score(c0, c1, n0, n1):
        """
        Compute the percentage of benign alerts covered by a candidate as a measure of token importance.
        Benign prevalence is measurable anytime, also in single-class windows.
        This benign prevalance alone is not a good measure of safety.

        - c0: the number of benign alerts in window k containing an itemset X.
        - n0: the total number of benign alerts in window k.

        c1 and n1 not used but needed to match the scorer signature
        """
        idx = c0.index.union(c1.index)
        c0 = c0.reindex(idx, fill_value=0)

        if n0 == 0:
            return pd.Series(0.0, index=idx, dtype=float)

        return c0 / n0

    return _benign_prevalence_score


# Don't use, score is computed post hoc. Cannot let the system reason (rank) based on this score bc of single class problem
def fp_contrast_scorer(alpha: float = 0.5):
    def _log_odds_contrast_score(c0, c1, n0, n1):
        """
        Smoothed log-odds contrast:

            log((c0+α)/(n0-c0+α)) - log((c1+α)/(n1-c1+α))

        Positive values means that a candidate is more benign-associated.
        """
        idx = c0.index.union(c1.index)
        c0 = c0.reindex(idx, fill_value=0)
        c1 = c1.reindex(idx, fill_value=0)

        if n0 == 0:
            left = pd.Series(0.0, index=idx, dtype=float)
        else:
            left = np.log((c0 + alpha) / ((n0 - c0) + alpha))

        if n1 == 0:
            right = pd.Series(0.0, index=idx, dtype=float)
        else:
            right = np.log((c1 + alpha) / ((n1 - c1) + alpha))

        return left - right

    return _log_odds_contrast_score


# This scorer is basically a combination of the two above
def split_metric_scorer(alpha: float = 0.5):
    def _coverage_risk_score(c0, c1, n0, n1):
        """
        Compute per-candidate coverage and risk log-odds scores.

        - Coverage: smoothed log-odds of appearing in benign alerts.
        - Risk: smoothed log-odds of appearing in attack alerts.

        Coverage is defined when n0 > 0; risk when n1 > 0.
        Returns a DataFrame with columns ["coverage", "risk"] indexed by candidate.
        """
        idx = c0.index.union(c1.index)
        c0 = c0.reindex(idx, fill_value=0)
        c1 = c1.reindex(idx, fill_value=0)

        if n0 > 0:
            coverage = np.log((c0 + alpha) / ((n0 - c0) + alpha))
        else:
            coverage = pd.Series(0.0, index=idx, dtype=float)

        if n1 > 0:
            risk = np.log((c1 + alpha) / ((n1 - c1) + alpha))
        else:
            risk = pd.Series(np.nan, index=idx, dtype=float)

        return pd.DataFrame({"coverage": coverage, "risk": risk})

    return _coverage_risk_score


def split_metric_scorer_bayes(alpha: float = 0.5):
    def _coverage_risk_score_b(c0, c1, n0, n1):

        # Align candidates so they appear in both series
        idx = c0.index.union(c1.index)

        # Fill in missing candidates with zero counts
        c0 = c0.reindex(idx, fill_value=0)
        c1 = c1.reindex(idx, fill_value=0)

        # Convert counts to probabilities with Bayesian smoothing
        # Posterior probabilities with symmetric Beta prior
        p0 = (c0 + alpha) / (n0 + 2 * alpha) if n0 > 0 else pd.Series(0.5, index=idx)
        p1 = (c1 + alpha) / (n1 + 2 * alpha) if n1 > 0 else pd.Series(0.5, index=idx)

        coverage = np.log(p0 / (1 - p0))
        risk = np.log(p1 / (1 - p1))

        return pd.DataFrame(
            {
                "bayes_log_odds_benign": coverage,
                "bayes_log_odds_attack": risk,
                "bayes_log_odds_ratio": np.log(p0 / p1),
            }
        )

    return _coverage_risk_score_b
