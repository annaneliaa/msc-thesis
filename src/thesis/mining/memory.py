def mem_score(cov_mem, risk_mem, cand, mem_lambda=1.0):
    """
    Compute the symbolic memory score for a candidate feature.

    The score combines two memory components:
    - Coverage memory: how consistently the candidate explains benign alerts.
    - Risk memory: how strongly the candidate is associated with attack windows.

    The final score is computed as:
        coverage_score − λ * risk_score

    where λ controls how strongly past risk associations suppress activation
    of otherwise benign-looking candidates, i.e. how conservative the system is.

    Args:
        cov_mem:
            Coverage memory object storing benign-related feature scores.
        risk_mem:
            Risk memory object storing attack-related feature scores.
        cand (str):
            Candidate token or itemset identifier.
        l (float, optional):
            Risk penalty weight (λ). Higher values increase suppression
            from risk memory. Defaults to 1.0.

    Returns:
        float:
            Memory-based adjustment score for the candidate.
            Positive values favor activation; negative values suppress it.
    """
    c = cov_mem.scores.get(f"cov::{cand}", 0.0)
    r = risk_mem.scores.get(f"risk::{cand}", 0.0)
    return (
        c - mem_lambda * r
    )  # minus because risk should limit activation of seemingly benign candidates


def compute_memory_scores(ranking_k, cov_mem, risk_mem, mem_lambda=1.0):
    """
    For each candidate, compute a memory-based score using coverage and risk memories.
    Returns dataframe with mem_score added.

    Args:
        ranking_k (pd.DataFrame):
            Current window ranking with at least columns:
            ["candidate", "contrast_score"].
        cov_mem:
            Coverage memory object.
        risk_mem:
            Risk memory object.
        mem_lambda (float):
            Weighting factor passed to memory scoring function.

    Returns:
        pd.DataFrame:
            Same dataframe with computed memory scores added as a column.
    """
    ranking_k = ranking_k.copy()
    ranking_k["mem_score"] = ranking_k["candidate"].map(
        lambda cand: mem_score(cov_mem, risk_mem, cand, mem_lambda=mem_lambda)
    )
    return ranking_k


def apply_utility_rerank(ranking_k, mem_beta=0.1):
    """
    Computes utility score for each proposed candidate in window k as
    score = contrast_score + mem_beta * mem_score

    Re-rank mined candidates using this computed utility score.

    For each candidate:
    - Combines the raw mining score with the memory score.
    - Returns a re-ranked dataframe.

    Args:
        ranking_k (pd.DataFrame):
            Current window ranking with at least columns:
            ["candidate", "contrast_score"].
        cov_mem:
            Coverage memory object.
        risk_mem:
            Risk memory object.
        mem_lambda (float):
            Weighting factor passed to memory scoring function.
        mem_beta (float):
            Scaling factor controlling influence of memory score
            on final ranking.

    Returns:
        pd.DataFrame:
            Re-ranked dataframe sorted by ("combined_score"), which is the contrast score in the current window combined
            with the memory score. Contains all original columns plus "mem_score" and "combined_score".
    """
    ranking_k = ranking_k.copy()
    ranking_k["combined_score"] = (
        ranking_k["contrast_score"] + mem_beta * ranking_k["mem_score"]
    )
    return ranking_k.sort_values("combined_score", ascending=False).reset_index(
        drop=True
    )


# TODO: add here removal of candidates whose score drops below threshold?
def update_memories_and_snapshot(
    ranking_k,
    cov_mem,
    risk_mem,
    n_benign,
    window_has_attack,
    start_k,
    end_k,
    top_cov=50,
    top_risk=50,
):
    """
    Update coverage and risk memories of candidates based on current window
    ranking, and return a snapshot of memory state.

    Steps:
    - Apply decay to both memories.
    - Reward top coverage candidates if benign alerts exist.
    - Reward top risk candidates if attacks occurred.
    - Return metadata and current memory state.

    Args:
        ranking_k (pd.DataFrame):
            Current window ranking containing candidate scores.
        cov_mem:
            Coverage memory object (to track benign-associated features).
        risk_mem:
            Risk memory object (to track attack-associated features).
        n_benign (int):
            Number of benign alerts in the window.
        window_has_attack (bool):
            Whether the window contains attack alerts.
        start_k (pd.Timestamp):
            Window start time.
        end_k (pd.Timestamp):
            Window end time.
        top_cov (int):
            Number of top coverage candidates to reward.
        top_risk (int):
            Number of top risk candidates to reward.

    Returns:
        dict:
            Snapshot containing window bounds, attack flag,
            active memory entries, and current memory score maps.
    """
    cov_mem.step_decay()
    risk_mem.step_decay()

    if n_benign > 0 and "coverage" in ranking_k.columns:
        cov_top = ranking_k.nlargest(top_cov, "coverage")["candidate"]
        cov_mem.reward_feats([f"cov::{it}" for it in cov_top])

    # window is dual-class
    if window_has_attack and "risk" in ranking_k.columns:
        tmp = ranking_k.dropna(subset=["risk"])
        if not tmp.empty:
            risk_top = tmp.nlargest(top_risk, "risk")["candidate"]
            risk_mem.reward_feats([f"risk::{it}" for it in risk_top])

    return {
        "start": start_k,
        "end": end_k,
        "has_attack": window_has_attack,
        "coverage_active": cov_mem.active(),
        "risk_active": risk_mem.active(),
        "coverage_scores": dict(cov_mem.scores),
        "risk_scores": dict(risk_mem.scores),
    }
