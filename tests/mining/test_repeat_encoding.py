"""
Test repeat encoding functions.
"""

import pytest
from thesis.mining.repeat_encoding import (
    encode_runs,
    encode_sequence_of_itemsets,
    decode_sequence_string,
    filter_redundant_sequences,
)
import pandas as pd


def test_encode_runs_basic():
    """Test basic run encoding."""
    seq = [
        "host:webserver",
        "host:webserver",
        "host:webserver",
        "host:webserver",
        "host:webserver",
        "event:login",
    ]
    result = encode_runs(seq)
    assert result == ["host:webserver__repeat_5_plus", "event:login"]


def test_encode_runs_various_counts():
    """Test various repeat counts."""
    seq = ["a", "b", "b", "c", "c", "c", "c", "d", "d", "d", "d", "d"]
    result = encode_runs(seq)
    assert result == ["a", "b__repeat_2", "c__repeat_3_4", "d__repeat_5_plus"]


def test_encode_runs_single_items():
    """Test single items are kept as-is."""
    seq = ["a", "b", "c"]
    result = encode_runs(seq)
    assert result == ["a", "b", "c"]


def test_encode_sequence_of_itemsets():
    """Test encoding sequences of itemsets (for prefixspan)."""
    seq = [
        {"host:webserver", "event:login"},
        {"host:webserver", "event:login"},
        {"event:logout"},
    ]
    result = encode_sequence_of_itemsets(seq)
    assert len(result) == 2
    assert result[0] == {"host:webserver_repeat_2", "event:login_repeat_2"}
    assert result[1] == {"event:logout"}


def test_decode_sequence_string():
    """Test decoding removes repeat markers."""
    seq_str = "host:webserver__repeat_5_plus -> event:login__repeat_2 -> event:logout"
    decoded = decode_sequence_string(seq_str)
    assert decoded == "host:webserver -> event:login -> event:logout"


def test_filter_redundant_sequences():
    """Test filtering near-duplicate sequences."""
    df = pd.DataFrame(
        [
            {
                "sequence_str": "host:webserver__repeat_5_plus -> event:login",
                "support_count": 50,
            },
            {
                "sequence_str": "host:webserver__repeat_3_4 -> event:login",
                "support_count": 45,
            },
            {
                "sequence_str": "host:webserver__repeat_2 -> event:login",
                "support_count": 40,
            },
            {"sequence_str": "event:logout", "support_count": 100},
        ]
    )

    filtered = filter_redundant_sequences(df, keep="highest_support")

    # Should keep only 2 sequences:
    # 1. The first group (host:webserver -> event:login) with highest support
    # 2. The event:logout sequence
    assert len(filtered) == 2
    assert 50 in filtered["support_count"].values  # highest from first group
    assert 100 in filtered["support_count"].values  # event:logout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
