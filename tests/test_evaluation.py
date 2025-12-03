import pytest
from evaluation import MSAEvaluator


def test_sum_of_pairs_score():
    """Test SP score calculation."""
    evaluator = MSAEvaluator()
    alignment = ["ACGT", "ACGT", "ACGT"]
    
    score = evaluator.sum_of_pairs_score(alignment)

    assert score == 24


if __name__ == "__main__":
    pytest.main([__file__])
