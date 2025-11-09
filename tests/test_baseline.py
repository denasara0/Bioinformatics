"""
Tests for baseline MSA implementation.
"""

import pytest
from src.baseline_msa import BaselineMSA


def test_pairwise_alignment():
    """Test pairwise alignment."""
    msa = BaselineMSA()
    seq1 = "ACGT"
    seq2 = "ACGT"
    
    aligned1, aligned2, score = msa.align_pairwise(seq1, seq2)
    
    assert aligned1 == aligned2 == "ACGT"
    assert score == 8  # 4 matches * 2 points each


def test_pairwise_with_gaps():
    """Test pairwise alignment with gaps."""
    msa = BaselineMSA()
    seq1 = "ACGT"
    seq2 = "AGT"
    
    aligned1, aligned2, score = msa.align_pairwise(seq1, seq2)
    
    # Should align A-G-T with gap in seq1
    assert len(aligned1) == len(aligned2)
    assert score > 0


def test_sum_of_pairs_score():
    """Test SP score calculation."""
    msa = BaselineMSA()
    alignment = ["ACGT", "ACGT", "ACGT"]
    
    score = msa.sum_of_pairs_score(alignment)
    # 3 sequences: 3 pairs, each with 4 matches
    # Each pair: 4 matches * 2 = 8 points
    # Total: 3 pairs * 8 = 24
    assert score == 24


def test_progressive_align():
    """Test progressive alignment."""
    msa = BaselineMSA()
    sequences = ["ACGT", "ACGT", "ACGT"]
    
    alignment = msa.progressive_align(sequences)
    
    assert len(alignment) == 3
    assert all(len(seq) == len(alignment[0]) for seq in alignment)


if __name__ == "__main__":
    pytest.main([__file__])

