import pytest
from baseline_msa import BaselineMSA


def test_pairwise_alignment():
    """Test pairwise alignment."""
    msa = BaselineMSA()
    seq1 = "ACGT"
    seq2 = "ACGT"
    
    aligned1, aligned2, score = msa.align_pairwise(seq1, seq2)
    
    assert aligned1 == aligned2 == "ACGT"
    assert score == 8


def test_pairwise_with_gaps():
    """Test pairwise alignment with gaps."""
    msa = BaselineMSA()
    seq1 = "ACGT"
    seq2 = "AGT"
    
    aligned1, aligned2, score = msa.align_pairwise(seq1, seq2)
    
    # Should align A-G-T with gap in seq1
    assert len(aligned1) == len(aligned2)
    assert score > 0


def test_progressive_align():
    """Test progressive alignment."""
    msa = BaselineMSA()
    sequences = ["ACGT", "ACGT", "ACGT"]
    
    alignment = msa.progressive_align(sequences)
    
    assert len(alignment) == 3
    assert all(len(seq) == len(alignment[0]) for seq in alignment)


if __name__ == "__main__":
    pytest.main([__file__])

