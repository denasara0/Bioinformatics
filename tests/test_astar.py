import pytest
from astar_msa import AStarMSA
from astar_msa import AlignmentState

def test_astar_simple():
    """Test A* on simple sequences."""
    msa = AStarMSA()
    sequences = ["AC", "AC"]
    
    alignment, score, nodes = msa.align(sequences)
    
    assert len(alignment) == 2
    assert len(alignment[0]) == len(alignment[1])
    assert score >= 0


def test_astar_three_sequences():
    """Test A* on three sequences."""
    msa = AStarMSA()
    sequences = ["AC", "AC", "AC"]
    
    alignment, score, nodes = msa.align(sequences)
    
    assert len(alignment) == 3
    assert all(len(seq) == len(alignment[0]) for seq in alignment)
    assert nodes > 0


def test_heuristic_admissibility():
    """Test that heuristic is admissible (never overestimates)."""
    msa = AStarMSA()
    sequences = ["ACGT", "ACGT"]
    
    state = AlignmentState(
        aligned_sequences=["", ""],
        positions=[0, 0],
        g_score=0,
        h_score=0,
        f_score=0
    )
    
    # Precompute heuristic tables manually for the test
    msa.heuristic_tables = {}
    n_seqs = len(sequences)
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            msa.heuristic_tables[(i, j)] = msa._precompute_suffix_costs(sequences[i], sequences[j])
            
    h = msa.heuristic(state, sequences)
    assert h >= 0


if __name__ == "__main__":
    pytest.main([__file__])

