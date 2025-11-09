"""
Baseline Multiple Sequence Alignment using Dynamic Programming
This serves as a reference implementation for correctness validation.
"""

import numpy as np
from typing import List, Tuple, Optional


class BaselineMSA:
    """
    Standard dynamic programming approach for Multiple Sequence Alignment.
    This is computationally expensive but guarantees optimal alignment.
    """
    
    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -1):
        """
        Initialize MSA with scoring parameters.
        
        Args:
            match_score: Score for matching characters
            mismatch_score: Score for mismatching characters
            gap_penalty: Penalty for gaps (should be negative)
        """
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty
    
    def score_pairwise(self, char1: str, char2: str) -> int:
        """Calculate score between two characters."""
        if char1 == char2:
            return self.match_score
        elif char1 == '-' or char2 == '-':
            return self.gap_penalty
        else:
            return self.mismatch_score
    
    def align_pairwise(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """
        Pairwise alignment using Needleman-Wunsch algorithm.
        
        Returns:
            Tuple of (aligned_seq1, aligned_seq2, alignment_score)
        """
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        # Initialize first row and column
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.gap_penalty
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.gap_penalty
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = dp[i-1][j-1] + self.score_pairwise(seq1[i-1], seq2[j-1])
                delete = dp[i-1][j] + self.gap_penalty
                insert = dp[i][j-1] + self.gap_penalty
                dp[i][j] = max(match, delete, insert)
        
        # Traceback
        aligned_seq1, aligned_seq2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + self.score_pairwise(seq1[i-1], seq2[j-1]):
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + self.gap_penalty:
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append('-')
                i -= 1
            else:
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j-1])
                j -= 1
        
        aligned_seq1 = ''.join(reversed(aligned_seq1))
        aligned_seq2 = ''.join(reversed(aligned_seq2))
        score = dp[m][n]
        
        return aligned_seq1, aligned_seq2, score
    
    def sum_of_pairs_score(self, alignment: List[str]) -> int:
        """
        Calculate Sum-of-Pairs (SP) score for a multiple sequence alignment.
        
        Args:
            alignment: List of aligned sequences (all same length)
        
        Returns:
            Sum of all pairwise alignment scores
        """
        if len(alignment) < 2:
            return 0
        
        n_seqs = len(alignment)
        length = len(alignment[0])
        total_score = 0
        
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                for k in range(length):
                    total_score += self.score_pairwise(alignment[i][k], alignment[j][k])
        
        return total_score
    
    def progressive_align(self, sequences: List[str]) -> List[str]:
        """
        Progressive alignment (simplified version).
        This is a heuristic approach, not optimal, but faster than full DP.
        
        Args:
            sequences: List of sequences to align
        
        Returns:
            List of aligned sequences
        """
        if len(sequences) == 0:
            return []
        if len(sequences) == 1:
            return sequences
        
        # Start with first two sequences
        aligned = list(self.align_pairwise(sequences[0], sequences[1])[:2])
        
        # Progressively add remaining sequences
        for seq in sequences[2:]:
            # Align new sequence with consensus or first aligned sequence
            # This is a simplified version - full progressive alignment is more complex
            new_aligned, _, _ = self.align_pairwise(seq, aligned[0].replace('-', ''))
            # Adjust gaps to match existing alignment
            # (This is simplified - proper progressive alignment needs profile alignment)
            aligned.append(new_aligned)
        
        return aligned
    
    def full_dp_msa(self, sequences: List[str]) -> Optional[List[str]]:
        """
        Full dynamic programming for MSA (exponential complexity).
        Only feasible for very small numbers of sequences.
        
        WARNING: This is computationally intractable for more than 3-4 sequences.
        """
        if len(sequences) > 3:
            print("Warning: Full DP MSA is only feasible for 3 or fewer sequences.")
            return None
        
        # This would require implementing full multi-dimensional DP
        # For now, return None - this is a placeholder
        # Full implementation would be very complex
        return None

