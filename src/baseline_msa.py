import numpy as np
from typing import List, Tuple, Optional


class BaselineMSA:
    """Standard dynamic programming approach for Multiple Sequence Alignment.
    """
    
    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -1):
        """Initialize MSA with scoring parameters.
        
        Args:
            match_score: Score for matching characters.
            mismatch_score: Score for mismatching characters.
            gap_penalty: Penalty for gaps (should be negative).
        """
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty
    
    def score_pairwise(self, char1: str, char2: str) -> int:
        """Calculate score between two characters.

        Args:
            char1: First character.
            char2: Second character.

        Returns:
            The score for the pair of characters.
        """
        if char1 == '-' and char2 == '-':
            return 0
        elif char1 == char2:
            return self.match_score
        elif char1 == '-' or char2 == '-':
            return self.gap_penalty
        else:
            return self.mismatch_score
    
    def align_pairwise(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Pairwise alignment using Needleman-Wunsch algorithm.
        
        Args:
            seq1: First sequence.
            seq2: Second sequence.

        Returns:
            Tuple of (aligned_seq1, aligned_seq2, alignment_score).
        """
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.gap_penalty
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.gap_penalty
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = dp[i-1][j-1] + self.score_pairwise(seq1[i-1], seq2[j-1])
                delete = dp[i-1][j] + self.gap_penalty
                insert = dp[i][j-1] + self.gap_penalty
                dp[i][j] = max(match, delete, insert)
        
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
    
    def progressive_align(self, sequences: List[str]) -> List[str]:
        """Progressive alignment using Star Alignment strategy.
        Args:
            sequences: List of sequences to align.
        
        Returns:
            List of algned sequences.
        """
        if len(sequences) == 0:
            return []
        if len(sequences) == 1:
            return sequences
        
        center_seq = sequences[0]
        n_seqs = len(sequences)
        
        pairwise_alignments = []
        
        # 1. Align all sequences to the center sequence
        for i in range(1, n_seqs):
            aligned_other, aligned_center, _ = self.align_pairwise(sequences[i], center_seq)
            pairwise_alignments.append((aligned_other, aligned_center))
            
        # 2. Determine maximum gaps needed before each position of center_seq
        max_gaps = [0] * (len(center_seq) + 1)
        
        for _, aligned_center in pairwise_alignments:
            current_pos = 0
            current_gaps = 0
            for char in aligned_center:
                if char == '-':
                    current_gaps += 1
                else:
                    max_gaps[current_pos] = max(max_gaps[current_pos], current_gaps)
                    current_pos += 1
                    current_gaps = 0
            max_gaps[current_pos] = max(max_gaps[current_pos], current_gaps)
            
        # 3. Construct the final MSA
        final_alignment = []
        
        master_center = []
        for i, char in enumerate(center_seq):
            master_center.append('-' * max_gaps[i])
            master_center.append(char)
        master_center.append('-' * max_gaps[len(center_seq)])
        final_alignment.append("".join(master_center))
        
        for aligned_other, aligned_center in pairwise_alignments:
            final_seq = []
            idx = 0
            
            for i in range(len(center_seq) + 1):
                target_gaps = max_gaps[i]
                actual_gaps = 0
                gap_chars = []
                
                while idx < len(aligned_center):
                    c_center = aligned_center[idx]
                    c_other = aligned_other[idx]
                    
                    if c_center == '-':
                        actual_gaps += 1
                        gap_chars.append(c_other)
                        idx += 1
                    else:
                        break
                
                final_seq.extend(gap_chars)
                final_seq.append('-' * (target_gaps - actual_gaps))
                
                if i < len(center_seq):
                    final_seq.append(aligned_other[idx])
                    idx += 1
            
            final_alignment.append("".join(final_seq))
            
        return final_alignment


