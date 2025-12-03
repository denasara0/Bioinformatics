import numpy as np
from typing import List, Tuple, Optional, Set
from heapq import heappush, heappop


class AlignmentState:
    """Represents a state in the A* search space.

    A state is a partial alignment of sequences.

    Attributes:
        aligned_sequences: Current partial alignment.
        positions: Current position in each original sequence.
        g_score: Cost from start to this state.
        h_score: Heuristic estimate of cost to goal.
        f_score: Total estimated cost (g + h).
    """
    def __init__(self, aligned_sequences: List[str], positions: List[int], g_score: int, h_score: int, f_score: int):
        self.aligned_sequences = aligned_sequences
        self.positions = positions
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = f_score
    
    def __lt__(self, other):
        if self.f_score != other.f_score:
            return self.f_score > other.f_score
        return self.g_score > other.g_score


class AStarMSA:
    """A* search for optimal Multiple Sequence Alignment.
    """
    
    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -1):
        """Initialize A* MSA with scoring parameters.
        
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
    
    def heuristic(self, state: AlignmentState, original_sequences: List[str]) -> int:
        """Heuristic function h(n): estimates minimum cost to complete alignment.
        Args:
            state: Current alignment state.
            original_sequences: Original unaligned sequences.
        
        Returns:
            Estimated cost to complete alignment.
        """
        n_seqs = len(original_sequences)
        if n_seqs < 2:
            return 0
        
        total_heuristic = 0
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                pos_i = state.positions[i]
                pos_j = state.positions[j]
                if (i, j) in self.heuristic_tables:
                    cost = self.heuristic_tables[(i, j)][pos_i, pos_j]
                else:
                    cost = self.heuristic_tables[(j, i)][pos_j, pos_i]
                total_heuristic += cost
        
        return total_heuristic
    
    def _precompute_suffix_costs(self, seq1: str, seq2: str) -> np.ndarray:
        """Precompute pairwise alignment costs for all suffixes.
        Args:
            seq1: First sequence.
            seq2: Second sequence.
            
        Returns:
            2D numpy array of scores.
        """
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        # Initialize boundaries (reverse)
        # dp[i, n] corresponds to aligning seq1[i:] with gap
        # The cost is (m-i) * gap_penalty
        for i in range(m + 1):
            dp[i, n] = (m - i) * self.gap_penalty
            
        for j in range(n + 1):
            dp[m, j] = (n - j) * self.gap_penalty
        
        # Fill DP table using Reverse Needleman-Wunsch
        for i in range(m - 1, -1, -1):
            char1 = seq1[i]
            for j in range(n - 1, -1, -1):
                char2 = seq2[j]
                
                if char1 == char2:
                    score = self.match_score
                else:
                    score = self.mismatch_score
                
                match = dp[i+1, j+1] + score
                delete = dp[i+1, j] + self.gap_penalty
                insert = dp[i, j+1] + self.gap_penalty
                
                dp[i, j] = max(match, delete, insert)
                
        return dp


    
    def calculate_column_score(self, column: List[str]) -> int:
        """Calculate score for a single column in the alignment.
        
        Args:
            column: List of characters at a position (one per sequence).
        
        Returns:
            Sum of pairwise scores in this column.
        """
        n = len(column)
        score = 0
        for i in range(n):
            for j in range(i + 1, n):
                score += self.score_pairwise(column[i], column[j])
        return score
    
    def get_successors(self, state: AlignmentState, original_sequences: List[str]) -> List[AlignmentState]:
        """Generate successor states from current state.        
        Args:
            state: Current alignment state.
            original_sequences: Original unaligned sequences.
        
        Returns:
            List of successor states.
        """
        successors = []
        n_seqs = len(original_sequences)
        
        # Generate all possible moves (1 to 2^n - 1)
        moves = []
        def generate_moves(current_move):
            if len(current_move) == n_seqs:
                if any(current_move):
                    moves.append(current_move)
                return
            
            generate_moves(current_move + [0])
            generate_moves(current_move + [1])
            
        generate_moves([])
        
        for move in moves:
            valid_move = True
            new_positions = []
            column = []
            new_aligned = []
            
            for i, advance in enumerate(move):
                current_pos = state.positions[i]
                if advance:
                    if current_pos < len(original_sequences[i]):
                        char = original_sequences[i][current_pos]
                        column.append(char)
                        new_positions.append(current_pos + 1)
                        new_aligned.append(state.aligned_sequences[i] + char)
                    else:
                        valid_move = False
                        break
                else:
                    column.append('-')
                    new_positions.append(current_pos)
                    new_aligned.append(state.aligned_sequences[i] + '-')
            
            if valid_move:
                column_cost = self.calculate_column_score(column)
                
                new_g = state.g_score + column_cost
                new_state = AlignmentState(
                    aligned_sequences=new_aligned,
                    positions=new_positions,
                    g_score=new_g,
                    h_score=0,
                    f_score=0
                )
                successors.append(new_state)
        
        return successors
    
    def is_goal(self, state: AlignmentState, original_sequences: List[str]) -> bool:
        """Check if state represents a complete alignment.

        Args:
            state: Current alignment state.
            original_sequences: Original unaligned sequences.

        Returns:
            True if all sequences are fully aligned, False otherwise.
        """
        return all(pos >= len(original_sequences[i]) 
                 for i, pos in enumerate(state.positions))
    
    def align(self, sequences: List[str], prune_threshold: Optional[int] = None) -> Tuple[List[str], int, int]:
        """Perform A* search to find optimal MSA.
        
        Args:
            sequences: List of sequences to align.
            prune_threshold: If provided, prune states with f_score > threshold.
        
        Returns:
            Tuple of (aligned_sequences, alignment_score, nodes_expanded).
        
        Raises:
            ValueError: If no valid alignment is found.
        """
        if not sequences:
            return [], 0, 0
        
        n_seqs = len(sequences)
        
        # Precompute heuristic tables
        self.heuristic_tables = {}
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                self.heuristic_tables[(i, j)] = self._precompute_suffix_costs(sequences[i], sequences[j])
        
        start_state = AlignmentState(
            aligned_sequences=[''] * n_seqs,
            positions=[0] * n_seqs,
            g_score=0,
            h_score=0,
            f_score=0
        )
        start_state.h_score = self.heuristic(start_state, sequences)
        start_state.f_score = start_state.g_score + start_state.h_score
        
        open_set = [start_state]
        closed_set: Set[Tuple[int, ...]] = set()
        
        best_score = float('-inf')
        best_alignment = None
        nodes_expanded = 0
        
        while open_set:
            current = heappop(open_set)
            nodes_expanded += 1
            
            if self.is_goal(current, sequences):
                if current.g_score > best_score:
                    best_score = current.g_score
                    best_alignment = current.aligned_sequences
                continue
            
            # Pruning: skip if f_score is worse than threshold (lower bound)
            # Since we are maximizing, we prune if f_score < threshold
            if prune_threshold is not None and current.f_score < prune_threshold:
                continue
            
            # Skip if already explored
            state_key = tuple(current.positions)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            successors = self.get_successors(current, sequences)
            
            for successor in successors:
                successor.h_score = self.heuristic(successor, sequences)
                successor.f_score = successor.g_score + successor.h_score
                
                if prune_threshold is not None and successor.f_score < prune_threshold:
                    continue
                
                heappush(open_set, successor)
        
        if best_alignment is None:
            raise ValueError("No valid alignment found")
        
        return best_alignment, int(best_score), nodes_expanded

