"""
A* Search for Optimal Multiple Sequence Alignment with Heuristic Pruning
Implements bounded-error optimal MSA using A* search with admissible heuristics.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from heapq import heappush, heappop


@dataclass
class AlignmentState:
    """
    Represents a state in the A* search space.
    A state is a partial alignment of sequences.
    """
    aligned_sequences: List[str]  # Current partial alignment
    positions: List[int]  # Current position in each original sequence
    g_score: int  # Cost from start to this state
    h_score: int  # Heuristic estimate of cost to goal
    f_score: int  # Total estimated cost (g + h)
    
    def __lt__(self, other):
        """For priority queue ordering (lower f_score = higher priority)."""
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.g_score < other.g_score


class AStarMSA:
    """
    A* search for optimal Multiple Sequence Alignment.
    Uses heuristic pruning to reduce search space.
    """
    
    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -1):
        """
        Initialize A* MSA with scoring parameters.
        
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
    
    def heuristic(self, state: AlignmentState, original_sequences: List[str]) -> int:
        """
        Heuristic function h(n): estimates minimum cost to complete alignment.
        
        Implementation: Sum of pairwise alignment scores for unaligned suffixes.
        This is admissible (never overestimates) because it assumes optimal
        pairwise alignments of remaining portions.
        
        Args:
            state: Current alignment state
            original_sequences: Original unaligned sequences
        
        Returns:
            Estimated cost to complete alignment
        """
        n_seqs = len(original_sequences)
        if n_seqs < 2:
            return 0
        
        # Get unaligned suffixes for each sequence
        suffixes = []
        for i, pos in enumerate(state.positions):
            if pos < len(original_sequences[i]):
                suffixes.append(original_sequences[i][pos:])
            else:
                suffixes.append("")
        
        # Calculate sum of pairwise alignment costs for suffixes
        total_heuristic = 0
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                # Estimate pairwise alignment cost for suffixes
                # This is a lower bound (admissible heuristic)
                cost = self._estimate_pairwise_cost(suffixes[i], suffixes[j])
                total_heuristic += cost
        
        return total_heuristic
    
    def _estimate_pairwise_cost(self, seq1: str, seq2: str) -> int:
        """
        Estimate minimum pairwise alignment cost for two sequences.
        Uses a simplified calculation (can be improved with actual DP).
        """
        if not seq1 and not seq2:
            return 0
        if not seq1:
            return len(seq2) * self.gap_penalty
        if not seq2:
            return len(seq1) * self.gap_penalty
        
        # Simple estimation: assume best case scenario
        # This is a lower bound
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        # Best case: all matches + gaps for difference
        best_matches = min_len * self.match_score
        gap_cost = (max_len - min_len) * self.gap_penalty
        
        return best_matches + gap_cost
    
    def calculate_column_score(self, column: List[str]) -> int:
        """
        Calculate score for a single column in the alignment.
        
        Args:
            column: List of characters at a position (one per sequence)
        
        Returns:
            Sum of pairwise scores in this column
        """
        n = len(column)
        score = 0
        for i in range(n):
            for j in range(i + 1, n):
                score += self.score_pairwise(column[i], column[j])
        return score
    
    def get_successors(self, state: AlignmentState, original_sequences: List[str]) -> List[AlignmentState]:
        """
        Generate successor states from current state.
        Each successor represents one step forward in the alignment.
        
        Args:
            state: Current alignment state
            original_sequences: Original unaligned sequences
        
        Returns:
            List of successor states
        """
        successors = []
        n_seqs = len(original_sequences)
        
        # Generate all possible ways to advance one position
        # Option 1: All sequences advance (match/match)
        if all(pos < len(original_sequences[i]) for i, pos in enumerate(state.positions)):
            new_aligned = [seq + original_sequences[i][pos] 
                          for i, (seq, pos) in enumerate(zip(state.aligned_sequences, state.positions))]
            new_positions = [pos + 1 for pos in state.positions]
            
            # Calculate cost of this column
            column = [original_sequences[i][pos] for i, pos in enumerate(state.positions)]
            column_cost = self.calculate_column_score(column)
            
            new_g = state.g_score + column_cost
            new_state = AlignmentState(
                aligned_sequences=new_aligned,
                positions=new_positions,
                g_score=new_g,
                h_score=0,  # Will be calculated later
                f_score=0   # Will be calculated later
            )
            successors.append(new_state)
        
        # Option 2: Each sequence can have a gap (insert gap in one sequence)
        for gap_idx in range(n_seqs):
            if state.positions[gap_idx] < len(original_sequences[gap_idx]):
                new_aligned = []
                for i, (seq, pos) in enumerate(zip(state.aligned_sequences, state.positions)):
                    if i == gap_idx:
                        new_aligned.append(seq + '-')
                    elif pos < len(original_sequences[i]):
                        new_aligned.append(seq + original_sequences[i][pos])
                    else:
                        new_aligned.append(seq + '-')
                
                new_positions = [pos + (1 if i != gap_idx and pos < len(original_sequences[i]) else 0)
                                for i, pos in enumerate(state.positions)]
                
                # Calculate cost of this column
                column = ['-' if i == gap_idx else 
                         (original_sequences[i][pos] if pos < len(original_sequences[i]) else '-')
                         for i, pos in enumerate(state.positions)]
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
        """Check if state represents a complete alignment."""
        return all(pos >= len(original_sequences[i]) 
                 for i, pos in enumerate(state.positions))
    
    def align(self, sequences: List[str], prune_threshold: Optional[int] = None) -> Tuple[List[str], int, int]:
        """
        Perform A* search to find optimal MSA.
        
        Args:
            sequences: List of sequences to align
            prune_threshold: If provided, prune states with f_score > threshold
        
        Returns:
            Tuple of (aligned_sequences, alignment_score, nodes_expanded)
        """
        if not sequences:
            return [], 0, 0
        
        n_seqs = len(sequences)
        
        # Initialize start state
        start_state = AlignmentState(
            aligned_sequences=[''] * n_seqs,
            positions=[0] * n_seqs,
            g_score=0,
            h_score=0,
            f_score=0
        )
        start_state.h_score = self.heuristic(start_state, sequences)
        start_state.f_score = start_state.g_score + start_state.h_score
        
        # Priority queue for A* search
        open_set = [start_state]
        closed_set: Set[Tuple[Tuple[int, ...], Tuple[str, ...]]] = set()
        
        best_score = float('-inf')
        best_alignment = None
        nodes_expanded = 0
        
        while open_set:
            current = heappop(open_set)
            nodes_expanded += 1
            
            # Check if goal state
            if self.is_goal(current, sequences):
                if current.g_score > best_score:
                    best_score = current.g_score
                    best_alignment = current.aligned_sequences
                continue
            
            # Pruning: skip if f_score exceeds threshold
            if prune_threshold is not None and current.f_score > prune_threshold:
                continue
            
            # Skip if already explored (simplified - using positions as key)
            state_key = (tuple(current.positions), tuple(current.aligned_sequences))
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            # Generate successors
            successors = self.get_successors(current, sequences)
            
            for successor in successors:
                # Calculate heuristic and f_score
                successor.h_score = self.heuristic(successor, sequences)
                successor.f_score = successor.g_score + successor.h_score
                
                # Pruning: skip if exceeds threshold
                if prune_threshold is not None and successor.f_score > prune_threshold:
                    continue
                
                heappush(open_set, successor)
        
        if best_alignment is None:
            raise ValueError("No valid alignment found")
        
        return best_alignment, best_score, nodes_expanded

