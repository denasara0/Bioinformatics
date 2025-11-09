"""
Evaluation framework for MSA algorithms.
Includes metrics for accuracy, runtime, and memory usage.
"""

import time
from typing import List, Tuple, Dict, Optional
import numpy as np
from src.baseline_msa import BaselineMSA
from src.astar_msa import AStarMSA


class MSAEvaluator:
    """Evaluates and compares different MSA algorithms."""
    
    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -1):
        """Initialize evaluator with scoring parameters."""
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty
        self.baseline = BaselineMSA(match_score, mismatch_score, gap_penalty)
        self.astar = AStarMSA(match_score, mismatch_score, gap_penalty)
    
    def sum_of_pairs_score(self, alignment: List[str]) -> int:
        """Calculate Sum-of-Pairs (SP) score for alignment."""
        if len(alignment) < 2:
            return 0
        
        n_seqs = len(alignment)
        length = len(alignment[0])
        total_score = 0
        
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                for k in range(length):
                    char1, char2 = alignment[i][k], alignment[j][k]
                    if char1 == char2:
                        total_score += self.match_score
                    elif char1 == '-' or char2 == '-':
                        total_score += self.gap_penalty
                    else:
                        total_score += self.mismatch_score
        
        return total_score
    
    def column_score(self, alignment: List[str], col_idx: int) -> int:
        """Calculate score for a specific column."""
        column = [seq[col_idx] for seq in alignment]
        n = len(column)
        score = 0
        for i in range(n):
            for j in range(i + 1, n):
                char1, char2 = column[i], column[j]
                if char1 == char2:
                    score += self.match_score
                elif char1 == '-' or char2 == '-':
                    score += self.gap_penalty
                else:
                    score += self.mismatch_score
        return score
    
    def evaluate_baseline(self, sequences: List[str]) -> Dict:
        """Evaluate baseline MSA algorithm."""
        start_time = time.time()
        alignment = self.baseline.progressive_align(sequences)
        runtime = time.time() - start_time
        
        score = self.sum_of_pairs_score(alignment)
        
        return {
            'algorithm': 'baseline_progressive',
            'alignment': alignment,
            'score': score,
            'runtime': runtime,
            'n_seqs': len(sequences),
            'avg_seq_len': np.mean([len(s) for s in sequences])
        }
    
    def evaluate_astar(self, sequences: List[str], prune_threshold: Optional[int] = None) -> Dict:
        """Evaluate A* MSA algorithm."""
        start_time = time.time()
        alignment, score, nodes_expanded = self.astar.align(sequences, prune_threshold)
        runtime = time.time() - start_time
        
        return {
            'algorithm': 'astar',
            'alignment': alignment,
            'score': score,
            'runtime': runtime,
            'nodes_expanded': nodes_expanded,
            'n_seqs': len(sequences),
            'avg_seq_len': np.mean([len(s) for s in sequences]),
            'prune_threshold': prune_threshold
        }
    
    def compare_algorithms(self, sequences: List[str], 
                          use_pruning: bool = False,
                          prune_threshold: Optional[int] = None) -> Dict:
        """
        Compare baseline and A* algorithms.
        
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # Evaluate baseline
        baseline_result = self.evaluate_baseline(sequences)
        results['baseline'] = baseline_result
        
        # Evaluate A*
        astar_result = self.evaluate_astar(sequences, 
                                          prune_threshold if use_pruning else None)
        results['astar'] = astar_result
        
        # Calculate comparison metrics
        results['score_improvement'] = astar_result['score'] - baseline_result['score']
        results['score_improvement_pct'] = (
            (astar_result['score'] - baseline_result['score']) / abs(baseline_result['score']) * 100
            if baseline_result['score'] != 0 else 0
        )
        results['speedup'] = baseline_result['runtime'] / astar_result['runtime'] if astar_result['runtime'] > 0 else float('inf')
        
        return results
    
    def print_comparison(self, comparison_results: Dict):
        """Pretty print comparison results."""
        print("\n" + "="*60)
        print("MSA Algorithm Comparison")
        print("="*60)
        
        baseline = comparison_results['baseline']
        astar = comparison_results['astar']
        
        print(f"\nSequences: {baseline['n_seqs']} sequences, "
              f"avg length: {baseline['avg_seq_len']:.1f}")
        
        print(f"\n{'Algorithm':<20} {'Score':<15} {'Runtime (s)':<15}")
        print("-" * 50)
        print(f"{'Baseline':<20} {baseline['score']:<15} {baseline['runtime']:<15.4f}")
        print(f"{'A* Search':<20} {astar['score']:<15} {astar['runtime']:<15.4f}")
        
        if 'nodes_expanded' in astar:
            print(f"\nA* nodes expanded: {astar['nodes_expanded']}")
        
        if 'score_improvement' in comparison_results:
            improvement = comparison_results['score_improvement']
            improvement_pct = comparison_results['score_improvement_pct']
            print(f"\nScore improvement: {improvement} ({improvement_pct:.2f}%)")
            
            if improvement > 0:
                print("✓ A* found better alignment")
            elif improvement < 0:
                print("⚠ A* found worse alignment (may need tuning)")
            else:
                print("= Same score")
        
        if 'speedup' in comparison_results:
            speedup = comparison_results['speedup']
            if speedup > 1:
                print(f"✓ A* is {speedup:.2f}x faster")
            elif speedup < 1:
                print(f"⚠ A* is {1/speedup:.2f}x slower")
        
        print("="*60 + "\n")

