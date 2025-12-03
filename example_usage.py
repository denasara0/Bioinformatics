import os
import time
import numpy as np
from typing import List, Dict

from src.baseline_msa import BaselineMSA
from src.astar_msa import AStarMSA
from src.evaluation import MSAEvaluator


def get_example_datasets() -> List[Dict]:
    return [
        {
            'name': '1',
            'sequences': [
                'ACGTACGT',
                'ACGTACGT',
                'ACGTACGT'
            ]
        },
        {
            'name': '2',
            'sequences': [
                'ACGTACGT',
                'ACGTACGT',
                'ACGT'
            ]
        },
        {
            'name': '3',
            'sequences': [
                'ATTGCCATT',
                'ATGGCCATT',
                'ATCCAATTT',
                'ATGGCCATT'
            ]
        },
        {
            'name': '4',
            'sequences': [
                'ATTGCCATT',
                'ATGGCCATT',
                'ATCCAATTT',
                'ATGGCCATT',
                'ATTGCCATT'
            ]
        },
        {
            'name': '5',
            'sequences': [
                'TATAGCCAT',
                'TATAGCCAT',
                'GATAGCCAT',
                'TATAGCCAG',
                'TATAAACCT'
            ]
        }
    ]


def run_benchmark():
    evaluator = MSAEvaluator()
    results = []
    
    datasets = get_example_datasets()
    
    print(f"Running {len(datasets)} data sets")
    print(f"\n{'Dataset':<12} {'Seqs':<5} {'Len':<5} {'Baseline':<10} {'A* (Opt)':<10} {'Diff':<6} {'Nodes':<10} {'Space Red.':<12}")
    print("-" * 90)
    
    for ds in datasets:
        dataset_name = ds['name']
        sequences = ds['sequences']
        
        seq_lengths = [len(s) for s in sequences]
        avg_len = np.mean(seq_lengths)
        
        comparison = evaluator.compare_algorithms(sequences, use_pruning=True)
        
        baseline_score = comparison['baseline']['score']
        astar_score = comparison['astar']['score']
        nodes = comparison['astar'].get('nodes_expanded', 0)
        reduction = comparison['astar'].get('space_reduction_pct', 0)
        
        diff = astar_score - baseline_score
        
        print(f"{dataset_name:<12} {len(sequences):<5} {int(avg_len):<5} {baseline_score:<10} {astar_score:<10} {diff:+6d} {nodes:<10} {reduction:.2f}%")
        
        results.append({
            'dataset': dataset_name,
            'n_seqs': len(sequences),
            'avg_len': avg_len,
            'baseline_score': baseline_score,
            'astar_score': astar_score,
            'nodes': nodes,
            'reduction': reduction
        })
        
    return results


def main():    
    os.makedirs('results', exist_ok=True)
    
    results = run_benchmark()


if __name__ == "__main__":
    main()

