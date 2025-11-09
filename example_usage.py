"""
Example usage of MSA algorithms.
Run this to see the algorithms in action.
"""

from src.baseline_msa import BaselineMSA
from src.astar_msa import AStarMSA
from src.evaluation import MSAEvaluator


def main():
    """Run example MSA alignment."""
    
    # Example sequences
    sequences = [
        "ACGTACGT",
        "ACGTACGT",
        "ACGTACGT"
    ]
    
    print("Example Sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i+1}: {seq}")
    
    # Initialize evaluator
    evaluator = MSAEvaluator()
    
    # Compare algorithms
    results = evaluator.compare_algorithms(sequences)
    
    # Print results
    evaluator.print_comparison(results)
    
    # Show alignments
    print("\nBaseline Alignment:")
    for i, seq in enumerate(results['baseline']['alignment']):
        print(f"  Seq {i+1}: {seq}")
    
    print("\nA* Alignment:")
    for i, seq in enumerate(results['astar']['alignment']):
        print(f"  Seq {i+1}: {seq}")


if __name__ == "__main__":
    main()

