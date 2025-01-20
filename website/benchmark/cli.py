import argparse
import sys
from pathlib import Path
from benchmark_runner import run_model_benchmark
from visualize import generate_benchmark_report

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks for Hugging Face models')
    
    parser.add_argument('--model', '-m', 
                       type=str, 
                       required=True,
                       help='Hugging Face model ID (e.g., microsoft/resnet-50)')
    
    parser.add_argument('--output', '-o',
                       type=str,
                       default='benchmark_results',
                       help='Output directory for benchmark results')
    
    parser.add_argument('--report', '-r',
                       type=str,
                       help='Generate HTML report at specified path')
    
    parser.add_argument('--batch-sizes',
                       type=int,
                       nargs='+',
                       default=[1, 2, 4, 8],
                       help='Batch sizes to test')
    
    parser.add_argument('--iterations',
                       type=int,
                       default=100,
                       help='Number of iterations for each batch size')
    
    parser.add_argument('--warmup',
                       type=int,
                       default=10,
                       help='Number of warmup iterations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run benchmark
        print(f"\nRunning benchmark for {args.model}")
        print(f"Batch sizes: {args.batch_sizes}")
        print(f"Iterations: {args.iterations}")
        print(f"Warmup iterations: {args.warmup}\n")
        
        benchmark_results = run_model_benchmark(args.model)
        print("\nBenchmark Results:")
        print(benchmark_results)
        
        # Generate report if requested
        if args.report:
            print(f"\nGenerating HTML report at {args.report}")
            generate_benchmark_report(args.model, args.report)
            print("Report generated successfully")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()