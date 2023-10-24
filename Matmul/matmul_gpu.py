"""
This script tests matrix-matrix multiplication on a GPU using CuPy.
"""

import argparse
import time
import numpy as np
import cupy as cp
from cupyx.profiler import benchmark

# Parse the command line data
parser = argparse.ArgumentParser(description="Performs matrix-matrix multiplication on the GPU.")

parser.add_argument("-N", type=int, default=64, help="Matrix elements N x N.")
parser.add_argument("-r", type=int, default=10, help="Number of repetitions to use.")
args = parser.parse_args()

def main():
	"""
	Driver for matrix-matrix multiplication.
	"""
	
	# Display the arguments in the console
	print("\nOptions used:")
	
	for arg in vars(args):
		print(arg, "=", getattr(args, arg))
	print("\n")

	# Extract command line args 
	N = args.N
	reps = args.r

	# Create the input matrix random
	# We use a seed to ensure consistency in the structure
	cp.random.seed(0)
	A = cp.random.rand(N,N) 
	B = cp.random.rand(N,N)

	# Run statistics using the benchmark algorithm
	print(benchmark(cp.matmul, (A,B,),  n_repeat=reps))

	return None

if __name__ == "__main__":

	main()


