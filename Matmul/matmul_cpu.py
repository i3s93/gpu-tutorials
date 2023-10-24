"""
This script tests a CPU matrix-matrix multiplication algorithm using NumPy
"""

import argparse
import time
import numpy as np

# Parse the command line data
parser = argparse.ArgumentParser(description="Performs matrix-matrix multiplication on the CPU.")

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
	np.random.seed(0)
	A = np.random.rand(N,N) 
	B = np.random.rand(N,N)

	# Array for the timing statistics
	run_times = np.zeros([reps])

	for n in range(reps):

		start = time.perf_counter()		

		C = np.matmul(A,B)

		end = time.perf_counter()
		
		run_times[n] = end - start

	# Run statistics
	print("Average time (s):", "{0:.6e}".format(np.mean(run_times)))
	print("Minimum time (s):", "{0:.6e}".format(np.min(run_times)))
	print("Maximum time (s):", "{0:.6e}".format(np.max(run_times)))
	print("Standard deviation (s):", "{0:.6e}".format(np.std(run_times)), "\n")

	return None

if __name__ == "__main__":

	main()













