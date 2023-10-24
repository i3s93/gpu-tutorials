"""
This script tests the GPU FFT algorithm for solving
Poisson's equation -\Delta u = f in three spatial dimensions.
"""

import argparse
import time
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.profiler import benchmark

# Parse the command line data
parser = argparse.ArgumentParser(description="A 3D Poisson solver based on the FFT (GPU).")

parser.add_argument("-N", type=int, default=64, help="Grid points per dimension.")
parser.add_argument("-r", type=int, default=10, help="Number of repetitions to use.")
args = parser.parse_args()

def exact_solution(x,y,z):
	"""
	Exact solution to a 3D Poisson equation.
	"""
	return cp.sin(10*x)*cp.cos(5*y)*cp.sin(10*z)

def f(x,y,z):
	"""
	The source term in the 3D Poisson equation.
	"""
	return 225*cp.sin(10*x)*cp.cos(5*y)*cp.sin(10*z)

def poisson_3d_fft(f, dx, dy, dz, plan):
	"""
	Solves the 3D Poisson problem using the FFT. 
	
	Returns the real-valued solution u. 
	"""
	
	# Step 1: Take the 3D FFT of the source data f
	f_hat = cufft.fftn(f, plan=plan)

	# Step 2: Define the fundamental frequencies
	# Note, this returns the k/L component,
	# so we need the 2*pi and factors of j
	k = 2*cp.pi*cufft.fftfreq(f_hat.shape[0], dx)
	l = 2*cp.pi*cufft.fftfreq(f_hat.shape[1], dy)
	m = 2*cp.pi*cufft.fftfreq(f_hat.shape[2], dz)

	# Create a matrix of the inverse fundamental frequencies (no loops)
	# Define 1/0 to be 0 (for the lowest frequency mode)
	K2 = cp.einsum("i,j,k", k**2, cp.ones_like(l), cp.ones_like(m))
	L2 = cp.einsum("i,j,k", cp.ones_like(k), l**2, cp.ones_like(m))
	M2 = cp.einsum("i,j,k", cp.ones_like(k), cp.ones_like(l), m**2)
	inv_freq_matrix = cp.zeros_like(K2)
	inv_freq_matrix[1:,1:,1:] = 1/(K2[1:,1:,1:] + L2[1:,1:,1:] + M2[1:,1:,1:])

	# Step 3: Create the array for the solution and compute the solution
	# at each Fourier mode.
	u_hat = cp.zeros_like(f_hat)
	u_hat[:,:,:] = inv_freq_matrix[:,:,:]*f_hat[:,:,:]

	# Step 4: Take the iFFT of u_hat and retain only the real part
	u = cp.real( cufft.ifftn(u_hat, plan=plan) )

	return u

def main():
	"""
	Driver for the Poisson solver.
	"""
	
	# Display the arguments in the console
	print("\nOptions used:")
	
	for arg in vars(args):
		print(arg, "=", getattr(args, arg))
	print("\n")

	# Extract command line args 
	N = args.N
	reps = args.r

	test = cp.random.random([N,N,N])
	plan = cufft.get_fft_plan(test, value_type="C2C")

	# Setup the mesh
	a_x = 0.0
	b_x = 2*cp.pi
	
	a_y = 0.0
	b_y = 2*cp.pi

	a_z = 0.0
	b_z = 2*cp.pi

	x = cp.linspace(a_x, b_x, N, endpoint=False)
	y = cp.linspace(a_y, b_y, N, endpoint=False)
	z = cp.linspace(a_z, b_z, N, endpoint=False)	

	dx = x[1] - x[0]
	dy = y[1] - y[0]
	dz = z[1] - z[0]	

	X_grid, Y_grid, Z_grid = cp.meshgrid(x,y,z,indexing="ij")

	# Generate the source data
	source = f(X_grid, Y_grid, Z_grid)

	# Get the solution for the error analysis
	u = poisson_3d_fft(source, dx, dy, dz, plan)

	# Measure the error with the analytical solution
	exact = exact_solution(X_grid,Y_grid, Z_grid)
	error1 = dx*dy*dz*cp.sum(cp.abs(u - exact))
	error2 = cp.sqrt( dx*dy*dz*cp.sum(cp.abs(u - exact)**2) )
	error_max = cp.max(cp.abs(u - exact))

	print("L1 error:", "{0:.6e}".format(error1))
	print("L2 error:", "{0:.6e}".format(error2))
	print("Linf error:", "{0:.6e}".format(error_max))

	# Run statistics using the benchmark algorithm
	print(benchmark(poisson_3d_fft, (source, dx, dy, dz, plan), n_repeat=reps))

	return None

if __name__ == "__main__":

	main()













