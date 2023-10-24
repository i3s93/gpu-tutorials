"""
This script tests the CPU FFT algorithm for solving
Poisson's equation -\Delta u = f in three spatial dimensions.
"""

import argparse
import time
import numpy as np
from scipy.fftpack import fftn, ifftn, fftfreq

# Parse the command line data
parser = argparse.ArgumentParser(description="A 3D Poisson solver based on the FFT (CPU).")

parser.add_argument("-N", type=int, default=64, help="Grid points per dimension.")
parser.add_argument("-r", type=int, default=10, help="Number of repetitions to use.")
args = parser.parse_args()

def exact_solution(x,y,z):
	"""
	Exact solution to a 3D Poisson equation.
	"""
	return np.sin(10*x)*np.cos(5*y)*np.sin(10*z)

def f(x,y,z):
	"""
	The source term in the 3D Poisson equation.
	"""
	return 225*np.sin(10*x)*np.cos(5*y)*np.sin(10*z)

def poisson_3d_fft(f, dx, dy, dz):
	"""
	Solves the 3D Poisson problem using the FFT. 
	
	Returns the real-valued solution u. 
	"""
	
	# Step 1: Take the 3D FFT of the source data f
	f_hat = fftn(f)

	# Step 2: Define the fundamental frequencies
	# Note, this returns the k/L component,
	# so we need the 2*pi and factors of j
	k = 2*np.pi*fftfreq(f_hat.shape[0], dx)
	l = 2*np.pi*fftfreq(f_hat.shape[1], dy)
	m = 2*np.pi*fftfreq(f_hat.shape[2], dz)

	# Create a matrix of the inverse fundamental frequencies (no loops)
	# Define 1/0 to be 0 (for the lowest frequency mode)
	K2 = np.einsum("i,j,k", k**2, np.ones_like(l), np.ones_like(m))
	L2 = np.einsum("i,j,k", np.ones_like(k), l**2, np.ones_like(m))
	M2 = np.einsum("i,j,k", np.ones_like(k), np.ones_like(l), m**2)
	inv_freq_matrix = np.zeros_like(K2)
	inv_freq_matrix[1:,1:,1:] = 1/(K2[1:,1:,1:] + L2[1:,1:,1:] + M2[1:,1:,1:])

	# Step 3: Create the array for the solution and compute the solution
	# at each Fourier mode.
	u_hat = np.zeros_like(f_hat)
	u_hat[:,:,:] = inv_freq_matrix[:,:,:]*f_hat[:,:,:]

	# Step 4: Take the iFFT of u_hat and retain only the real part
	u = np.real( ifftn(u_hat) )

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

	# Setup the mesh
	a_x = 0.0
	b_x = 2*np.pi

	a_y = 0.0
	b_y = 2*np.pi

	a_z = 0.0
	b_z = 2*np.pi

	x = np.linspace(a_x, b_x, N, endpoint=False)
	y = np.linspace(a_y, b_y, N, endpoint=False)
	z = np.linspace(a_z, b_z, N, endpoint=False)
	
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	dz = z[1] - z[0]	

	X_grid, Y_grid, Z_grid = np.meshgrid(x,y,z,indexing="ij")

	# Generate the source data
	source = f(X_grid, Y_grid, Z_grid)

	# Array for the timing statistics
	run_times = np.zeros([reps])

	# Call the Poisson solver several times to generate statistics
	# We only time the Poisson solver
	for n in range(reps):

		start = time.perf_counter()		

		u = poisson_3d_fft(source, dx, dy, dz)

		end = time.perf_counter()
		
		run_times[n] = end - start

	# Measure the error with the analytical solution
	exact = exact_solution(X_grid,Y_grid,Z_grid)
	error1 = dx*dy*dz*np.sum(np.abs(u - exact))
	error2 = np.sqrt( dx*dy*dz*np.sum(np.abs(u - exact)**2) )
	error_max = np.max(np.abs(u - exact))

	print("L1 error:", "{0:.6e}".format(error1))
	print("L2 error:", "{0:.6e}".format(error2))
	print("Linf error:", "{0:.6e}".format(error_max), "\n")

	# Run statistics
	print("Average time (s):", "{0:.6e}".format(np.mean(run_times)))
	print("Minimum time (s):", "{0:.6e}".format(np.min(run_times)))
	print("Maximum time (s):", "{0:.6e}".format(np.max(run_times)))
	print("Standard deviation (s):", "{0:.6e}".format(np.std(run_times)), "\n")

	return None

if __name__ == "__main__":

	main()













