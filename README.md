## Purpose of the repository
This repo will holds code used in our tutorials for GPU programming. In particular, we shall focus on computational kernels that are relevant to the algorithms used in tensor decompositions.

## Summary of current benchmarks

We compare the performance of several kernels written using tools from SciPy and NumPy for CPUs and CuPy for a single GPU. The kernels are as follows:
* FFT: We provide code for a single CPU thread using SciPy's FFT. For the GPU, we use cuFFT to perform the same operation and use a plan to test different algorithms.
* QR: Here we compare the performance of a QR factorization written in NumPy which calls BLAS and Lapack methods behind the scenes. These implementations are threaded for CPUs. For the GPU side, we call CuPy's implementation of QR that uses CUDA.
* SVD: We compare the performance of a QR factorization written in NumPy which calls BLAS and Lapack methods behind the scenes. These implementations are threaded for CPUs. For the GPU side, we call CuPy's implementation of QR that uses CUDA.
* Matmul: This test compares matrix-matrix multiplication using threaded NumPy calls to BLAS and Lapack with those of CuPy, which make calls to CUDA code.

__Note__: The CuPy API we leverage does not specify more advanced configurations for threads such as the number of threads per threadblock and streams. This is merely to demonstrate that it is possible to quickly take some existing CPU code and convert it to something that runs on a GPU. Other optimizations can be made, but this will require more delicate tuning and usage of more advanced capabilities that are closer to the CUDA API.

## Running the codes

After setting up your environment, you can begin running the scripts. Each script can be launched with a command of the following form:
```
python <filename.py> -N <integer> -r <integer>
```
where `filename.py` is the name of the python script (which can be CPU or GPU), `N` is the number of entries per mode/dimension (controls the problem size), and `r` is the number of repetitions to be performed while collecting timing data.

## Dependencies

Here is a summary of the modules that are required to run the code and the version which I am using on the MSU HPCC:
* GCC (GCCcore/12.2.0)
* CUDA (CUDA/12.0.0)
* Python (Python/3.10.8)
* NumPy (numpy 1.26.1)
* SciPy (scipy 1.11.3)
* CuPy (cupy 12.2.0)
  
__Note__: Other versions/configurations may also be supported, but this one works for me. Note that if you run this on the MSU HPCC, the module system will swap the version of GCC with an older version and will mark Python as an inactive module if you put this in a `~/.bashrc` file. One solution is to manually load the modules using the following sequence of commands
1. `module load GCCcore/12.2.0`
2. `module load CUDA/12.0.0`
3. `module load Python/3.10.8` 
The remaining packages can be installed using a package manager such as pip.

## Other helpful commands

Currently, MSU's HPCC has a partition of Intel Xeon Phi CPUs with NVIDIA GPUs (Tesla V100 microarchitecture). It has 48 CPUs per node and 4 GPUs per node, which can be verified, respectively by the commands`lscpu` and `nvidia-smi`. You can request an interactive job session using the command below:
```
salloc -N 1 -c 48 --gres=gpu:1 --time=1:00:00 --constraint=amd20
```
This command requests 1 node (all 48 CPU cores) and a single GPU for 1 hour. The last option informs the scheduler of the partition that you would like to use. Here are some additional links that you might find helpful:
* [CuPy documentation page](https://docs.cupy.dev/en/stable/)
* [ICER: Development Nodes](https://docs.icer.msu.edu/development_nodes/)
* [ICER: Setting up and interactive job](https://docs.icer.msu.edu/Interactive_Job/)
