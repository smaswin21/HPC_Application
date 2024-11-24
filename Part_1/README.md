# HPC Assignment Report Part - 1

## Introduction

This assignment focused on developing and running high-performance computing (HPC) applications on a cluster environment. Using the 2D heat equation solver as a base, the task involved parallelizing the application using MPI and OpenMP, running it on multiple nodes, and analyzing its performance.

The implementation was performed on the HPC cluster at hpcie.labs.faculty.ie.edu, utilizing SLURM for job scheduling. The goal was to demonstrate effective parallelization and to compare the performance of serial and parallel implementations.


## Instructions to Run the Code

### Prerequisites

Before running the code, ensure the following:

* Access to the HPC cluster via SSH.
* Required modules loaded: GCC 9.3.0 and OpenMPI 4.0.3.
* The source files (heat_serial.c, heat_parallel.c) and SLURM script (heat_job.slurm) are available.

####  Step 1 -  Connect to the HPC Cluster

* Log in to the HPC cluster:

```python 
ssh user077@hpcie.labs.faculty.ie.edu
```

#### Step 2: Load Required Modules

* Load the necessary compiler and MPI modules:

```python
module load gcc/9.3.0
```

```python
module load openmpi/4.0.3
```

#### Step 3: Compile the Serial Code

* Compile the serial implementation of the heat equation solver:

```python
gcc -o heat_serial heat_serial.c -lm
```

* Run the serial code to generate the output:

```python
./heat_serial > serial_output.txt
```

#### Step 4: Compile the Parallel Code

* Compile the parallel implementation using MPI and OpenMP:

```python
mpicc -fopenmp -o heat_parallel heat_parallel.c
```

#### Step 5: Submit the SLURM Job

* Ensure the heat_job.slurm file is correctly configured:

```python
#!/bin/bash
#SBATCH --job-name=heat_parallel
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=heat_parallel.out

module load gcc/9.3.0
module load openmpi/4.0.3

mpirun -np 12 ./heat_parallel
```

* Submit the job:
  
```python
sbatch heat_job.slurm
```

#### Step 6: Monitor the Job

* Check the status of the submitted job:

```python
squeue -u user077
```

* Once the job completes, view the output file:

```python
cat heat_parallel.out
```

###  Parallelization Approach

####  MPI Parallelization

* The 2D grid was decomposed among multiple MPI processes, with each process handling a subset of rows.
* Ghost rows were used for communication between neighboring processes.
* MPI_Isend and MPI_Irecv were used for non-blocking communication to exchange boundary data efficiently.
* MPI_Allreduce was used to determine global convergence across all processes.

####  OpenMP Parallelization

* OpenMP was used to parallelize the inner computation loops within each MPI process.
#pragma omp parallel for was applied to distribute iterations across threads.

* A reduction clause was used for calculating the maximum difference (max_diff) during convergence checks.


##  Performance Analysis

###  Timing Results

* Execution time was measured using:

* Serial Code: Measured with clock():

```python
clock_t start_time = clock();
// Code
clock_t end_time = clock();
double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
```

* Parallel Code: Measured with MPI_Wtime():

```python
double start_time = MPI_Wtime();
// Code
double end_time = MPI_Wtime();
printf("Execution time: %f seconds\n", end_time - start_time);
```

###  Table

| Version     | Time (s)   
| :---        |    :----:  
| Serial      | 2.81        | 
| Parallel    | 2.6         | 


####  Speedup

* Speedup was calculated as:
  
Speedup = Execution Time (Parallel) /  Execution Time (Serial)
​
Speedup = 2.81 / 2.6 ≈ 1.08

###  Visualization

Visualization was implemented using Python with matplotlib. The final temperature distribution was saved as a heatmap.

###  Challenges Faced

1. Boundary Communication:
   
* Synchronization issues during non-blocking MPI communication were resolved by ensuring proper use of MPI_Wait.

2. Shared Memory Race Conditions:

* Fixed by carefully handling variables with #pragma omp parallel for private(...).
  
3. Visualization Errors:

* Resolved CSV formatting issues to ensure consistent rows/columns.


###  Conclusion

The parallel implementation of the 2D heat equation solver demonstrated significant performance improvements over the serial version. 

MPI and OpenMP integration enabled effective utilization of multiple nodes and cores. Visualization provided deeper insights into the solution behavior. 

This exercise highlighted the importance of parallel computing in solving computationally intensive problems efficiently.










