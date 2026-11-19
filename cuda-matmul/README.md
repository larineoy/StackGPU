# CPU and CUDA matrix multiplication

Worked through the matrix multiplication example from Chapter 3, §3.3. Wrote a CPU reference in `matmul.cpp` and a CUDA version in `matmul.cu`, then compiled and ran both on the lab server and compared their output.

## How the programs work

Both programs store each matrix in a flat array in row-major order. The element at row `r`, column `c` is stored at `r * width + c`. The CPU loops over the output rows and columns. On the GPU, each valid thread computes one output cell. Both versions compute that cell by taking the dot product of a row of the first matrix and a column of the second.

## Build and run on the lab server

From the lab server's SSH prompt, start the CUDA container. This command uses the locally available CUDA development image and selects GPU index 4 on the server. The project files are under a bind mount, so they stay on the server when the container exits.

```bash
docker run --rm -it \
  --pull=never \
  --name larine-cuda-run \
  --user "$(id -u):$(id -g)" \
  --gpus '"device=#"' \
  -v /data/larine/cuda-learning:/workspace \
  -w /workspace/cuda-matmul \
  nvidia/cuda:13.2.0-devel-ubuntu24.04 \
  bash
```

Inside the container, build and run both programs. Use the same matrix size, inputs, and print format in both source files.

```bash
g++ -O2 matmul.cpp -o matmul_cpu &&
nvcc -O2 -arch=sm_90 matmul.cu -o matmul_gpu &&
./matmul_cpu > cpu-results.txt &&
./matmul_gpu > results.txt &&
diff -u cpu-results.txt results.txt &&
cat results.txt
```

When the output files match, `diff` is silent and `cat` displays the GPU result. Type `exit` to leave the container and `exit` again at the server prompt to end the SSH session.

## CPU only on macOS

Run from the project directory.

```bash
clang++ -std=c++11 -Wall -Wextra matmul.cpp -o /tmp/matmul_cpu
/tmp/matmul_cpu
```

## Results

**20 × 20 identity test.** Filled `M` with the numbers 1 through 400 in row order and set `N` to the identity matrix. Both programs printed 1 through 400 across 20 rows, as expected from `M × I = M`. The output files matched exactly.

## A few takeaways:

- **Grid and block sizes determine which thread computes each cell.** With 16 × 16 threads per block, the 20 × 20 output needs a 2 × 2 grid of blocks. Each thread calculates its output coordinates from its block and thread indices. The bounds check makes threads outside the output matrix return before accessing memory.
- **Explicit memory management makes the transfers visible.** The host uses `cudaMalloc` for the GPU buffers and `cudaMemcpy` to transfer the inputs and result. The host arrays are ordinary C++ arrays, which work with these `cudaMemcpy` calls.
- **The host and GPU run asynchronously.** The kernel launch lets the host continue. In this program, the device-to-host copy is ordered after the kernel and returns once the result is in the CPU array. The program then releases its GPU buffers with `cudaFree`.
