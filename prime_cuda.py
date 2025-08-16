#!/usr/bin/env python3
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel that marks composites for a given prime p
kernel_code = r'''
__global__ void mark_composites(bool *is_prime, int p, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = p * p + idx * p;
    if (start <= limit) {
        is_prime[start] = false;
    }
}
'''
mod = SourceModule(kernel_code)
mark_composites = mod.get_function("mark_composites")

def gpu_sieve(limit):
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[:2] = False  # 0,1 are not prime

    d_is_prime = cuda.mem_alloc(is_prime.nbytes)
    cuda.memcpy_htod(d_is_prime, is_prime)

    threads_per_block = 256

    p = 2
    while p * p <= limit:
        # (Optional) you could check host-side is_prime[p] to skip composites,
        # but since the host copy isn't updated until the end, we'll just launch anyway.
        start = p * p
        if start <= limit:
            # how many multiples to mark for this p?
            n = (limit - start) // p + 1
            blocks = (n + threads_per_block - 1) // threads_per_block

            # --- KEY FIX: launch config via keyword args, after kernel args ---
            mark_composites(
                d_is_prime, np.int32(p), np.int32(limit),
                block=(threads_per_block, 1, 1),
                grid=(blocks, 1, 1)
            )
        p += 1

    cuda.memcpy_dtoh(is_prime, d_is_prime)
    primes_gpu = np.where(is_prime[2:])[0]
    return primes_gpu


def main():
    limit = 1000000000
    print(f"Calculating all prime numbers up to {limit} on GPU...")
    primes = gpu_sieve(limit)
    print(f"Found {len(primes)} prime numbers.")
    print(f"The first 10 primes are: {primes[:100]}")
    print(f"The last 10 primes are: {primes[-100:]}")

if __name__ == "__main__":
    main()

