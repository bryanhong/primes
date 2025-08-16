#!/usr/bin/env python3

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Helper function to compute primes up to sqrt(limit) using a simple sieve

def small_primes(limit):
    """Return list of primes up to 'limit' using a simple sieve."""
    is_prime = [True] * (limit + 1)
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
        p += 1
    return [p for p in range(2, limit + 1) if is_prime[p]]

# Function to sieve a segment [low, high] using the small primes

def sieve_segment(low, high, primes):
    """Return primes in the segment [low, high] using the provided small primes."""
    size = high - low + 1
    is_prime = [True] * size
    for p in primes:
        # Find the first multiple of p within [low, high]
        start = max(p * p, ((low + p - 1) // p) * p)
        for multiple in range(start, high + 1, p):
            is_prime[multiple - low] = False
    # Collect primes in this segment
    return [n for i, n in enumerate(range(low, high + 1)) if is_prime[i] and n >= 2]


def main():
    limit = 1000000000
    print(f"Calculating all prime numbers up to {limit} using all CPU cores...")

    # Compute small primes up to sqrt(limit)
    sqrt_limit = int(math.isqrt(limit))
    small = small_primes(sqrt_limit)

    # Determine number of workers (CPU cores)
    workers = os.cpu_count() or 1
    print(f"Using {workers} worker processes.")

    # Divide the range [2, limit] into segments for each worker
    segment_size = (limit - 1) // workers + 1
    segments = []
    start = 2
    while start <= limit:
        end = min(start + segment_size - 1, limit)
        segments.append((start, end))
        start = end + 1

    primes = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(sieve_segment, low, high, small) for low, high in segments]
        for future in as_completed(futures):
            primes.extend(future.result())

    primes.sort()
    print(f"Found {len(primes)} prime numbers.")
    print(f"The first 10 primes are: {primes[:100]}")
    print(f"The last 10 primes are: {primes[-100:]}")

if __name__ == "__main__":
    main()
