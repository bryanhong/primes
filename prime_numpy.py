#!/usr/bin/env python3
import math
import numpy as np
from collections import deque

def simple_sieve(limit: int) -> np.ndarray:
    if limit < 2: return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    r = int(limit**0.5)
    for p in range(2, r + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False
    return np.flatnonzero(is_prime).astype(np.int64)

def primes_upto(limit: int, segment_odd_count: int = 20_000_000):
    """Stream all primes <= limit, returning (first100, last100, count)."""
    first100 = []
    last100 = deque(maxlen=100)
    total = 0

    # base primes up to sqrt(limit)
    base_limit = int(math.isqrt(limit)) + 1
    base = simple_sieve(base_limit)

    # handle 2
    if limit >= 2:
        first100.append(2)
        last100.append(2)
        total += 1

    # odd-only segmented sieve: segment covers 2*segment_odd_count integers
    span = 2 * segment_odd_count
    low = 3 if limit >= 3 else limit + 1
    if low % 2 == 0: low += 1

    while low <= limit:
        high = min(low + span, limit + 1)   # exclusive
        odd_count = (high - low) // 2
        mask = np.ones(odd_count, dtype=bool)

        for p in base:
            if p == 2: 
                continue
            p2 = p * p
            if p2 > limit:
                break
            start = max(p2, ((low + p - 1) // p) * p)
            if (start & 1) == 0:
                start += p
            if start >= high:
                continue
            offset = (start - low) // 2
            mask[offset::p] = False  # vectorized strided clear

        # collect primes in this segment (streaming)
        if mask.any():
            idx = np.flatnonzero(mask)
            seg_primes = (low + 2*idx).tolist()
            for pr in seg_primes:
                if total < 100: first100.append(pr)
                last100.append(pr)
                total += 1

        low = high

    return first100, list(last100), total

if __name__ == "__main__":
    LIMIT = 10_000_000_000
    f100, l100, cnt = primes_upto(LIMIT, segment_odd_count=20_000_000)
    print(f"Mode: primes <= {LIMIT:,} | Count: {cnt:,}")
    print("First 100:", f100)
    print("Last 100:", l100)

