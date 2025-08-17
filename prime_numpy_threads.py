#!/usr/bin/env python3
import argparse
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def simple_sieve(limit: int) -> np.ndarray:
    """Classic sieve up to 'limit' (inclusive), returns primes as int64 numpy array."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    r = int(limit ** 0.5)
    for p in range(2, r + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = False
    return np.flatnonzero(is_prime).astype(np.int64)


def sieve_segment_worker(idx: int, low: int, high_exclusive: int, base_primes: np.ndarray):
    """
    Worker: sieve odd numbers in [low, high_exclusive) using base_primes.
    Returns a compact summary to minimize IPC:
      {
        "idx": segment index (for reordering),
        "count": number of primes in this segment,
        "first": up to first 100 primes from this segment,
        "last": up to last 100 primes from this segment,
      }
    """
    odd_count = (high_exclusive - low) // 2
    if odd_count <= 0:
        return {"idx": idx, "count": 0, "first": [], "last": []}

    mask = np.ones(odd_count, dtype=bool)

    for p in base_primes:
        if p == 2:
            continue
        p2 = p * p
        # If p^2 > max value we ever mark, we can stop (bounds by this segment's high or global limit is fine)
        if p2 >= high_exclusive:
            # We *could* still need to mark below in some cases if low is below p2,
            # but if p^2 >= high_exclusive, no multiples exist in this segment.
            break

        # first multiple in [low, high)
        start = max(p2, ((low + p - 1) // p) * p)
        # ensure odd start (we only store odds)
        if (start & 1) == 0:
            start += p
        if start >= high_exclusive:
            continue

        offset = (start - low) // 2
        mask[offset::p] = False  # vectorized strided clear

    if not mask.any():
        return {"idx": idx, "count": 0, "first": [], "last": []}

    idxs = np.flatnonzero(mask)
    seg_primes = (low + 2 * idxs)

    count = int(sep := seg_primes.size)
    if count == 0:
        return {"idx": idx, "count": 0, "first": [], "last": []}

    k = 100 if count >= 100 else count
    first = seg_primes[:k].tolist()
    last = seg_primes[-k:].tolist()
    return {"idx": idx, "count": count, "first": first, "last": last}


def chunk_plan(limit: int, segment_odd_count: int):
    """
    Yields (idx, low, high_exclusive) for odd-only segments covering [3..limit].
    Each segment spans 2*segment_odd_count integers, i.e., segment_odd_count odds.
    """
    if limit < 3:
        return
    span = 2 * segment_odd_count  # numeric span per segment
    low = 3 if (3 <= limit) else limit + 1
    if (low & 1) == 0:
        low += 1
    idx = 0
    while low <= limit:
        high_exclusive = min(low + span, limit + 1)
        yield (idx, low, high_exclusive)
        low = high_exclusive
        idx += 1


def aggregate_results(limit: int, base_primes: np.ndarray, segment_odd_count: int, workers: int):
    """
    Drive the multiprocessing across segments and aggregate:
    - total prime count (including 2 if <= limit)
    - first 100 primes
    - last 100 primes
    """
    first100 = []
    last100 = deque(maxlen=100)
    total = 0

    # Account for prime 2 once
    if limit >= 2:
        first100.append(2)
        last100.append(2)
        total += 1

    segments = list(chunk_plan(limit, segment_odd_count))
    if not segments:
        # Nothing more to do
        return first100[: min(100, total)], list(last100), total

    # Launch workers
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (idx, low, high_excl) in segments:
            futures.append(ex.submit(sieve_segment_worker, idx, low, high_excl, base_primes))

        # Collect results; store by idx to restore order
        by_idx = {}
        for fut in as_completed(futures):
            res = fut.result()
            by_idx[res["idx"]] = res

    # Merge in segment order to preserve “first”/“last” globally in numeric order
    for idx, _, _ in segments:
        res = by_idx.get(idx)
        if not res:
            continue
        # Update total first
        total += res["count"]

        # Merge first100: append from this segment until we reach 100
        need = 100 - len(first100)
        if need > 0 and res["first"]:
            take = res["first"][:need]
            first100.extend(take)

        # Merge last100: extend with this segment's last, then clip to 100
        if res["last"]:
            for v in res["last"]:
                last100.append(v)

    return first100[: min(100, total)], list(last100), total


def main():
    ap = argparse.ArgumentParser(description="Multiprocess segmented, odd-only sieve (NumPy vectorized).")
    ap.add_argument("--limit", type=int, required=True, help="Generate all primes <= LIMIT.")
    ap.add_argument("--segment-odds", type=int, default=20_000_000,
                    help="Odd numbers per segment (default: 20,000,000).")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker processes (default: os.cpu_count()).")
    args = ap.parse_args()

    limit = int(args.limit)
    workers = args.workers if args.workers and args.workers > 0 else None  # None -> use cpu_count()

    # Precompute base primes up to sqrt(limit) once, in the parent
    base_limit = int(math.isqrt(limit)) + 1
    base_primes = simple_sieve(base_limit)

    first100, last100, total = aggregate_results(limit, base_primes, args.segment_odds, workers or 0)

    print(f"Mode: primes <= {limit:,} | Workers: {workers or 'auto'} | Segment odds: {args.segment_odds:,}")
    print(f"Total count: {total:,}")
    print("\nFirst 100 primes:")
    print(", ".join(map(str, first100)))
    print("\nLast 100 primes:")
    print(", ".join(map(str, last100)))


if __name__ == "__main__":
    main()

