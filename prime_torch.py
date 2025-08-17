#!/usr/bin/env python3
"""
Segmented Sieve on Apple Silicon with PyTorch MPS

Two modes:
  1) --count N   : generate the first N primes (last ones will be near p_N)
  2) --limit M   : generate all primes <= M      (last ones will be near M)

Examples:
  # What you expected: primes up to 1e9, then print first/last 100
  python primes_mps_segmented.py --limit 1000000000 --segment-odds 20000000

  # First N primes (impractical for N=1e9 on one machine)
  python primes_mps_segmented.py --count 1000000
"""

import argparse
import math
from collections import deque
import torch

def nth_prime_upper_bound(n: int) -> int:
    if n < 6:
        return 15
    nf = float(n)
    # Dusart-style safe bound
    return int(math.ceil(nf * (math.log(nf) + math.log(math.log(nf)))))

def simple_sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    size = limit + 1
    is_prime = bytearray(b"\x01") * size
    is_prime[0:2] = b"\x00\x00"
    r = int(limit ** 0.5)
    for p in range(2, r + 1):
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start:size:step] = b"\x00" * (((size - 1 - start) // step) + 1)
    return [i for i, v in enumerate(is_prime) if v]

def segmented_sieve(
    *,
    n_target: int | None = None,
    limit: int | None = None,
    segment_odd_count: int = 20_000_000,
    prefer_mps: bool = True,
):
    if (n_target is None) == (limit is None):
        raise ValueError("Specify exactly one of --count or --limit.")

    device = torch.device("mps") if (prefer_mps and torch.backends.mps.is_available()) else torch.device("cpu")

    # Determine the global upper numeric bound we’ll scan to
    if limit is not None:
        upper = int(limit)
    else:
        upper = nth_prime_upper_bound(int(n_target))

    # Base primes up to sqrt(upper) on CPU
    base_limit = int(math.isqrt(upper)) + 1
    base_primes = simple_sieve(base_limit)

    # Output buffers
    first100 = []
    last100 = deque(maxlen=100)
    total_found = 0

    # Handle 2 explicitly
    if upper >= 2:
        first100.append(2)
        last100.append(2)
        total_found = 1
        if n_target is not None and total_found >= n_target:
            return first100[:100], list(last100), total_found

    # Stream odd segments from 3..upper (inclusive)
    odd_span = segment_odd_count
    step_span = 2 * odd_span
    low = 3 if 3 <= upper else upper + 1
    if low % 2 == 0: low += 1

    while low <= upper:
        high = min(low + step_span, upper + 1)  # exclusive
        odd_count = (high - low) // 2
        if odd_count <= 0:
            break

        mask = torch.ones(odd_count, dtype=torch.bool, device=device)

        # Mark composites for odd base primes
        for p in base_primes:
            if p == 2:
                continue
            p2 = p * p
            if p2 > upper:
                break
            start = max(p2, ((low + p - 1) // p) * p)
            if (start & 1) == 0:
                start += p
            if start >= high:
                continue
            offset = (start - low) // 2
            mask[offset::p] = False

        # Extract primes in this segment
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel():
            # Convert to numbers: n = low + 2*idx
            primes_segment = (low + 2 * idx).to("cpu").tolist()
            for pr in primes_segment:
                if total_found < 100:
                    first100.append(pr)
                last100.append(pr)
                total_found += 1
                if n_target is not None and total_found >= n_target:
                    # Early stop only in --count mode
                    return first100[:100], list(last100), total_found

        low = high

    return first100[: min(100, total_found)], list(last100), total_found

def main():
    ap = argparse.ArgumentParser(description="Segmented sieve with PyTorch MPS.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", "-n", type=int, help="Generate the first N primes.")
    g.add_argument("--limit", type=int, help="Generate all primes <= LIMIT.")
    ap.add_argument("--segment-odds", type=int, default=20_000_000,
                    help="Odd numbers per segment (default: 20,000,000).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = ap.parse_args()

    first100, last100, total = segmented_sieve(
        n_target=args.count,
        limit=args.limit,
        segment_odd_count=args.segment_odds,
        prefer_mps=(not args.cpu),
    )

    print(f"Device: {'MPS' if (not args.cpu and torch.backends.mps.is_available()) else 'CPU'}")
    if args.count is not None:
        print(f"Mode: first N primes | N requested: {args.count:,} | Found: {total:,}")
    else:
        print(f"Mode: primes <= LIMIT | LIMIT: {args.limit:,} | Count: {total:,}")

    print("\nFirst 100 primes:")
    print(", ".join(map(str, first100)))
    print("\nLast 100 primes seen:")
    print(", ".join(map(str, last100)))

if __name__ == "__main__":
    main()

