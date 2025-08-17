#!/usr/bin/env python3
"""
Fast segmented sieve on CUDA (PyTorch), minimal PCIe traffic.

Modes:
  --limit M  : all primes <= M
  --count N  : first N primes

Transfers off GPU:
  - per-segment count (scalar)
  - first <=100 primes (only until filled)
  - last <=100 primes per segment (tiny)
"""

import argparse, math
from collections import deque
import torch

def nth_prime_upper_bound(n: int) -> int:
    if n < 6: return 15
    nf = float(n)
    return int(math.ceil(nf * (math.log(nf) + math.log(math.log(nf)))))

def simple_sieve(limit: int) -> list[int]:
    if limit < 2: return []
    size = limit + 1
    is_prime = bytearray(b"\x01") * size
    is_prime[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r + 1):
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start:size:step] = b"\x00" * (((size - 1 - start) // step) + 1)
    return [i for i, v in enumerate(is_prime) if v]

def segmented_sieve_cuda(
    *,
    n_target: int | None = None,
    limit:   int | None = None,
    segment_odd_count: int = 10_000_000,  # ~10MB mask per segment
    prefer_cuda: bool = True,
):
    if (n_target is None) == (limit is None):
        raise ValueError("Specify exactly one of --count or --limit.")
    use_cuda = prefer_cuda and torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Determine numeric bound
    upper = limit if limit is not None else nth_prime_upper_bound(int(n_target))

    # Base primes on CPU
    base_limit = int(math.isqrt(upper)) + 1
    base_primes = simple_sieve(base_limit)

    # Results
    first100: list[int] = []
    last100  = deque(maxlen=100)
    total    = 0

    # Prime 2
    if upper >= 2:
        first100.append(2)
        last100.append(2)
        total = 1
        if n_target is not None and total >= n_target:
            return first100[:100], list(last100), total

    # Segmentation (odd-only)
    low  = 3 if upper >= 3 else upper + 1
    if (low & 1) == 0: low += 1
    span = 2 * segment_odd_count  # numeric width per segment

    while low <= upper:
        high = min(low + span, upper + 1)  # exclusive
        odd_count = (high - low) // 2
        if odd_count <= 0: break

        # 1-byte mask: 1 = candidate prime
        mask = torch.ones(odd_count, dtype=torch.uint8, device=device)

        # Mark composites (strided stores on device)
        for p in base_primes:
            if p == 2: continue
            p2 = p * p
            if p2 > upper: break
            start = max(p2, ((low + p - 1) // p) * p)
            if (start & 1) == 0:
                start += p
            if start >= high: continue
            offset = (start - low) // 2
            mask[offset::p] = 0

        # Count on device; bring back a scalar
        seg_cnt = int(torch.count_nonzero(mask).item())
        if seg_cnt == 0:
            low = high
            continue

        # Handle --count (first N primes) early-exit logic
        remaining = (n_target - total) if n_target is not None else None

        # We only form indices when needed, and we only copy small slices
        need_first = max(0, 100 - len(first100))
        want_last  = min(100, seg_cnt)

        # If we need anything beyond counts, compute idx on device once
        need_indices = (need_first > 0) or (want_last > 0) or (remaining is not None and remaining <= seg_cnt)
        idx = None
        if need_indices:
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)  # GPU tensor of true positions

        # Fill first100 (only once, globally)
        if need_first > 0:
            take = min(need_first, seg_cnt)
            if take > 0:
                first_vals = (low + 2 * idx[:take]).to("cpu").tolist()
                first100.extend(first_vals)

        # Update total & last100 depending on mode
        if n_target is None:
            # LIMIT mode: take the last <=100 primes from this segment (small copy)
            last_vals = (low + 2 * idx[-want_last:]).to("cpu").tolist() if want_last > 0 else []
            for v in last_vals: last100.append(v)
            total += seg_cnt
        else:
            # COUNT mode: do we finish within this segment?
            if seg_cnt < remaining:
                # Not finishing: update total and keep the segment's tail for now
                total += seg_cnt
                last_vals = (low + 2 * idx[-want_last:]).to("cpu").tolist()
                for v in last_vals: last100.append(v)
            else:
                # We finish here: copy exactly the last needed primes
                need = remaining
                finish_vals = (low + 2 * idx[:need]).to("cpu").tolist()
                # Update total as we stream these in (and last100)
                for v in finish_vals:
                    last100.append(v)
                    total += 1
                return first100[:100], list(last100), total

        low = high

    return first100[: min(100, total)], list(last100), total

def main():
    ap = argparse.ArgumentParser(description="Segmented sieve on CUDA (minimal host transfers).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--limit", type=int, help="Generate all primes <= LIMIT.")
    g.add_argument("--count", "-n", type=int, help="Generate the first N primes.")
    ap.add_argument("--segment-odds", type=int, default=10_000_000,
                    help="Odd numbers per segment (default: 10,000,000).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = ap.parse_args()

    first100, last100, total = segmented_sieve_cuda(
        n_target=args.count,
        limit=args.limit,
        segment_odd_count=args.segment_odds,
        prefer_cuda=(not args.cpu),
    )

    dev = "CUDA" if (not args.cpu and torch.cuda.is_available()) else "CPU"
    if args.count is not None:
        print(f"Device: {dev} | Mode: first N primes | N requested: {args.count:,} | Found: {total:,}")
    else:
        print(f"Device: {dev} | Mode: primes <= LIMIT | LIMIT: {args.limit:,} | Count: {total:,}")

    print("\nFirst 100 primes:")
    print(", ".join(map(str, first100)))
    print("\nLast 100 primes:")
    print(", ".join(map(str, last100)))

if __name__ == "__main__":
    main()

