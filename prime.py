#!/usr/bin/env python3

def sieve_of_eratosthenes(limit):
    """Use the Sieve of Eratosthenes algorithm to find all prime numbers up to 'limit'."""
    # Initialize a boolean array that indicates whether each number is prime
    is_prime = [True] * (limit + 1)
    p = 2

    while (p * p <= limit):
        if is_prime[p]:
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
        p += 1

    # Collect all prime numbers
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    return primes

def main():
    # Set the upper limit to 1,000,000,000
    limit = 1000000000
    print(f"Calculating all prime numbers up to {limit}...")

    # Calculate prime numbers using the Sieve of Eratosthenes
    primes = sieve_of_eratosthenes(limit)

    # Print some statistics about the primes found
    print(f"Found {len(primes)} prime numbers.")
    print(f"The first 100 primes are: {primes[:100]}")
    print(f"The last 100 primes are: {primes[-100:]}")

if __name__ == "__main__":
    main()
