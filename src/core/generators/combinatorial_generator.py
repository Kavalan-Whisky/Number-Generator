"""
Combinatorial Generators: Permutations, Combinations, Partitions, etc.
Implements Heap's algorithm, Steinhaus-Johnson-Trotter, and more.
"""

import math
from typing import Any, Iterator, List, Optional, Tuple


class PermutationGenerator:
    """Generate permutations using various algorithms."""

    @staticmethod
    def heap_algorithm(lst: List) -> Iterator[List]:
        """
        Heap's algorithm for generating all permutations.
        More efficient than lexicographic order for enumeration.
        """
        a = list(lst)
        n = len(a)
        c = [0] * n
        yield list(a)
        i = 0
        while i < n:
            if c[i] < i:
                if i % 2 == 0:
                    a[0], a[i] = a[i], a[0]
                else:
                    a[c[i]], a[i] = a[i], a[c[i]]
                yield list(a)
                c[i] += 1
                i = 0
            else:
                c[i] = 0
                i += 1

    @staticmethod
    def lexicographic(lst: List) -> Iterator[List]:
        """Generate permutations in lexicographic order."""
        a = sorted(lst)
        yield list(a)
        while True:
            # Find largest i where a[i] < a[i+1]
            i = len(a) - 2
            while i >= 0 and a[i] >= a[i + 1]:
                i -= 1
            if i < 0:
                return
            # Find largest j where a[i] < a[j]
            j = len(a) - 1
            while a[j] <= a[i]:
                j -= 1
            a[i], a[j] = a[j], a[i]
            # Reverse suffix
            a[i + 1 :] = reversed(a[i + 1 :])
            yield list(a)

    @staticmethod
    def steinhaus_johnson_trotter(n: int) -> Iterator[List[int]]:
        """
        Steinhaus-Johnson-Trotter algorithm.
        Each successive permutation differs by a single swap.
        """
        perm = list(range(1, n + 1))
        direction = [-1] * (n + 1)  # -1 = left, 1 = right
        direction[0] = 0

        yield list(perm)

        while True:
            # Find largest mobile element
            mobile = -1
            mobile_idx = -1
            for i in range(n):
                e = perm[i]
                d = direction[e]
                ni = i + d
                if d != 0 and 0 <= ni < n and perm[ni] < e:
                    if e > mobile:
                        mobile = e
                        mobile_idx = i

            if mobile == -1:
                return

            # Swap mobile with the element it's looking at
            new_idx = mobile_idx + direction[mobile]
            perm[mobile_idx], perm[new_idx] = perm[new_idx], perm[mobile_idx]
            yield list(perm)

            # Reverse direction of elements larger than mobile
            for i in range(n):
                if perm[i] > mobile:
                    direction[perm[i]] = -direction[perm[i]]

    @staticmethod
    def count(n: int) -> int:
        """Number of permutations of n elements."""
        return math.factorial(n)

    @staticmethod
    def kth_permutation(lst: List, k: int) -> List:
        """Return the kth permutation (0-indexed, lexicographic order)."""
        lst = sorted(lst)
        n = len(lst)
        result = []
        k = k % math.factorial(n)
        for i in range(n, 0, -1):
            f = math.factorial(i - 1)
            idx = k // f
            result.append(lst[idx])
            lst.pop(idx)
            k %= f
        return result

    @staticmethod
    def generate_all(lst: List, algorithm: str = "heap") -> List[List]:
        """Generate all permutations using specified algorithm."""
        if algorithm == "heap":
            return list(PermutationGenerator.heap_algorithm(lst))
        elif algorithm == "lexicographic":
            return list(PermutationGenerator.lexicographic(lst))
        elif algorithm == "sjt":
            n = len(lst)
            perms = list(PermutationGenerator.steinhaus_johnson_trotter(n))
            return [[lst[i - 1] for i in p] for p in perms]
        raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def partial_permutations(lst: List, r: int) -> List[List]:
        """Generate all r-permutations (P(n, r))."""
        result = []
        def backtrack(current, remaining):
            if len(current) == r:
                result.append(list(current))
                return
            for i, item in enumerate(remaining):
                backtrack(current + [item], remaining[:i] + remaining[i+1:])
        backtrack([], list(lst))
        return result


class CombinationGenerator:
    """Generate combinations."""

    @staticmethod
    def combinations(lst: List, r: int) -> Iterator[Tuple]:
        """Generate all r-combinations from lst."""
        n = len(lst)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(lst[i] for i in indices)
        while True:
            for i in range(r - 1, -1, -1):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            yield tuple(lst[i] for i in indices)

    @staticmethod
    def combinations_with_replacement(lst: List, r: int) -> Iterator[Tuple]:
        """Generate combinations with replacement."""
        n = len(lst)
        if not n and r:
            return
        indices = [0] * r
        yield tuple(lst[i] for i in indices)
        while True:
            for i in range(r - 1, -1, -1):
                if indices[i] != n - 1:
                    break
            else:
                return
            new_val = indices[i] + 1
            for j in range(i, r):
                indices[j] = new_val
            yield tuple(lst[i] for i in indices)

    @staticmethod
    def count(n: int, r: int) -> int:
        """C(n, r) = n! / (r! * (n-r)!)"""
        return math.comb(n, r)

    @staticmethod
    def kth_combination(lst: List, r: int, k: int) -> Tuple:
        """Return the kth combination (0-indexed, lexicographic)."""
        n = len(lst)
        result = []
        start = 0
        for i in range(r):
            for j in range(start, n):
                c = math.comb(n - j - 1, r - i - 1)
                if k < c:
                    result.append(lst[j])
                    start = j + 1
                    break
                k -= c
        return tuple(result)

    @staticmethod
    def generate_all(lst: List, r: int) -> List[Tuple]:
        return list(CombinationGenerator.combinations(lst, r))

    @staticmethod
    def powerset(lst: List) -> List[Tuple]:
        """Generate all subsets (power set)."""
        result = []
        for r in range(len(lst) + 1):
            result.extend(CombinationGenerator.combinations(lst, r))
        return result


class PartitionGenerator:
    """Integer partition generators."""

    @staticmethod
    def partitions(n: int) -> Iterator[List[int]]:
        """Generate all integer partitions of n (in descending order)."""
        if n == 0:
            yield []
            return

        def _part(n, max_val):
            if n == 0:
                yield []
                return
            for i in range(min(n, max_val), 0, -1):
                for rest in _part(n - i, i):
                    yield [i] + rest

        yield from _part(n, n)

    @staticmethod
    def count_partitions(n: int) -> int:
        """Count the number of partitions of n using dynamic programming."""
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                dp[j] += dp[j - i]
        return dp[n]

    @staticmethod
    def restricted_partitions(n: int, k: int) -> Iterator[List[int]]:
        """Generate partitions of n into exactly k parts."""
        def _part(n, k, max_val):
            if k == 0:
                if n == 0:
                    yield []
                return
            for i in range(min(n, max_val), 0, -1):
                for rest in _part(n - i, k - 1, i):
                    yield [i] + rest

        yield from _part(n, k, n)

    @staticmethod
    def compositions(n: int) -> Iterator[List[int]]:
        """Generate all ordered compositions of n."""
        if n == 0:
            yield []
            return
        for i in range(1, n + 1):
            for rest in PartitionGenerator.compositions(n - i):
                yield [i] + rest

    @staticmethod
    def count_compositions(n: int) -> int:
        """Number of compositions: 2^(n-1)."""
        return 2 ** (n - 1) if n > 0 else 1

    @staticmethod
    def generate_all(n: int) -> List[List[int]]:
        return list(PartitionGenerator.partitions(n))


class SubsetGenerator:
    """Generate subsets and power sets."""

    @staticmethod
    def all_subsets(lst: List) -> Iterator[List]:
        """Generate all subsets in binary order."""
        n = len(lst)
        for mask in range(1 << n):
            yield [lst[i] for i in range(n) if mask & (1 << i)]

    @staticmethod
    def subsets_of_size(lst: List, k: int) -> List[List]:
        """All subsets of exactly k elements."""
        return [list(c) for c in CombinationGenerator.combinations(lst, k)]

    @staticmethod
    def random_subset(lst: List, k: Optional[int] = None) -> List:
        """Random subset of size k (or random size if k is None)."""
        import secrets
        if k is None:
            k = secrets.randbelow(len(lst) + 1)
        if k > len(lst):
            raise ValueError("k cannot exceed list length")
        # Fisher-Yates shuffle subset
        lst_copy = list(lst)
        for i in range(k):
            j = secrets.randbelow(len(lst_copy) - i) + i
            lst_copy[i], lst_copy[j] = lst_copy[j], lst_copy[i]
        return lst_copy[:k]


class CartesianProductGenerator:
    """Generate Cartesian products of iterables."""

    @staticmethod
    def product(*iterables: List) -> Iterator[Tuple]:
        """Generate Cartesian product of input iterables."""
        pools = [list(p) for p in iterables]
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        for item in result:
            yield tuple(item)

    @staticmethod
    def product_with_repeat(lst: List, r: int) -> Iterator[Tuple]:
        """Cartesian product of lst with itself r times."""
        yield from CartesianProductGenerator.product(*([lst] * r))

    @staticmethod
    def count(*iterables: List) -> int:
        result = 1
        for it in iterables:
            result *= len(it)
        return result


class DerangementGenerator:
    """
    Generate derangements (permutations with no fixed points).
    D_n = n! * sum((-1)^k / k! for k in 0..n)
    """

    @staticmethod
    def count(n: int) -> int:
        """Number of derangements of n elements."""
        if n == 0:
            return 1
        if n == 1:
            return 0
        d_prev2, d_prev1 = 1, 0
        for i in range(2, n + 1):
            d = (i - 1) * (d_prev1 + d_prev2)
            d_prev2, d_prev1 = d_prev1, d
        return d_prev1

    @staticmethod
    def generate_all(lst: List) -> List[List]:
        """Generate all derangements of lst."""
        n = len(lst)
        result = []
        for perm in PermutationGenerator.lexicographic(list(range(n))):
            if all(perm[i] != i for i in range(n)):
                result.append([lst[i] for i in perm])
        return result

    @staticmethod
    def random_derangement(lst: List) -> List:
        """Generate a random derangement using Sattolo's algorithm."""
        import secrets
        a = list(lst)
        n = len(a)
        for i in range(n - 1, 0, -1):
            j = secrets.randbelow(i)  # j in [0, i-1]
            a[i], a[j] = a[j], a[i]
        return a

    @staticmethod
    def probability(n: int) -> float:
        """Probability that a random permutation is a derangement ≈ 1/e."""
        return DerangementGenerator.count(n) / math.factorial(n)


class CombinatorialGeneratorFactory:
    """Factory for combinatorial generators."""

    @staticmethod
    def permutations(lst: List, algorithm: str = "heap") -> List[List]:
        return PermutationGenerator.generate_all(lst, algorithm)

    @staticmethod
    def combinations(lst: List, r: int) -> List[Tuple]:
        return CombinationGenerator.generate_all(lst, r)

    @staticmethod
    def partitions(n: int) -> List[List[int]]:
        return PartitionGenerator.generate_all(n)

    @staticmethod
    def derangements(lst: List) -> List[List]:
        return DerangementGenerator.generate_all(lst)

    @staticmethod
    def powerset(lst: List) -> List[Tuple]:
        return CombinationGenerator.powerset(lst)
