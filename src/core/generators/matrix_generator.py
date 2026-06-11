"""
Matrix-based number generators.

Determinants, permanents, characteristic polynomials, Pascal matrix sequences,
Hadamard matrix entries, magic squares, Latin squares, and eigenvalue-related
integer sequences all live here.
"""

import math
from typing import List, Tuple, Optional, Iterator


# ---------------------------------------------------------------------------
# Basic matrix helpers
# ---------------------------------------------------------------------------

Matrix = List[List[float]]


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _mat_pow(M: Matrix, p: int) -> Matrix:
    n = len(M)
    result: Matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    while p:
        if p & 1:
            result = _mat_mul(result, M)
        M = _mat_mul(M, M)
        p >>= 1
    return result


def _identity(n: int) -> Matrix:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Pascal matrix
# ---------------------------------------------------------------------------

def pascal_matrix(n: int) -> Matrix:
    """Return the lower-triangular Pascal matrix of order n."""
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        M[i][0] = 1
        for j in range(1, i + 1):
            M[i][j] = M[i - 1][j - 1] + M[i - 1][j]
    return M


def pascal_row(n: int) -> List[int]:
    """Return the n-th row of Pascal's triangle (0-indexed)."""
    row = [0] * (n + 1)
    row[0] = 1
    for i in range(1, n + 1):
        row[i] = row[i - 1] * (n - i + 1) // i
    return row


def pascal_diagonal(diag: int) -> List[int]:
    """Return all values along the given diagonal of Pascal's triangle."""
    return [math.comb(diag + k, k) for k in range(diag + 1)]


# ---------------------------------------------------------------------------
# Magic squares
# ---------------------------------------------------------------------------

def magic_square_odd(n: int) -> Matrix:
    """Construct a magic square of odd order n using the Siamese method."""
    if n % 2 == 0:
        raise ValueError("n must be odd")
    M = [[0] * n for _ in range(n)]
    i, j = 0, n // 2
    for num in range(1, n * n + 1):
        M[i][j] = num
        ni, nj = (i - 1) % n, (j + 1) % n
        if M[ni][nj]:
            ni, nj = (i + 1) % n, j
        i, j = ni, nj
    return M


def magic_square_doubly_even(n: int) -> Matrix:
    """Magic square for n divisible by 4 (doubly-even order)."""
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4")
    M = [[n * i + j + 1 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            # flip entries in diagonal sub-blocks
            bi, bj = i * 4 // n, j * 4 // n
            on_diag = (bi == bj) or (bi + bj == 3)
            if on_diag:
                M[i][j] = n * n + 1 - M[i][j]
    return M


def magic_constant(n: int) -> int:
    """Magic constant for an n×n magic square."""
    return n * (n * n + 1) // 2


def is_magic_square(M: Matrix) -> bool:
    n = len(M)
    mc = magic_constant(n)
    for row in M:
        if sum(row) != mc:
            return False
    for j in range(n):
        if sum(M[i][j] for i in range(n)) != mc:
            return False
    if sum(M[i][i] for i in range(n)) != mc:
        return False
    if sum(M[i][n - 1 - i] for i in range(n)) != mc:
        return False
    return True


# ---------------------------------------------------------------------------
# Latin squares
# ---------------------------------------------------------------------------

def latin_square(n: int) -> Matrix:
    """Construct the cyclic Latin square of order n."""
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def reduced_latin_square(n: int) -> Matrix:
    """Return a reduced (normalised) Latin square using the cyclic construction."""
    return latin_square(n)


def is_latin_square(M: Matrix) -> bool:
    n = len(M)
    s = set(range(n))
    for row in M:
        if set(row) != s:
            return False
    for j in range(n):
        if {M[i][j] for i in range(n)} != s:
            return False
    return True


# ---------------------------------------------------------------------------
# Determinant and permanent (Ryser's formula)
# ---------------------------------------------------------------------------

def determinant(M: Matrix) -> float:
    """Gaussian elimination determinant."""
    n = len(M)
    A = [row[:] for row in M]
    det = 1.0
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if abs(A[row][col]) > 1e-12:
                pivot = row
                break
        if pivot is None:
            return 0.0
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            det *= -1
        det *= A[col][col]
        for row in range(col + 1, n):
            factor = A[row][col] / A[col][col]
            for k in range(col, n):
                A[row][k] -= factor * A[col][k]
    return det


def permanent(M: Matrix) -> int:
    """Compute the permanent using Ryser's inclusion-exclusion formula."""
    n = len(M)
    total = 0
    for s in range(1, 1 << n):
        bits = [i for i in range(n) if s & (1 << i)]
        col_sums = [sum(M[row][col] for col in bits) for row in range(n)]
        prod = 1
        for v in col_sums:
            prod *= v
        total += ((-1) ** (n - len(bits))) * prod
    return int(round((-1) ** n * total))


# ---------------------------------------------------------------------------
# Hadamard matrix sequences
# ---------------------------------------------------------------------------

def hadamard_matrix(order: int) -> Matrix:
    """Build the Sylvester-type Hadamard matrix of the given power-of-2 order."""
    if order == 1:
        return [[1]]
    if order & (order - 1):
        raise ValueError("order must be a power of 2")
    H = [[1]]
    size = 1
    while size < order:
        H2 = []
        for row in H:
            H2.append(row + row)
        for row in H:
            H2.append(row + [-x for x in row])
        H = H2
        size *= 2
    return H


def hadamard_flat_sequence(order: int) -> List[int]:
    """Flatten a Hadamard matrix row-major into a sequence of ±1."""
    H = hadamard_matrix(order)
    return [v for row in H for v in row]


# ---------------------------------------------------------------------------
# Characteristic-polynomial integer sequences
# ---------------------------------------------------------------------------

def companion_matrix(coeffs: List[float]) -> Matrix:
    """Companion matrix for monic polynomial with given coefficients (constant last)."""
    n = len(coeffs)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        C[i + 1][i] = 1.0
    for i in range(n):
        C[i][n - 1] = -coeffs[n - 1 - i]
    return C


def linear_recurrence_sequence(coeffs: List[int], init: List[int], count: int) -> List[int]:
    """Generate a linear recurrence sequence: a[n] = sum(c[i]*a[n-1-i]).

    coeffs[0] multiplies a[n-1], coeffs[1] multiplies a[n-2], etc.
    init gives the first len(coeffs) terms.
    """
    if len(init) != len(coeffs):
        raise ValueError("len(init) must equal len(coeffs)")
    seq = list(init)
    while len(seq) < count:
        nxt = sum(coeffs[i] * seq[-1 - i] for i in range(len(coeffs)))
        seq.append(nxt)
    return seq[:count]


# ---------------------------------------------------------------------------
# Toeplitz and Circulant integer sequences
# ---------------------------------------------------------------------------

def toeplitz_row(first_row: List[int], first_col: List[int]) -> Matrix:
    n = len(first_col)
    m = len(first_row)
    M = []
    for i in range(n):
        row = []
        for j in range(m):
            if j >= i:
                row.append(first_row[j - i])
            else:
                row.append(first_col[i - j])
        M.append(row)
    return M


def circulant_matrix(first_row: List[int]) -> Matrix:
    n = len(first_row)
    return [[first_row[(j - i) % n] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Knight's tour (Warnsdorff's heuristic) — numeric sequence of moves
# ---------------------------------------------------------------------------

def knights_tour(n: int = 8, start: Tuple[int, int] = (0, 0)) -> Optional[List[int]]:
    """Return a knight's tour as a flat list of square indices (row*n+col) or None."""
    moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
             (1, 2), (1, -2), (-1, 2), (-1, -2)]

    board = [[-1] * n for _ in range(n)]
    r, c = start
    board[r][c] = 0
    path = [r * n + c]

    def degree(rr: int, cc: int) -> int:
        return sum(
            1 for dr, dc in moves
            if 0 <= rr + dr < n and 0 <= cc + dc < n and board[rr + dr][cc + dc] == -1
        )

    for step in range(1, n * n):
        candidates = [
            (dr, dc) for dr, dc in moves
            if 0 <= r + dr < n and 0 <= c + dc < n and board[r + dr][c + dc] == -1
        ]
        if not candidates:
            return None
        dr, dc = min(candidates, key=lambda d: degree(r + d[0], c + d[1]))
        r, c = r + dr, c + dc
        board[r][c] = step
        path.append(r * n + c)

    return path


# ---------------------------------------------------------------------------
# Matrix sequence generator class
# ---------------------------------------------------------------------------

class MatrixSequenceGenerator:
    """High-level generator for matrix-derived integer sequences."""

    def pascal_rows(self, count: int) -> List[List[int]]:
        return [pascal_row(n) for n in range(count)]

    def pascal_flat(self, rows: int) -> List[int]:
        out = []
        for n in range(rows):
            out.extend(pascal_row(n))
        return out

    def magic_squares(self, sizes: List[int]) -> List[Matrix]:
        result = []
        for n in sizes:
            if n % 2 == 1:
                result.append(magic_square_odd(n))
            elif n % 4 == 0:
                result.append(magic_square_doubly_even(n))
        return result

    def recurrence(self, coeffs: List[int], init: List[int], count: int) -> List[int]:
        return linear_recurrence_sequence(coeffs, init, count)

    def hadamard_sequence(self, order: int) -> List[int]:
        return hadamard_flat_sequence(order)

    def tour(self, n: int = 8) -> Optional[List[int]]:
        return knights_tour(n)
