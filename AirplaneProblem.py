from collections import defaultdict
import random
from typing import List, Tuple, Dict


class Node:
    def __init__(self, x: int, y: int):
        self.x = x  # cost to move RIGHT
        self.y = y  # cost to move DOWN
        self.price = 0

    def __repr__(self):
        return f"({self.x},{self.y}, price={self.price})"


# --------------------------------------------------
# Internal DP helper: compute all distinct costs up to limit
# --------------------------------------------------

def _dp_costs(mat, limit):
    """
    Dynamic Programming to compute distinct costs and their path counts.
    
    Algorithm:
    - For each cell (i,j), maintain a dictionary mapping cost -> number of paths
    - Transition: from (i,j) we can go to (i+1,j) with cost+mat[i][j].y
                   or to (i,j+1) with cost+mat[i][j].x
    - We limit the number of distinct costs per cell to 'limit' to avoid explosion
    
    Complexity: O(n*m*limit) time, O(n*m*limit) space
    """
    n, m = len(mat), len(mat[0])

    # dp[i][j] = dictionary mapping cost -> count of paths reaching (i,j) with that cost
    dp = [[defaultdict(int) for _ in range(m)] for _ in range(n)]
    dp[0][0][0] = 1  # cost 0, one path at start

    for i in range(n):
        for j in range(m):
            # Get costs sorted to keep only the smallest 'limit' costs
            costs_at_cell = sorted(dp[i][j].items())
            if len(costs_at_cell) > limit:
                # Keep only the smallest 'limit' costs
                costs_at_cell = costs_at_cell[:limit]
                dp[i][j] = defaultdict(int, costs_at_cell)
            
            for cost, cnt in dp[i][j].items():
                # Move down
                if i + 1 < n:
                    new_cost = cost + mat[i][j].y
                    dp[i + 1][j][new_cost] += cnt
                
                # Move right
                if j + 1 < m:
                    new_cost = cost + mat[i][j].x
                    dp[i][j + 1][new_cost] += cnt

    return dp[n - 1][m - 1]


# --------------------------------------------------
# Question 1
# --------------------------------------------------

def second_best_cost(mat) -> int | None:
    """
    Returns the second best (strictly larger) cost to reach the destination,
    or None if it doesn't exist.
    
    Algorithm: Compute top 2 costs using DP, return the second one if exists.
    Complexity: O(n*m) time, O(n*m) space
    """
    n, m = len(mat), len(mat[0])
    
    # Edge case: single row or column means only one path
    if n == 1 or m == 1:
        return None
    
    costs_dict = _dp_costs(mat, 2)
    costs = sorted(costs_dict.keys())
    
    return costs[1] if len(costs) >= 2 else None


def second_best_count(mat) -> int:
    """
    Returns how many paths achieve the second best cost.
    Returns 0 if second best doesn't exist.
    
    Complexity: O(n*m) time, O(n*m) space
    """
    n, m = len(mat), len(mat[0])
    
    # Edge case: single row or column means only one path
    if n == 1 or m == 1:
        return 0
    
    costs_dict = _dp_costs(mat, 2)
    sc = second_best_cost(mat)
    
    return costs_dict[sc] if sc is not None else 0


def one_second_best_path(mat) -> str | None:
    """
    Returns a path string (e.g. '0011' where 0=Right, 1=Down)
    that achieves the second best cost, or None if doesn't exist.
    
    Algorithm: DFS to find all second-best paths, return the first one.
    Complexity: O(n*m + paths) where paths is number of second-best paths
    """
    all_paths = all_second_best_paths(mat, L=1)
    return all_paths[0] if all_paths else None


def all_second_best_paths(mat, L=50) -> List[str]:
    """
    Returns up to L second-best paths (paths whose cost equals the second-best cost)
    encoded as a string of moves: 0=Right, 1=Down.
    
    Algorithm:
    1. Find second-best cost using DP
    2. DFS from (0,0) to (n-1,m-1) collecting only paths with target cost
    3. Return up to L paths in lexicographic order
    
    Complexity: O(n*m + total_paths) where total_paths can be exponential
    """
    n, m = len(mat), len(mat[0])
    
    # Edge case: single row or column means only one path
    if n == 1 or m == 1:
        return []

    # Find second-best cost
    costs_dict = _dp_costs(mat, 2)
    costs = sorted(costs_dict.keys())
    
    if len(costs) < 2:
        return []

    target = costs[1]
    paths = []

    def dfs(i, j, cost, path):
        """DFS to enumerate paths with specific cost"""
        if len(paths) >= L:
            return
        
        if i == n - 1 and j == m - 1:
            if cost == target:
                paths.append(path)
            return
        
        # Try down first (produces '1' = lexicographically larger)
        if i + 1 < n:
            dfs(i + 1, j, cost + mat[i][j].y, path + "1")
        
        # Try right (produces '0' = lexicographically smaller)
        if j + 1 < m:
            dfs(i, j + 1, cost + mat[i][j].x, path + "0")

    dfs(0, 0, 0, "")
    
    # Return sorted paths (lexicographic order)
    return sorted(paths)[:L]


# --------------------------------------------------
# Question 2
# --------------------------------------------------

def top_k_costs(mat, k=5) -> List[int]:
    """
    Return up to k distinct smallest costs to destination, sorted.
    
    Algorithm: Use DP to compute top k costs efficiently.
    Complexity: O(n*m*k) time, O(n*m*k) space
    """
    costs_dict = _dp_costs(mat, k)
    return sorted(costs_dict.keys())[:k]


def top_k_costs_with_counts(mat, k=5) -> List[Tuple[int, int]]:
    """
    Return up to k pairs (cost, count) for distinct smallest costs.
    
    Algorithm: Use DP to compute costs and counts, return top k.
    Complexity: O(n*m*k) time, O(n*m*k) space
    """
    costs_dict = _dp_costs(mat, k)
    return sorted(costs_dict.items())[:k]


# --------------------------------------------------
# Question 3
# --------------------------------------------------

def all_distinct_costs(mat, T=200) -> List[int]:
    """
    Return up to T distinct costs to destination, sorted.
    
    Algorithm: Use DP with limit T to find all distinct costs.
    Complexity: O(n*m*T) time, O(n*m*T) space
    """
    costs_dict = _dp_costs(mat, T)
    return sorted(costs_dict.keys())


def count_paths_by_cost(mat, T=200) -> Dict[int, int]:
    """
    Return dict cost->count for up to T smallest distinct costs.
    
    Algorithm: Use DP to compute all costs and their path counts.
    Complexity: O(n*m*T) time, O(n*m*T) space
    """
    costs_dict = _dp_costs(mat, T)
    return dict(sorted(costs_dict.items()))


# --------------------------------------------------
# Question 4
# --------------------------------------------------

def student_seed(student_id: str) -> int:
    """
    Return deterministic seed derived from student_id (9 digits).
    
    Algorithm: Use polynomial rolling hash with base 31 for determinism.
    This ensures same student_id always produces same seed across all systems.
    
    Requirements:
    - student_id must be exactly 9 digits
    - Result must be deterministic (same on all machines/Python versions)
    - Cannot use hash() as it's not deterministic across runs
    
    Complexity: O(1) time, O(1) space
    """
    if not (isinstance(student_id, str) and len(student_id) == 9 and student_id.isdigit()):
        raise ValueError("student_id must be a string of exactly 9 digits")
    
    # Polynomial rolling hash for determinism
    seed = 0
    for ch in student_id:
        seed = seed * 31 + int(ch)
    
    return seed


def build_matrix_from_seed(student_id: str, n: int, m: int, max_w: int = 20):
    """
    Return mat[n][m] of Nodes using deterministic randomness from seed.
    
    Algorithm:
    1. Generate seed from student_id
    2. Create local Random instance with this seed
    3. Generate random weights for each cell (0 to max_w)
    4. Set boundary conditions (last row y=0, last column x=0)
    
    Special conditions:
    - Last cell (n-1, m-1): x=0, y=0
    - Last row (i=n-1): y=0 (no down movement)
    - Last column (j=m-1): x=0 (no right movement)
    
    Complexity: O(n*m) time, O(n*m) space
    """
    seed = student_seed(student_id)
    rng = random.Random(seed)  # Local random instance for reproducibility

    mat = [[None for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            # Generate x: 0 if last column, else random
            x = rng.randint(0, max_w) if j < m - 1 else 0
            
            # Generate y: 0 if last row, else random
            y = rng.randint(0, max_w) if i < n - 1 else 0
            
            mat[i][j] = Node(x, y)

    # Ensure destination has both costs as 0
    mat[n - 1][m - 1].x = 0
    mat[n - 1][m - 1].y = 0
    
    return mat


# --------------------------------------------------
# Question 5 - main
# --------------------------------------------------

def main():
    """
    Main function demonstrating all functionality.
    
    !!! IMPORTANT: Replace the student_id with YOUR actual student ID !!!
    """
    # !!! REPLACE THIS WITH YOUR STUDENT ID !!!
    student_id = "123456789"  # TODO: Replace with your 9-digit student ID
    
    # Parameters
    n, m = 4, 4
    max_w = 10
    k = 5
    T = 200
    L = 50

    # Build matrix from seed
    mat = build_matrix_from_seed(student_id, n, m, max_w)

    print("=" * 60)
    print("Dynamic Programming - Airplane Problem")
    print("=" * 60)
    print(f"Student ID: {student_id}")
    print(f"Seed: {student_seed(student_id)}")
    print(f"Grid size: n={n}, m={m}")
    print(f"Max weight: {max_w}")
    print(f"Parameters: k={k}, T={T}, L={L}")
    print("=" * 60)

    # Question 1: Second Best
    print("\n[Question 1] Second Best Path:")
    print("-" * 60)
    best = top_k_costs(mat, 1)[0] if top_k_costs(mat, 1) else None
    second_best = second_best_cost(mat)
    print(f"Best cost: {best}")
    print(f"Second best cost: {second_best}")
    print(f"Second best count: {second_best_count(mat)}")
    print(f"One second best path: {one_second_best_path(mat)}")
    
    all_sb_paths = all_second_best_paths(mat, L)
    print(f"All second best paths ({len(all_sb_paths)} total):")
    if len(all_sb_paths) <= 10:
        for path in all_sb_paths:
            print(f"  {path}")
    else:
        for path in all_sb_paths[:5]:
            print(f"  {path}")
        print(f"  ... ({len(all_sb_paths) - 5} more paths)")

    # Question 2: Top-k costs
    print(f"\n[Question 2] Top-{k} Costs:")
    print("-" * 60)
    print(f"Top {k} costs: {top_k_costs(mat, k)}")
    
    top_k_with_counts = top_k_costs_with_counts(mat, k)
    print(f"Top {k} costs with counts:")
    for cost, count in top_k_with_counts:
        print(f"  Cost {cost}: {count} path(s)")

    # Question 3: All distinct costs
    print(f"\n[Question 3] All Distinct Costs (up to T={T}):")
    print("-" * 60)
    all_costs = all_distinct_costs(mat, T)
    print(f"Number of distinct costs: {len(all_costs)}")
    print(f"All distinct costs (first 20): {all_costs[:20]}")
    if len(all_costs) > 20:
        print(f"  ... ({len(all_costs) - 20} more costs)")
    
    paths_by_cost = count_paths_by_cost(mat, T)
    print(f"\nCount paths by cost (first 10):")
    for i, (cost, count) in enumerate(paths_by_cost.items()):
        if i >= 10:
            print(f"  ... ({len(paths_by_cost) - 10} more entries)")
            break
        print(f"  Cost {cost}: {count} path(s)")

    print("\n" + "=" * 60)
    print("End of execution")
    print("=" * 60)


if __name__ == "__main__":
    main()
