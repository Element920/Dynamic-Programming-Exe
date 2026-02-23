from collections import defaultdict
import random
from typing import List, Tuple, Dict


class Node:
    def __init__(self, x: int, y: int):
        self.x = x  # עלות מעבר ימינה
        self.y = y  # עלות מעבר למטה
        self.price = 0  # העלות המינימלית מ-(0,0) עד לצומת זה

    def __repr__(self):
        return f"({self.x},{self.y}, price={self.price})"


def _dp_costs(mat, limit):
    """
    פונקציית עזר המשתמשת בתכנות דינמי כדי לחשב עלויות שונות ומספר מסלולים לכל עלות.

    הרעיון:
    עבור כל תא נשמור מילון:
        עלות -> מספר מסלולים שמגיעים לתא בעלות זו

    נשמור רק את limit העלויות הקטנות ביותר בכל תא כדי להגביל סיבוכיות.

    סיבוכיות זמן: O(n*m*limit)
    סיבוכיות מקום: O(n*m*limit)
    """
    n, m = len(mat), len(mat[0])

    # dp[i][j] = מילון שממפה עלות -> מספר מסלולים שמגיעים לתא (i,j)
    dp = [[defaultdict(int) for _ in range(m)] for _ in range(n)]
    dp[0][0][0] = 1  # בתא ההתחלה יש מסלול אחד בעלות 0

    for i in range(n):
        for j in range(m):
            # נשמור רק את העלויות הקטנות ביותר לפי המגבלה
            costs_at_cell = sorted(dp[i][j].items())
            if len(costs_at_cell) > limit:
                costs_at_cell = costs_at_cell[:limit]
                dp[i][j] = defaultdict(int, costs_at_cell)

            for cost, cnt in dp[i][j].items():
                # מעבר למטה
                if i + 1 < n:
                    new_cost = cost + mat[i][j].y
                    dp[i + 1][j][new_cost] += cnt

                # מעבר ימינה
                if j + 1 < m:
                    new_cost = cost + mat[i][j].x
                    dp[i][j + 1][new_cost] += cnt

    return dp[n - 1][m - 1]


# --------------------------------------------------
# שאלה 1
# --------------------------------------------------

def second_best_cost(mat) -> int | None:
    """
    מחזיר את העלות השנייה בטיבה (Strictly גדולה מהמינימום),
    או None אם לא קיימת עלות כזו.

    שיטה:
    מחשבים את שתי העלויות הקטנות ביותר בעזרת DP ומחזירים את השנייה.

    סיבוכיות:
    זמן: O(n*m)
    מקום: O(n*m)
    """
    n, m = len(mat), len(mat[0])

    # אם יש רק שורה אחת או עמודה אחת קיים מסלול יחיד בלבד
    if n == 1 or m == 1:
        return None

    costs_dict = _dp_costs(mat, 2)
    costs = sorted(costs_dict.keys())

    return costs[1] if len(costs) >= 2 else None


def second_best_count(mat) -> int:
    """
    מחזיר כמה מסלולים משיגים את העלות השנייה בטיבה.

    אם לא קיימת עלות שנייה — מוחזר 0.

    סיבוכיות:
    זמן: O(n*m)
    מקום: O(n*m)
    """
    n, m = len(mat), len(mat[0])

    if n == 1 or m == 1:
        return 0

    costs_dict = _dp_costs(mat, 2)
    sc = second_best_cost(mat)

    return costs_dict[sc] if sc is not None else 0


def one_second_best_path(mat) -> str | None:
    """
    מחזיר מסלול אחד שמממש את העלות השנייה בטיבה.

    הפורמט:
        0 = ימינה
        1 = למטה

    אם לא קיים מסלול כזה מוחזר None.
    """
    all_paths = all_second_best_paths(mat, L=1)
    return all_paths[0] if all_paths else None


def all_second_best_paths(mat, L=50) -> List[str]:
    """
    מחזיר עד L מסלולים שהעלות שלהם שווה לעלות השנייה בטיבה.

    שלבים:
    1. מחשבים את העלות השנייה בעזרת DP
    2. מבצעים DFS על כל המסלולים
    3. שומרים רק מסלולים עם העלות הרצויה
    4. מחזירים עד L מסלולים ממוינים לקסיקוגרפית

    סיבוכיות:
    O(n*m + מספר המסלולים האפשריים)
    """
    n, m = len(mat), len(mat[0])

    if n == 1 or m == 1:
        return []

    costs_dict = _dp_costs(mat, 2)
    costs = sorted(costs_dict.keys())

    if len(costs) < 2:
        return []

    target = costs[1]
    paths = []

    def dfs(i, j, cost, path):
        if len(paths) >= L:
            return

        if i == n - 1 and j == m - 1:
            if cost == target:
                paths.append(path)
            return

        # למטה
        if i + 1 < n:
            dfs(i + 1, j, cost + mat[i][j].y, path + "1")

        # ימינה
        if j + 1 < m:
            dfs(i, j + 1, cost + mat[i][j].x, path + "0")

    dfs(0, 0, 0, "")
    return sorted(paths)[:L]


# --------------------------------------------------
# שאלה 2
# --------------------------------------------------

def top_k_costs(mat, k=5) -> List[int]:
    """
    מחזיר עד k העלויות הקטנות ביותר (שונות) להגיע ליעד.
    """
    costs_dict = _dp_costs(mat, k)
    return sorted(costs_dict.keys())[:k]


def top_k_costs_with_counts(mat, k=5) -> List[Tuple[int, int]]:
    """
    מחזיר עד k זוגות מהצורה:
        (עלות, מספר מסלולים שמממשים אותה)
    """
    costs_dict = _dp_costs(mat, k)
    return sorted(costs_dict.items())[:k]


# --------------------------------------------------
# שאלה 3
# --------------------------------------------------

def all_distinct_costs(mat, T=200) -> List[int]:
    """
    מחזיר עד T עלויות שונות (ממוינות).
    """
    costs_dict = _dp_costs(mat, T)
    return sorted(costs_dict.keys())


def count_paths_by_cost(mat, T=200) -> Dict[int, int]:
    """
    מחזיר מילון:
        עלות -> מספר מסלולים
    עבור עד T העלויות הקטנות ביותר.
    """
    costs_dict = _dp_costs(mat, T)
    return dict(sorted(costs_dict.items()))


# --------------------------------------------------
# שאלה 4
# --------------------------------------------------

def student_seed(student_id: str) -> int:
    
    if not (isinstance(student_id, str) and len(student_id) == 9 and student_id.isdigit()):
        raise ValueError("student_id must be a string of exactly 9 digits")

    seed = 0
    for ch in student_id:
        seed = seed * 31 + int(ch)

    return seed


def build_matrix_from_seed(student_id: str, n: int, m: int, max_w: int = 20):
    """
    בונה מטריצת Nodes אקראית אך דטרמיניסטית לפי seed הנגזר מהת"ז.

    תנאים:
    - בתא האחרון: x=y=0
    - בשורה האחרונה: y=0
    - בעמודה האחרונה: x=0
    """
    seed = student_seed(student_id)
    rng = random.Random(seed)

    mat = [[None for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            x = rng.randint(0, max_w) if j < m - 1 else 0
            y = rng.randint(0, max_w) if i < n - 1 else 0
            mat[i][j] = Node(x, y)

    mat[n - 1][m - 1].x = 0
    mat[n - 1][m - 1].y = 0

    return mat


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    """
    פונקציית הדגמה ראשית.

    מבצעת:
    - יצירת מטריצה לפי ת"ז
    - הרצת כל הפונקציות
    - הדפסת התוצאות
    """

    student_id = "123456789"

    n, m = 4, 4
    max_w = 10
    k = 5
    T = 200
    L = 50

    mat = build_matrix_from_seed(student_id, n, m, max_w)

    print("Student ID:", student_id)
    print("Seed:", student_seed(student_id))
    print("Grid:", n, "x", m)

    best = top_k_costs(mat, 1)[0]
    print("Best cost:", best)
    print("Second best:", second_best_cost(mat))
    print("Second best count:", second_best_count(mat))
    print("One second best path:", one_second_best_path(mat))
    print("All second best paths:", all_second_best_paths(mat, L))

    print("Top k costs:", top_k_costs(mat, k))
    print("Top k costs with counts:", top_k_costs_with_counts(mat, k))

    print("All distinct costs:", all_distinct_costs(mat, T))
    print("Counts:", count_paths_by_cost(mat, T))


if __name__ == "__main__":
    main()
