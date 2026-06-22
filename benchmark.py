"""
=============================================================================
 BENCHMARK FRAMEWORK - So Sanh Hieu Nang Cac Thuat Toan AI (Timetable)
=============================================================================
 So sanh toan bo 12 thuat toan theo:
   - Thoi gian chay (ms)
   - So node mo rong
   - Chi phi vi pham (cost)
   - Bo nho su dung (KB)
=============================================================================
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import time
import random
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Đảm bảo import đúng project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DEFAULT_SMALL_TIMETABLE_DATA_PATH
from data.timetable_loader import TimetableLoader
from src.algorithms.uninformed.bfs import BreadthFirstSearch
from src.algorithms.uninformed.dfs import DepthFirstSearch
from src.algorithms.informed.greedy import GreedyBestFirstSearch
from src.algorithms.informed.astar import AStarSearch
from src.algorithms.local.hill_climbing import HillClimbingSearch
from src.algorithms.local.simulated_annealing import SimulatedAnnealingSearch
from src.algorithms.complex.and_or_search import AndOrSearch
from src.algorithms.complex.online_search import LrtaStarSearch
from src.algorithms.csp.backtracking import CSPBacktrackingSearch
from src.algorithms.csp.min_conflicts import CSPMinConflictsSearch
from src.algorithms.adversarial.minimax import MinimaxSearch
from src.algorithms.adversarial.alpha_beta import AlphaBetaSearch

@dataclass
class BenchmarkResult:
    """Kết quả đo lường của một lần chạy thuật toán."""
    algorithm_name: str
    category: str
    success: bool
    execution_time_ms: float       # Thời gian chạy (ms)
    explored_nodes: int            # Số node mở rộng
    path_cost: float               # Điểm phạt vi phạm
    memory_kb: float               # Bộ nhớ peak (KB)
    path_length: int = 0          # Số lớp được xếp
    extra_info: Dict[str, Any] = field(default_factory=dict)

def run_with_memory_tracking(func, *args, **kwargs) -> Tuple[Any, float]:
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024  # bytes → KB

def bench_uninformed(graph) -> List[BenchmarkResult]:
    algorithms = {
        "BFS": BreadthFirstSearch(),
        "DFS": DepthFirstSearch(),
    }
    results = []
    print("  [1/6] Dang benchmark Nhom 1 - Uninformed Search (BFS / DFS)...")

    for name, searcher in algorithms.items():
        (path, cost, stats), mem_kb = run_with_memory_tracking(
            searcher.search, graph, "", ""
        )
        results.append(BenchmarkResult(
            algorithm_name=name,
            category="Uninformed Search",
            success=path is not None,
            execution_time_ms=stats.get("execution_time", 0.0) * 1000,
            explored_nodes=stats.get("explored_nodes", 0),
            path_cost=cost,
            memory_kb=mem_kb,
            path_length=len(path) if path else 0,
        ))
        status = "[OK]" if path else "[FAIL]"
        print(f"     {status} {name}: {results[-1].execution_time_ms:.3f} ms | "
              f"{results[-1].explored_nodes} nodes | cost={cost:.2f} | "
              f"mem={mem_kb:.2f} KB")
    return results

def bench_informed(graph) -> List[BenchmarkResult]:
    algorithms = {
        "Greedy Best-First": GreedyBestFirstSearch(),
        "A*": AStarSearch(),
    }
    results = []
    print("  [2/6] Dang benchmark Nhom 2 - Informed Search (Greedy / A*)...")

    for name, searcher in algorithms.items():
        (path, cost, stats), mem_kb = run_with_memory_tracking(
            searcher.search, graph, "", ""
        )
        results.append(BenchmarkResult(
            algorithm_name=name,
            category="Informed Search",
            success=path is not None,
            execution_time_ms=stats.get("execution_time", 0.0) * 1000,
            explored_nodes=stats.get("explored_nodes", 0),
            path_cost=cost,
            memory_kb=mem_kb,
            path_length=len(path) if path else 0,
        ))
        status = "[OK]" if path else "[FAIL]"
        print(f"     {status} {name}: {results[-1].execution_time_ms:.3f} ms | "
              f"{results[-1].explored_nodes} nodes | cost={cost:.2f} | "
              f"mem={mem_kb:.2f} KB")
    return results

def bench_local_search(graph) -> List[BenchmarkResult]:
    algorithms = {
        "Hill Climbing": HillClimbingSearch(),
        "Simulated Annealing": SimulatedAnnealingSearch(),
    }
    results = []
    print("  [3/6] Dang benchmark Nhom 3 - Local Search (Hill Climbing / SA)...")

    for name, searcher in algorithms.items():
        (path, cost, stats), mem_kb = run_with_memory_tracking(
            searcher.search, graph, "", ""
        )
        results.append(BenchmarkResult(
            algorithm_name=name,
            category="Local Search",
            success=path is not None,
            execution_time_ms=stats.get("execution_time", 0.0) * 1000,
            explored_nodes=stats.get("explored_nodes", 0),
            path_cost=cost,
            memory_kb=mem_kb,
            path_length=len(path) if path else 0,
        ))
        status = "[OK]" if path else "[FAIL]"
        print(f"     {status} {name}: {results[-1].execution_time_ms:.3f} ms | "
              f"{results[-1].explored_nodes} states | cost={cost:.2f} | "
              f"mem={mem_kb:.2f} KB")
    return results

def bench_complex_env(graph) -> List[BenchmarkResult]:
    results = []
    print("  [4/6] Dang benchmark Nhom 4 - Complex Environment (AND-OR / LRTA*)...")

    # AND-OR Search
    (path, cost, stats), mem_kb = run_with_memory_tracking(
        AndOrSearch().search, graph, "CS101_N01", ""
    )
    results.append(BenchmarkResult(
        algorithm_name="AND-OR Search",
        category="Complex Env",
        success=path is not None,
        execution_time_ms=stats.get("execution_time", 0.0) * 1000,
        explored_nodes=stats.get("explored_nodes", 0),
        path_cost=cost,
        memory_kb=mem_kb,
        path_length=len(path) if path else 0,
    ))
    status = "[OK]" if path else "[FAIL]"
    print(f"     {status} AND-OR Search: {results[-1].execution_time_ms:.3f} ms | "
          f"{results[-1].explored_nodes} nodes | cost={cost:.2f} | "
          f"mem={mem_kb:.2f} KB")

    # LRTA* (Online Search)
    (path, cost, stats), mem_kb = run_with_memory_tracking(
        LrtaStarSearch().search, graph, "", "", dynamic_changes=True
    )
    results.append(BenchmarkResult(
        algorithm_name="LRTA* (Online)",
        category="Complex Env",
        success=path is not None,
        execution_time_ms=stats.get("execution_time", 0.0) * 1000,
        explored_nodes=stats.get("explored_nodes", 0),
        path_cost=cost,
        memory_kb=mem_kb,
        path_length=len(path) if path else 0,
    ))
    status = "[OK]" if path else "[FAIL]"
    print(f"     {status} LRTA*: {results[-1].execution_time_ms:.3f} ms | "
          f"{results[-1].explored_nodes} steps | cost={cost:.2f} | "
          f"mem={mem_kb:.2f} KB")

    return results

def bench_csp(graph) -> List[BenchmarkResult]:
    algorithms = {
        "CSP Backtracking": CSPBacktrackingSearch(),
        "CSP Min-Conflicts": CSPMinConflictsSearch(),
    }
    results = []
    print("  [5/6] Dang benchmark Nhom 5 - CSP (Backtracking / Min-Conflicts)...")

    for name, solver in algorithms.items():
        (path, cost, stats), mem_kb = run_with_memory_tracking(
            solver.search, graph, "", ""
        )
        results.append(BenchmarkResult(
            algorithm_name=name,
            category="CSP",
            success=path is not None,
            execution_time_ms=stats.get("execution_time", 0.0) * 1000,
            explored_nodes=stats.get("explored_nodes", 0),
            path_cost=cost,
            memory_kb=mem_kb,
            path_length=len(path) if path else 0,
        ))
        status = "[OK]" if path else "[FAIL]"
        print(f"     {status} {name}: {results[-1].execution_time_ms:.3f} ms | "
              f"{results[-1].explored_nodes} states | "
              f"mem={mem_kb:.2f} KB")
    return results

def bench_adversarial(graph, depth: int = 2) -> List[BenchmarkResult]:
    algorithms = {
        "Minimax": MinimaxSearch(),
        "Alpha-Beta": AlphaBetaSearch(),
    }
    results = []
    print("  [6/6] Dang benchmark Nhom 6 - Adversarial (Minimax / Alpha-Beta)...")

    for name, searcher in algorithms.items():
        (path, cost, stats), mem_kb = run_with_memory_tracking(
            searcher.search, graph, "CS101_N01", "", depth=depth
        )
        results.append(BenchmarkResult(
            algorithm_name=name,
            category="Adversarial",
            success=path is not None,
            execution_time_ms=stats.get("execution_time", 0.0) * 1000,
            explored_nodes=stats.get("explored_nodes", 0),
            path_cost=cost,
            memory_kb=mem_kb,
            path_length=len(path) if path else 0,
        ))
        status = "[OK]" if path else "[FAIL]"
        print(f"     {status} {name}: {results[-1].execution_time_ms:.3f} ms | "
              f"{results[-1].explored_nodes} nodes | "
              f"mem={mem_kb:.2f} KB")
    return results

# ANSI colors
C_HEADER  = "\033[1;36m"   # Cyan bold
C_GREEN   = "\033[0;32m"
C_YELLOW  = "\033[0;33m"
C_RED     = "\033[0;31m"
C_BOLD    = "\033[1m"
C_DIM     = "\033[2m"
C_RESET   = "\033[0m"
C_BEST    = "\033[1;33m"   # Yellow bold

def _col(text: str, width: int, align: str = "<") -> str:
    import re
    plain = re.sub(r'\033\[[0-9;]*m', '', text)
    padding = width - len(plain)
    if align == ">":
        return " " * padding + text
    elif align == "^":
        lp = padding // 2
        rp = padding - lp
        return " " * lp + text + " " * rp
    return text + " " * padding

def print_group_table(results: List[BenchmarkResult], group_title: str) -> None:
    if not results:
        return

    successful = [r for r in results if r.success]
    best_time    = min((r.execution_time_ms for r in successful), default=None)
    best_nodes   = min((r.explored_nodes    for r in successful), default=None)
    best_cost    = min((r.path_cost         for r in successful), default=None)
    best_mem     = min((r.memory_kb         for r in successful), default=None)

    COL_W = [22, 10, 14, 14, 14, 14, 8]
    headers = ["Thuat Toan", "Ket Qua", "TG (ms)",
               "Node MR", "Diem Phat", "Bo Nho KB", "Lop HP"]
    sep = "-" * (sum(COL_W) + len(COL_W) * 3 + 1)

    print(f"\n{C_HEADER}{'=' * len(sep)}{C_RESET}")
    print(f"{C_HEADER}  {group_title}{C_RESET}")
    print(f"{C_HEADER}{'=' * len(sep)}{C_RESET}")

    hrow = "|"
    for h, w in zip(headers, COL_W):
         hrow += f" {C_BOLD}{_col(h, w, '^')}{C_RESET} |"
    print(hrow)
    print(f"{C_DIM}{sep}{C_RESET}")

    for r in results:
        def hl(val, best, fmt_fn):
            s = fmt_fn(val)
            if best is not None and abs(val - best) < 1e-9:
                return f"{C_BEST}{s}{C_RESET}"
            return s

        status_str = f"{C_GREEN}✓ OK{C_RESET}" if r.success else f"{C_RED}✗ Fail{C_RESET}"

        cols = [
            _col(r.algorithm_name, COL_W[0]),
            _col(status_str,       COL_W[1], "^"),
            _col(hl(r.execution_time_ms, best_time,  lambda v: f"{v:.3f}"), COL_W[2], ">"),
            _col(hl(r.explored_nodes,    best_nodes,  lambda v: f"{v:,}"),  COL_W[3], ">"),
            _col(hl(r.path_cost,         best_cost,   lambda v: f"{v:.2f}"), COL_W[4], ">"),
            _col(hl(r.memory_kb,         best_mem,    lambda v: f"{v:.2f}"), COL_W[5], ">"),
            _col(str(r.path_length),     COL_W[6], ">"),
        ]
        row = "│"
        for c in cols:
            row += f" {c} │"
        print(row)

    print(f"{C_DIM}{sep}{C_RESET}")
    if best_time is not None:
        print(f"  {C_BEST}★{C_RESET} = giá trị tốt nhất trong nhóm\n")

def print_master_summary(all_results: List[BenchmarkResult]) -> None:
    print(f"\n{C_HEADER}{'='*80}{C_RESET}")
    print(f"{C_HEADER}  BANG TONG KET TOAN BO THUAT TOAN{C_RESET}")
    print(f"{C_HEADER}{'='*80}{C_RESET}")

    categories: Dict[str, List[BenchmarkResult]] = {}
    for r in all_results:
        categories.setdefault(r.category, []).append(r)

    COL_W2 = [22, 20, 14, 14, 14, 14]
    headers2 = ["Thuat Toan", "Nhom", "TG (ms)", "Node MR",
                "Diem Phat", "Bo Nho KB"]
    sep2 = "-" * (sum(COL_W2) + len(COL_W2) * 3 + 1)

    hrow = "|"
    for h, w in zip(headers2, COL_W2):
        hrow += f" {C_BOLD}{_col(h, w, '^')}{C_RESET} |"
    print(hrow)
    print(f"{C_DIM}{sep2}{C_RESET}")

    for cat, rlist in categories.items():
        for r in rlist:
            status_icon = f"{C_GREEN}[OK]{C_RESET}" if r.success else f"{C_RED}[FAIL]{C_RESET}"
            cols = [
                _col(f"{status_icon} {r.algorithm_name}", COL_W2[0]),
                _col(r.category,           COL_W2[1]),
                _col(f"{r.execution_time_ms:.3f}", COL_W2[2], ">"),
                _col(f"{r.explored_nodes:,}",      COL_W2[3], ">"),
                _col(f"{r.path_cost:.2f}",         COL_W2[4], ">"),
                _col(f"{r.memory_kb:.2f}",         COL_W2[5], ">"),
            ]
            row = "|"
            for c in cols:
                row += f" {c} |"
            print(row)
        print(f"{C_DIM}{sep2}{C_RESET}")

    successful = [r for r in all_results if r.success]
    if successful:
        fastest   = min(successful, key=lambda r: r.execution_time_ms)
        fewest    = min(successful, key=lambda r: r.explored_nodes)
        cheapest  = min(successful, key=lambda r: r.path_cost)
        smallest  = min(successful, key=lambda r: r.memory_kb)

        print(f"\n  {C_BOLD}Thong ke:{C_RESET}")
        print(f"   [Nhanh nhat]  : {C_BEST}{fastest.algorithm_name}{C_RESET}"
              f" [{fastest.category}] - {fastest.execution_time_ms:.3f} ms")
        print(f"   [It node nhat]: {C_BEST}{fewest.algorithm_name}{C_RESET}"
              f" [{fewest.category}] - {fewest.explored_nodes:,} nodes")
        print(f"   [Diem phạt thap]: {C_BEST}{cheapest.algorithm_name}{C_RESET}"
              f" [{cheapest.category}] - {cheapest.path_cost:.2f}")
        print(f"   [It bo nho]   : {C_BEST}{smallest.algorithm_name}{C_RESET}"
              f" [{smallest.category}] - {smallest.memory_kb:.2f} KB")
    print()

def export_csv(all_results: List[BenchmarkResult], filepath: str) -> None:
    import csv
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Thuat Toan", "Nhom", "Ket Qua",
            "Thoi Gian (ms)", "So Node Mo Rong",
            "Diem Phat Vi Pham", "Bo Nho (KB)", "So Lop Xep Duoc"
        ])
        for r in all_results:
            writer.writerow([
                r.algorithm_name,
                r.category,
                "OK" if r.success else "FAIL",
                f"{r.execution_time_ms:.4f}",
                r.explored_nodes,
                f"{r.path_cost:.4f}",
                f"{r.memory_kb:.4f}",
                r.path_length,
            ])
    print(f"  [CSV] Da xuat ket qua: {filepath}")

def main():
    print(f"\n{C_HEADER}{'='*70}{C_RESET}")
    print(f"{C_HEADER}  BENCHMARK FRAMEWORK - AI Timetabling Search Algorithms{C_RESET}")
    print(f"{C_HEADER}{'='*70}{C_RESET}\n")

    if not os.path.exists(DEFAULT_SMALL_TIMETABLE_DATA_PATH):
        print(f"[LOI] Khong tim thay du lieu mau: {DEFAULT_SMALL_TIMETABLE_DATA_PATH}")
        return

    print(f"  Dang tai du lieu: {DEFAULT_SMALL_TIMETABLE_DATA_PATH}")
    graph = TimetableLoader.load_from_json(DEFAULT_SMALL_TIMETABLE_DATA_PATH)
    print(f"  [OK] Tai du lieu thanh cong.\n")

    all_results: List[BenchmarkResult] = []

    r1 = bench_uninformed(graph)
    r2 = bench_informed(graph)
    r3 = bench_local_search(graph)
    r4 = bench_complex_env(graph)
    r5 = bench_csp(graph)
    r6 = bench_adversarial(graph, depth=2)

    all_results = r1 + r2 + r3 + r4 + r5 + r6

    print_group_table(r1, "NHOM 1 - UNINFORMED SEARCH (BFS / DFS)")
    print_group_table(r2, "NHOM 2 - INFORMED SEARCH (Greedy / A*)")
    print_group_table(r3, "NHOM 3 - LOCAL SEARCH (Hill Climbing / SA)")
    print_group_table(r4, "NHOM 4 - COMPLEX ENVIRONMENT")
    print_group_table(r5, "NHOM 5 - CSP (Backtracking / Min-Conflicts)")
    print_group_table(r6, "NHOM 6 - ADVERSARIAL SEARCH")

    print_master_summary(all_results)

    csv_path = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    export_csv(all_results, csv_path)

    print(f"\n{C_GREEN}  [DONE] Benchmark hoan tat!{C_RESET}\n")

if __name__ == "__main__":
    main()
