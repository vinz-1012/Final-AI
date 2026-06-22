import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import random
import json
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
from src.visualization.timetable_visualizer import TimetableVisualizer

def main():
    print("=" * 70)
    print("=== HỆ THỐNG XẾP THỜI KHÓA BIỂU TỰ ĐỘNG BẰNG AI (UNIVERSITY TIMETABLE) ===")
    print("=" * 70)
    
    # 1. Load dữ liệu bài toán thời khóa biểu
    if not os.path.exists(DEFAULT_SMALL_TIMETABLE_DATA_PATH):
        print(f"Lỗi: Không tìm thấy file dữ liệu mặc định tại {DEFAULT_SMALL_TIMETABLE_DATA_PATH}")
        return
        
    print(f"Đang tải dữ liệu TKB mẫu nhỏ từ: {DEFAULT_SMALL_TIMETABLE_DATA_PATH}...")
    graph = TimetableLoader.load_from_json(DEFAULT_SMALL_TIMETABLE_DATA_PATH)
    print(f"Tải thành công! Không gian bài toán có:")
    print(f"  - {len(graph.courses)} Học phần")
    print(f"  - {len(graph.sections)} Lớp học phần cần xếp lịch")
    print(f"  - {len(graph.rooms)} Phòng học")
    print(f"  - {len(graph.lecturers)} Giảng viên")
    print(f"  - {len(graph.periods)} Ca học/tuần\n")

    # ----------------------------------------------------
    # PHẦN 1: UNINFORMED & INFORMED SEARCH
    # ----------------------------------------------------
    print("-" * 60)
    print("--- PHẦN 1: So sánh Uninformed & Informed Search ---")
    print("-" * 60)
    
    algorithms_systematic = {
        "1. Breadth-First Search (BFS)": BreadthFirstSearch(),
        "2. Depth-First Search (DFS)": DepthFirstSearch(),
        "3. Greedy Best-First Search (GBFS)": GreedyBestFirstSearch(),
        "4. A* Search": AStarSearch()
    }
    
    for name, searcher in algorithms_systematic.items():
        print(f"\n[+] Đang chạy: {name}...")
        path, cost, stats = searcher.search(graph, "", "")
        
        if path:
            print(f"  -> Xếp lịch thành công cho {len(path)}/{len(graph.sections)} lớp.")
            print(f"  -> Điểm phạt vi phạm: {cost:.1f}")
            print(f"  -> Số trạng thái đã duyệt: {stats.get('explored_nodes', 0)}")
            print(f"  -> Thời gian thực thi: {stats.get('execution_time', 0.0) * 1000:.2f} ms")
            print(f"  -> Fitness đạt được: {stats.get('fitness', 0.0):.2f}")
        else:
            print(f"  -> Thất bại: {stats.get('error', 'Không tìm thấy lời giải hợp lệ')}")

    # ----------------------------------------------------
    # PHẦN 2: LOCAL SEARCH (TỐI ƯU HÓA TKB)
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("--- PHẦN 2: Tối ưu hóa thời khóa biểu (Local Search) ---")
    print("="*70)
    
    local_searchers = {
        "Hill Climbing (Leo đồi)": HillClimbingSearch(),
        "Simulated Annealing (Luyện kim)": SimulatedAnnealingSearch()
    }
    
    for name, searcher in local_searchers.items():
        print(f"\n[+] Đang chạy: {name}...")
        path, cost, stats = searcher.search(graph, "", "")
        
        if path:
            print(f"  -> Đã tối ưu xong! Số lớp đã xếp: {len(path)}/{len(graph.sections)}")
            print(f"  -> Tổng điểm vi phạm (Cứng + Mềm): {cost:.1f}")
            print(f"  -> Số lân cận đã đánh giá: {stats.get('explored_nodes', 0)}")
            print(f"  -> Thời gian thực thi: {stats.get('execution_time', 0.0) * 1000:.2f} ms")
            print(f"  -> Fitness cuối cùng: {stats.get('fitness', 0.0):.2f}")
        else:
            print("  -> Thất bại.")

    # ----------------------------------------------------
    # PHẦN 3: BÀI TOÁN THỎA MÃN RÀNG BUỘC - CSP
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("--- PHẦN 3: Bài toán thỏa mãn ràng buộc (CSP) ---")
    print("="*70)
    
    csp_solvers = {
        "Backtracking Search (Quay lui CSP)": CSPBacktrackingSearch(),
        "Min-Conflicts (Cực tiểu mâu thuẫn CSP)": CSPMinConflictsSearch()
    }
    
    best_timetable = None
    
    for name, solver in csp_solvers.items():
        print(f"\n[+] Đang chạy: {name}...")
        path, cost, stats = solver.search(graph, "", "")
        
        if path:
            print(f"  -> Đã tìm thấy lời giải CSP hợp lệ hoàn toàn!")
            print(f"  -> Số vi phạm ràng buộc cứng: {stats.get('hard_violations', 0)}")
            print(f"  -> Số vi phạm ràng buộc mềm: {stats.get('soft_violations', {}).get('total', 0)}")
            print(f"  -> Số trạng thái đã duyệt: {stats.get('explored_nodes', 0)}")
            print(f"  -> Thời gian thực thi: {stats.get('execution_time', 0.0) * 1000:.2f} ms")
            if best_timetable is None and stats.get("hard_violations", 999) == 0:
                best_timetable = stats.get("timetable")
        else:
            print("  -> Thất bại: Không tìm được phương án thỏa mãn mọi ràng buộc cứng.")

    # ----------------------------------------------------
    # PHẦN 4: ADVERSARIAL & COMPLEX SEARCH
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("--- PHẦN 4: Tìm kiếm Đối kháng & Môi trường Phức tạp ---")
    print("="*70)
    
    # Chạy thử nghiệm AND-OR Search
    print("\n[+] Đang chạy: AND-OR Search (Sinh phương án dự phòng khi sự cố)...")
    and_or_searcher = AndOrSearch()
    path, cost, stats = and_or_searcher.search(graph, "CS101_N01", "")
    
    if path:
        print(f"  -> Đường đi thành công chính của lớp: {' -> '.join(path)}")
        print(f"  -> Kế hoạch dự phòng (Contingency Plan) dạng cây:")
        plan_str = json.dumps(stats.get("contingency_plan"), indent=2, ensure_ascii=False)
        print("\n".join(f"     {line}" for line in plan_str.split("\n")))
    else:
        print("  -> Thất bại: Không tìm thấy kế hoạch dự phòng khả thi.")
        
    # Chạy thử nghiệm LRTA* (Online Search)
    print("\n[+] Đang chạy: LRTA* Search (Xếp TKB trực tuyến + Sự cố phòng học động)...")
    lrta_searcher = LrtaStarSearch()
    path, cost, stats = lrta_searcher.search(graph, "", "", dynamic_changes=True)
    
    if path:
        print(f"  -> Xếp lịch thành công trực tuyến!")
        print(f"  -> Tổng điểm vi phạm cuối cùng: {cost:.1f}")
        print(f"  -> Số bước học trực tuyến: {stats.get('explored_nodes', 0)}")
        print(f"  -> Số lượng trạng thái đã học Heuristic: {stats.get('learned_heuristics_count', 0)}")
    else:
        print(f"  -> Thất bại: {stats.get('error', 'Lỗi không xác định')}")

    # Chạy thử nghiệm Đối kháng Minimax và Alpha-Beta
    adversarial_searchers = {
        "Minimax Search": MinimaxSearch(),
        "Alpha-Beta Search": AlphaBetaSearch()
    }
    
    adversarial_section = "CS101_N01"
    
    for name, searcher in adversarial_searchers.items():
        print(f"\n[+] Đang chạy: {name} cho Lớp HP '{adversarial_section}'...")
        path, cost, stats = searcher.search(graph, adversarial_section, "", depth=2)
        
        if path:
            print(f"  -> Phương án xếp ca/phòng an toàn nhất đề xuất: {stats.get('next_step')}")
            print(f"  -> Điểm đánh giá TKB an toàn nhất: {stats.get('best_value'):.2f}")
            print(f"  -> Số trạng thái mô phỏng sự cố đã duyệt: {stats.get('explored_nodes', 0)}")
            print(f"  -> Thời gian thực thi: {stats.get('execution_time', 0.0) * 1000:.2f} ms")
        else:
            print("  -> Thất bại.")

    # ----------------------------------------------------
    # PHẦN 5: TRỰC QUAN HÓA KẾT QUẢ ĐẠT ĐƯỢC
    # ----------------------------------------------------
    if best_timetable:
        print("\n" + "="*70)
        print("--- THỜI KHÓA BIỂU ĐỒNG BỘ CHI TIẾT (KẾT QUẢ CSP) ---")
        print("="*70)
        TimetableVisualizer.print_console(best_timetable, graph)
        print("=" * 70)

if __name__ == "__main__":
    main()
