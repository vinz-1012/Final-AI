# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# - Trạng thái gốc: TKB rỗng.
# - Mỗi bước: Chọn 1 lớp HP chưa xếp → thử gán vào 1 cặp (ca, phòng) hợp lệ.
# - DFS sẽ đi sâu theo một nhánh (xếp hết lớp HP) trước khi thử nhánh khác.
# - Ưu điểm: Tiết kiệm bộ nhớ hơn BFS rất nhiều (chỉ lưu 1 đường đi).
# - Nhược điểm: Có thể bị kẹt trong nhánh sai rất lâu, không đảm bảo tối ưu.

import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker

class DepthFirstSearch(BaseSearchAlgorithm):
    """
    DFS — Tìm kiếm theo chiều sâu cho bài toán xếp Thời Khóa Biểu.

    Nguyên lý hoạt động:
      1. Khởi tạo ngăn xếp (stack) với trạng thái gốc (TKB rỗng).
      2. Lặp: Lấy trạng thái trên đỉnh ngăn xếp (pop) → kiểm tra hoàn thành →
         nếu chưa, sinh các trạng thái con hợp lệ và đẩy vào đỉnh ngăn xếp.
      3. Dừng khi tìm được TKB đầy đủ hoặc hết trạng thái.

    Ưu điểm: Tiết kiệm bộ nhớ O(b*d), nhanh tìm được lời giải nếu cây nông.
    Nhược điểm: Không đảm bảo tối ưu, có thể kẹt ở nhánh sâu vô ích.
    """

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện tìm kiếm DFS trên không gian trạng thái TKB.

        Args:
            graph: Đối tượng TimetableProblem chứa dữ liệu bài toán.
            start, goal: Tham số interface (không sử dụng trực tiếp trong DFS cho TKB).
            **kwargs: max_explored (int) — giới hạn số trạng thái tối đa được duyệt.

        Returns:
            Tuple (path, cost, info) — tương tự BFS.
        """
        start_time = time.perf_counter()

        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        # Danh sách lớp HP cần xếp lịch (không sắp MRV như BFS — giữ nguyên thứ tự gốc)
        sections_list = list(graph.sections.keys())
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ── Trạng thái ban đầu: TKB rỗng ──
        start_state = TimetableState("", set(), Timetable(), 0.0)

        # ── Ngăn xếp LIFO ──
        # DFS dùng stack: nút được push vào sau sẽ được pop ra trước → đi sâu trước.
        stack: List[TimetableState] = [start_state]
        explored_count = 0
        success_state: Optional[TimetableState] = None

        # Giới hạn số trạng thái duyệt để tránh quá tải bộ nhớ/thời gian
        max_explored = kwargs.get("max_explored", 2000)

        # ══════════════════════════════════════════════════════════════
        # VÒNG LẶP CHÍNH CỦA DFS
        # ══════════════════════════════════════════════════════════════
        while stack:
            # Pop trạng thái trên đỉnh ngăn xếp (LIFO → ưu tiên nhánh sâu nhất)
            current_state = stack.pop()
            explored_count += 1

            # ── Kiểm tra đích: TKB đã xếp xong tất cả lớp HP ──
            if current_state.is_complete(len(sections_list)):
                success_state = current_state
                break

            # Kiểm tra giới hạn duyệt
            if explored_count >= max_explored:
                break

            # ── Chọn lớp HP tiếp theo cần xếp lịch ──
            next_sec_idx = len(current_state.assigned_sections)
            next_sec = sections_list[next_sec_idx]

            # ── Duyệt domain theo thứ tự đảo ngược ──
            # reversed() vì stack là LIFO: phần tử cuối sẽ được pop trước,
            # nên ta reverse để phần tử đầu domain sẽ nằm ở đỉnh stack → được duyệt trước.
            domain = graph.get_domain_for_section(next_sec)
            for period_id, room_id in reversed(domain):
                # Kiểm tra ràng buộc cứng trước khi tạo trạng thái con
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    # Tạo trạng thái con mới (sao chép TKB + gán thêm lớp HP)
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id,
                        lecturer_id=graph.sections[next_sec].lecturer_id if next_sec in graph.sections else None
                    )
                    # Đẩy vào ĐỈNH ngăn xếp (đặc trưng LIFO của DFS)
                    stack.append(child_state)

        execution_time = time.perf_counter() - start_time

        # ══════════════════════════════════════════════════════════════
        # ĐÁNH GIÁ KẾT QUẢ
        # ══════════════════════════════════════════════════════════════
        if success_state:
            # Tính vi phạm cứng (hard) và mềm (soft) cho TKB tìm được
            hard_violations = ConstraintChecker.count_hard_violations(
                success_state.timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            soft_violations = ConstraintChecker.count_soft_violations(
                success_state.timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
            )
            cost = float(hard_violations + soft_violations["total"])
            return (
                list(success_state.timetable.entries.keys()),
                cost,
                {
                    "explored_nodes": explored_count,
                    "execution_time": execution_time,
                    "timetable": success_state.timetable,
                    "hard_violations": hard_violations,
                    "soft_violations": soft_violations,
                    "fitness": success_state.fitness
                }
            )
        else:
            return None, 0.0, {
                "explored_nodes": explored_count,
                "execution_time": execution_time,
                "error": "Không tìm thấy lời giải hợp lệ hoàn chỉnh."
            }
