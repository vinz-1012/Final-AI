# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
#   - h(n) = Ước lượng số xung đột còn lại của các lớp HP chưa xếp lịch.
#   - GBFS luôn mở rộng TKB có h(n) thấp nhất → ưu tiên trạng thái mà
#     các lớp chưa xếp có ít xung đột tiềm năng nhất.
#   - Không xét g(n) → bỏ qua vi phạm mềm đã tích lũy.


import time
import heapq
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker

class GreedyBestFirstSearch(BaseSearchAlgorithm):
    """
    Greedy Best-First Search — Tìm kiếm tham lam cho bài toán xếp TKB.

    Nguyên lý hoạt động:
      1. Khởi tạo priority queue với trạng thái gốc, ưu tiên theo h(n).
      2. Lặp: Pop trạng thái có h(n) nhỏ nhất → kiểm tra hoàn thành →
         nếu chưa, sinh các trạng thái con, tính h(n') cho mỗi con,
         rồi push vào priority queue.
      3. Dừng khi tìm được TKB đầy đủ hoặc hết trạng thái.

    Ưu điểm: Nhanh, ít nút duyệt hơn Uninformed Search.
    Nhược điểm: Không đảm bảo tối ưu, phụ thuộc hoàn toàn vào chất lượng h(n).
    """

    def _get_heuristic(self, state: TimetableState, graph: TimetableProblem) -> float:
        """
        Hàm heuristic h(n): Ước lượng số xung đột còn lại.
        
        Giống hệt hàm heuristic của A*, nhưng trong GBFS nó là tiêu chí
        DUY NHẤT để sắp xếp ưu tiên (không cộng thêm g(n)).
        """
        unassigned = [s for s in graph.sections if s not in state.assigned_sections]
        h_val = 0.0
        for s in unassigned:
            h_val += graph.count_remaining_conflicts(s, state.assigned_sections)
        return h_val

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện tìm kiếm Greedy Best-First trên không gian trạng thái TKB.
        """
        start_time = time.perf_counter()

        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        # ── MRV Heuristic ──
        # Sắp xếp lớp HP theo kích thước domain tăng dần (lớp khó xếp trước).
        sections_list = sorted(list(graph.sections.keys()), key=lambda s: len(graph.get_domain_for_section(s)))
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ── Khởi tạo trạng thái gốc ──
        start_state = TimetableState("", set(), Timetable(), 0.0)
        h_start = self._get_heuristic(start_state, graph)
        
        # ── Priority Queue theo h(n) ──
        # Khác với A*: ở đây chỉ dùng h(n) làm khóa sắp xếp, KHÔNG cộng g(n).
        # → Trạng thái "trông có vẻ gần đích nhất" sẽ được duyệt trước.
        counter = 0
        pq: List[Tuple[float, int, TimetableState]] = [(h_start, counter, start_state)]
        
        explored_count = 0
        success_state: Optional[TimetableState] = None
        max_explored = kwargs.get("max_explored", 2000)

        # ══════════════════════════════════════════════════════════════
        # VÒNG LẶP CHÍNH CỦA GREEDY BEST-FIRST SEARCH
        # ══════════════════════════════════════════════════════════════
        while pq:
            # Pop trạng thái có h(n) nhỏ nhất (tham lam → chỉ xét heuristic)
            h_val, _, current_state = heapq.heappop(pq)
            explored_count += 1

            # ── Kiểm tra đích ──
            if current_state.is_complete(len(sections_list)):
                success_state = current_state
                break

            if explored_count >= max_explored:
                break

            # ── Chọn lớp HP tiếp theo ──
            next_sec_idx = len(current_state.assigned_sections)
            next_sec = sections_list[next_sec_idx]

            # ── Mở rộng nút ──
            domain = graph.get_domain_for_section(next_sec)
            for period_id, room_id in domain:
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id,
                        lecturer_id=graph.sections[next_sec].lecturer_id if next_sec in graph.sections else None
                    )
                    # ── CHỈ dùng h(n') ──
                    # Đây là điểm khác biệt cốt lõi so với A*:
                    # A* dùng f = g + h, còn Greedy chỉ dùng f = h.
                    h_val = self._get_heuristic(child_state, graph)
                    counter += 1
                    heapq.heappush(pq, (h_val, counter, child_state))

        execution_time = time.perf_counter() - start_time

        # ══════════════════════════════════════════════════════════════
        # ĐÁNH GIÁ KẾT QUẢ
        # ══════════════════════════════════════════════════════════════
        if success_state:
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
