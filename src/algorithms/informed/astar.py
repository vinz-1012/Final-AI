# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
#   - g(n) = Tổng vi phạm ràng buộc mềm (soft violations) tích lũy của TKB hiện tại.
#   - h(n) = Ước lượng số xung đột còn lại của các lớp HP chưa xếp lịch.
#     (Tính bằng count_remaining_conflicts: đếm số lớp chưa xếp mà có tiềm năng
#      xung đột với các lớp đã xếp — ví dụ: cùng GV, cùng sinh viên, ...)
#   - f(n) = g(n) + h(n) → TKB có tổng vi phạm + ước lượng xung đột thấp nhất
#     sẽ được xét trước.

import time
import heapq
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker

class AStarSearch(BaseSearchAlgorithm):
    """
    A* Search — Tìm kiếm A* cho bài toán xếp Thời Khóa Biểu.

    Nguyên lý hoạt động:
      1. Khởi tạo priority queue với trạng thái gốc, f(root) = 0 + h(root).
      2. Lặp: Pop trạng thái có f(n) nhỏ nhất → kiểm tra hoàn thành →
         nếu chưa, sinh các trạng thái con, tính f = g + h cho mỗi con,
         rồi push vào priority queue.
      3. Dừng khi tìm được TKB đầy đủ hoặc hết trạng thái.

    Ưu điểm: Tối ưu (nếu h admissible), thường duyệt ít nút hơn BFS/DFS.
    Nhược điểm: Tốn bộ nhớ, phụ thuộc chất lượng hàm heuristic.
    """

    def _get_heuristic(self, state: TimetableState, graph: TimetableProblem) -> float:
        """
        Hàm heuristic h(n): Ước lượng chi phí còn lại để hoàn thành TKB.

        Cách tính: Với mỗi lớp HP chưa được xếp lịch, đếm số xung đột tiềm năng
        với các lớp đã xếp (cùng GV, cùng nhóm sinh viên, ...).
        Tổng xung đột này là ước lượng dưới (admissible) của vi phạm còn lại.

        Args:
            state: Trạng thái TKB hiện tại.
            graph: Bài toán TKB chứa thông tin sections, constraints.

        Returns:
            h_val (float): Giá trị heuristic ước lượng.
        """
        # Lấy danh sách các lớp HP chưa được xếp lịch
        unassigned = [s for s in graph.sections if s not in state.assigned_sections]
        h_val = 0.0
        for s in unassigned:
            # Đếm xung đột tiềm năng của lớp s với các lớp đã xếp
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
        Thực hiện tìm kiếm A* trên không gian trạng thái TKB.

        Args:
            graph: Đối tượng TimetableProblem chứa dữ liệu bài toán.
            start, goal: Tham số interface.
            **kwargs: max_explored (int) — giới hạn số trạng thái tối đa.

        Returns:
            Tuple (path, cost, info).
        """
        start_time = time.perf_counter()

        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        # ── MRV Heuristic ──
        # Sắp xếp lớp HP theo domain tăng dần → xếp lớp bị ràng buộc nhiều trước.
        sections_list = sorted(list(graph.sections.keys()), key=lambda s: len(graph.get_domain_for_section(s)))
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ── Khởi tạo trạng thái gốc ──
        start_state = TimetableState("", set(), Timetable(), 0.0)
        h_start = self._get_heuristic(start_state, graph)
        
        # ── Priority Queue (Min-Heap) ──
        # Mỗi phần tử: (f_val, counter, state)
        #   - f_val: giá trị f(n) = g(n) + h(n) → ưu tiên nút có f nhỏ nhất.
        #   - counter: bộ đếm tie-breaking khi f bằng nhau (tránh so sánh state).
        counter = 0
        pq: List[Tuple[float, int, TimetableState]] = [(h_start, counter, start_state)]
        
        explored_count = 0
        success_state: Optional[TimetableState] = None
        max_explored = kwargs.get("max_explored", 2000)

        # ══════════════════════════════════════════════════════════════
        # VÒNG LẶP CHÍNH CỦA A*
        # ══════════════════════════════════════════════════════════════
        while pq:
            # Pop trạng thái có f(n) nhỏ nhất từ min-heap
            f_val, _, current_state = heapq.heappop(pq)
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

            # ── Mở rộng nút: thử tất cả giá trị hợp lệ trong domain ──
            domain = graph.get_domain_for_section(next_sec)
            for period_id, room_id in domain:
                # Kiểm tra ràng buộc cứng
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    # ── Tính g(n') = tổng vi phạm mềm sau khi gán ──
                    # Tối ưu: Gán thử trực tiếp trên timetable hiện tại rồi gỡ,
                    # thay vì copy toàn bộ → tiết kiệm bộ nhớ và thời gian.
                    lec_id = graph.sections[next_sec].lecturer_id if next_sec in graph.sections else None
                    current_state.timetable.assign(next_sec, period_id, room_id, lec_id)
                    soft_violations = ConstraintChecker.count_soft_violations(
                        current_state.timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
                    )
                    g_val = soft_violations["total"]  # g(n') = tổng vi phạm mềm thực tế
                    current_state.timetable.unassign(next_sec)  # Khôi phục trạng thái
                    
                    # Tạo trạng thái con chính thức
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id, extra_violations=g_val - current_state.violation_score,
                        lecturer_id=lec_id
                    )
                    # ── Tính h(n') và f(n') = g(n') + h(n') ──
                    h_val = self._get_heuristic(child_state, graph)
                    f_val = g_val + h_val
                    counter += 1
                    # Push vào priority queue với f(n') làm khóa sắp xếp
                    heapq.heappush(pq, (f_val, counter, child_state))

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
