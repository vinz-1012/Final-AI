import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker


class BreadthFirstSearch(BaseSearchAlgorithm):
    """
    BFS xếp TKB từng lớp HP theo chiều rộng.
    Fix: dùng deque, tăng max_explored, thêm visited set,
         và pruning sớm khi domain rỗng.
    """

    def search(
        self,
        graph: Graph,
        start: str,
        goal: str,
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        start_time = time.perf_counter()

        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        # Sắp xếp các lớp học phần theo số lượng ca/phòng khả thi tăng dần (MRV Heuristic)
        # giúp hướng tìm kiếm vào các biến bị ràng buộc nhiều trước, giảm tối đa số nút phải duyệt.
        sections_list = sorted(list(graph.sections.keys()), key=lambda s: len(graph.get_domain_for_section(s)))
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # Trạng thái ban đầu
        start_state = TimetableState("", set(), Timetable(), 0.0)

        # ✅ Fix 1: dùng deque thay list để pop(0) O(1) thay vì O(n)
        queue: deque[TimetableState] = deque([start_state])

        # ✅ Fix 2: visited set tránh duyệt lại trạng thái giống nhau
        visited: set = set()

        explored_count = 0
        success_state: Optional[TimetableState] = None

        # ✅ Fix 3: tăng max_explored lên đủ lớn (hoặc dùng -1 = không giới hạn)
        max_explored = kwargs.get("max_explored", 2000)

        while queue:
            current_state = queue.popleft()
            explored_count += 1

            # Kiểm tra hoàn thành
            if current_state.is_complete(len(sections_list)):
                success_state = current_state
                break

            if 0 < max_explored <= explored_count:
                break

            # ✅ Fix 4: tạo state key để kiểm tra visited
            state_key = self._state_key(current_state)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Chọn lớp HP tiếp theo
            next_sec_idx = len(current_state.assigned_sections)
            if next_sec_idx >= len(sections_list):
                continue
            next_sec = sections_list[next_sec_idx]

            domain = graph.get_domain_for_section(next_sec)

            # ✅ Fix 5: pruning — nếu domain rỗng, nhánh này chắc chắn thất bại
            if not domain:
                continue

            for period_id, room_id in domain:
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id,
                        lecturer_id=graph.sections[next_sec].lecturer_id
                        if next_sec in graph.sections else None
                    )
                    queue.append(child_state)

        execution_time = time.perf_counter() - start_time

        if success_state:
            hard_violations = ConstraintChecker.count_hard_violations(
                success_state.timetable,
                graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            soft_violations = ConstraintChecker.count_soft_violations(
                success_state.timetable,
                graph.sections, graph.rooms, graph.periods, graph.lecturers
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
                    "fitness": success_state.fitness,
                }
            )
        else:
            return None, 0.0, {
                "explored_nodes": explored_count,
                "execution_time": execution_time,
                "error": "Không tìm thấy lời giải hợp lệ hoàn chỉnh."
            }

    # ✅ Fix 6: hàm tạo key duy nhất cho mỗi trạng thái
    @staticmethod
    def _state_key(state: TimetableState) -> frozenset:
        """
        Key = tập các (section_id, period_id, room_id) đã được xếp.
        frozenset giúp hash nhanh và không phụ thuộc thứ tự.
        """
        entries = state.timetable.entries  # dict: section_id -> Entry
        return frozenset(
            (sec_id, entry.period_id, entry.room_id)
            for sec_id, entry in entries.items()
        )