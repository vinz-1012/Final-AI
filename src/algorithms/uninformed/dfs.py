import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker

class DepthFirstSearch(BaseSearchAlgorithm):
    """
    Thuật toán tìm kiếm theo chiều sâu (Depth-First Search - DFS).
    DFS xây dựng TKB bằng cách xếp từng lớp học phần một theo chiều sâu.
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

        sections_list = list(graph.sections.keys())
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # Trạng thái ban đầu: TKB rỗng
        start_state = TimetableState("", set(), Timetable(), 0.0)

        # Ngăn xếp chứa các trạng thái TimetableState
        stack: List[TimetableState] = [start_state]
        explored_count = 0
        success_state: Optional[TimetableState] = None

        # Giới hạn số trạng thái duyệt để tránh quá tải
        max_explored = kwargs.get("max_explored", 2000)

        while stack:
            current_state = stack.pop()
            explored_count += 1

            # Kiểm tra xem đã xếp xong tất cả các lớp HP chưa
            if current_state.is_complete(len(sections_list)):
                success_state = current_state
                break

            if explored_count >= max_explored:
                break

            # Chọn lớp học phần tiếp theo để xếp
            next_sec_idx = len(current_state.assigned_sections)
            next_sec = sections_list[next_sec_idx]

            # Duyệt các ca và phòng khả thi trong domain
            domain = graph.get_domain_for_section(next_sec)
            for period_id, room_id in reversed(domain):
                # Kiểm tra ràng buộc cứng
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id,
                        lecturer_id=graph.sections[next_sec].lecturer_id if next_sec in graph.sections else None
                    )
                    stack.append(child_state)

        execution_time = time.perf_counter() - start_time

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
