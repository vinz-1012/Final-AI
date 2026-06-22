import time
import heapq
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker

class AStarSearch(BaseSearchAlgorithm):
    """
    Thuật toán tìm kiếm A* (A Star Search).
    A* đánh giá các trạng thái TKB bằng hàm f(n) = g(n) + h(n), trong đó:
    - g(n): chi phí tích lũy (vi phạm ràng buộc mềm của trạng thái hiện tại).
    - h(n): ước lượng vi phạm (xung đột) còn lại của các lớp chưa xếp lịch.
    """

    def _get_heuristic(self, state: TimetableState, graph: TimetableProblem) -> float:
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
        start_time = time.perf_counter()

        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        sections_list = list(graph.sections.keys())
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # Trạng thái ban đầu: TKB rỗng
        start_state = TimetableState("", set(), Timetable(), 0.0)
        h_start = self._get_heuristic(start_state, graph)
        
        # Priority Queue: (f_val, counter, state)
        counter = 0
        pq: List[Tuple[float, int, TimetableState]] = [(h_start, counter, start_state)]
        
        explored_count = 0
        success_state: Optional[TimetableState] = None
        max_explored = kwargs.get("max_explored", 2000)

        while pq:
            f_val, _, current_state = heapq.heappop(pq)
            explored_count += 1

            if current_state.is_complete(len(sections_list)):
                success_state = current_state
                break

            if explored_count >= max_explored:
                break

            # Chọn lớp học phần tiếp theo để xếp
            next_sec_idx = len(current_state.assigned_sections)
            next_sec = sections_list[next_sec_idx]

            domain = graph.get_domain_for_section(next_sec)
            for period_id, room_id in domain:
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    # Tính vi phạm mềm cộng thêm cho g(n)
                    temp_timetable = current_state.timetable.copy()
                    lec_id = graph.sections[next_sec].lecturer_id if next_sec in graph.sections else None
                    temp_timetable.assign(next_sec, period_id, room_id, lec_id)
                    soft_violations = ConstraintChecker.count_soft_violations(
                        temp_timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
                    )
                    g_val = soft_violations["total"]
                    
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id, extra_violations=g_val - current_state.violation_score,
                        lecturer_id=lec_id
                    )
                    h_val = self._get_heuristic(child_state, graph)
                    f_val = g_val + h_val
                    counter += 1
                    heapq.heappush(pq, (f_val, counter, child_state))

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
