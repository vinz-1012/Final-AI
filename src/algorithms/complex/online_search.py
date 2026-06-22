import time
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker

class LrtaStarSearch(BaseSearchAlgorithm):
    """
    Thuật toán tìm kiếm trực tuyến LRTA* (Learning Real-Time A*).
    Đại diện cho nhóm Online Search trong môi trường thời khóa biểu biến động.
    """

    def __init__(self):
        # Bảng lưu trữ heuristic đã học: {state_hash: H_value}
        self.H: Dict[int, float] = {}

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

        current_state = TimetableState("", set(), Timetable(), 0.0)
        steps = 0
        max_steps = kwargs.get("max_steps", 200)
        dynamic_changes = kwargs.get("dynamic_changes", True)
        
        while not current_state.is_complete(len(sections_list)) and steps < max_steps:
            steps += 1
            curr_hash = hash(current_state)
            if curr_hash not in self.H:
                self.H[curr_hash] = self._get_heuristic(current_state, graph)
                
            next_sec_idx = len(current_state.assigned_sections)
            next_sec = sections_list[next_sec_idx]
            
            domain = graph.get_domain_for_section(next_sec)
            valid_successors = []
            
            for period_id, room_id in domain:
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    # Tạo child state
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
                    valid_successors.append((period_id, room_id, child_state))
                    
            if not valid_successors:
                break
                
            # Tìm successor tối thiểu hóa f(s') = c(s, s') + H(s')
            best_successor = None
            min_f = float('inf')
            
            for p_id, r_id, child in valid_successors:
                child_hash = hash(child)
                if child_hash not in self.H:
                    self.H[child_hash] = self._get_heuristic(child, graph)
                cost = child.violation_score - current_state.violation_score
                f_val = cost + self.H[child_hash]
                
                if f_val < min_f:
                    min_f = f_val
                    best_successor = child
                    
            # Cập nhật heuristic của trạng thái hiện tại
            self.H[curr_hash] = min_f
            
            # Di chuyển sang trạng thái tốt nhất
            if best_successor:
                current_state = best_successor
            else:
                break
                
            # Giả lập thay đổi trực tuyến (room đột ngột đóng cửa với xác suất 10%)
            if dynamic_changes and random.random() < 0.1:
                rooms_list = list(graph.rooms.values())
                if rooms_list:
                    r = random.choice(rooms_list)
                    r.is_available = not r.is_available

        execution_time = time.perf_counter() - start_time
        success = current_state.is_complete(len(sections_list))
        
        # Đảm bảo khôi phục trạng thái phòng
        for room in graph.rooms.values():
            room.is_available = True
            
        if success:
            hard_violations = ConstraintChecker.count_hard_violations(
                current_state.timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            soft_violations = ConstraintChecker.count_soft_violations(
                current_state.timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
            )
            cost = float(hard_violations + soft_violations["total"])
            return (
                list(current_state.timetable.entries.keys()),
                cost,
                {
                    "explored_nodes": steps,
                    "execution_time": execution_time,
                    "timetable": current_state.timetable,
                    "hard_violations": hard_violations,
                    "soft_violations": soft_violations,
                    "fitness": current_state.fitness,
                    "learned_heuristics_count": len(self.H)
                }
            )
        else:
            return None, 0.0, {
                "explored_nodes": steps,
                "execution_time": execution_time,
                "learned_heuristics_count": len(self.H),
                "error": "Không tìm thấy thời khóa biểu trực tuyến do kẹt hoặc thay đổi môi trường."
            }
