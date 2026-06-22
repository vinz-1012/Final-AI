import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class CSPBacktrackingSearch(BaseSearchAlgorithm):
    """
    Thuật toán tìm kiếm quay lui (Backtracking Search) giải bài toán CSP.
    Bài toán: Xếp thời khóa biểu (các lớp học phần vào ca và phòng).
    Ràng buộc: Thỏa mãn tất cả ràng buộc cứng (ConstraintChecker.check_all_hard).
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

        # Variables: list of section IDs
        # MRV Heuristic: sort sections by domain size (smaller domain first)
        variables = list(graph.sections.keys())
        variables.sort(key=lambda s: len(graph.get_domain_for_section(s)))
        
        timetable = Timetable()
        self.explored_states = 0
        self.max_explored = kwargs.get("max_explored", 2000)
        # Chạy backtracking
        success = self._backtrack(timetable, variables, 0, graph)
        
        execution_time = time.perf_counter() - start_time
        
        hard_violations = ConstraintChecker.count_hard_violations(
            timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
        )
        soft_violations = ConstraintChecker.count_soft_violations(
            timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
        )
        cost = float(hard_violations + soft_violations["total"])
        
        # Nếu không xếp được hoàn thành do chạm giới hạn duyệt
        if not success and self.explored_states >= self.max_explored:
            error_msg = f"Đạt giới hạn duyệt tối đa ({self.max_explored} trạng thái) mà chưa tìm được lời giải."
        else:
            error_msg = "Không tìm thấy lời giải hợp lệ hoàn chỉnh."

        return (
            list(timetable.entries.keys()) if success else None,
            cost,
            {
                "explored_nodes": self.explored_states,
                "execution_time": execution_time,
                "timetable": timetable,
                "hard_violations": hard_violations,
                "soft_violations": soft_violations,
                "error": None if success else error_msg,
                "fitness": ConstraintChecker.compute_fitness(
                    timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
                )
            }
        )

    def _backtrack(
        self, 
        timetable: Timetable, 
        variables: List[str], 
        var_idx: int, 
        graph: TimetableProblem
    ) -> bool:
        self.explored_states += 1
        # Tránh lặp vô hạn hoặc chạy quá lâu
        if self.explored_states >= self.max_explored:
            return False
        
        if var_idx == len(variables):
            return True
            
        sec_id = variables[var_idx]
        domain = graph.get_domain_for_section(sec_id)
        
        # LCV Heuristic: Sắp xếp các giá trị trong domain theo sức chứa phòng học tăng dần
        domain.sort(key=lambda item: graph.rooms[item[1]].capacity)
        
        for period_id, room_id in domain:
            # Kiểm tra ràng buộc cứng
            is_valid, _ = ConstraintChecker.check_all_hard(
                timetable, sec_id, period_id, room_id,
                graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            
            if is_valid:
                lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                timetable.assign(sec_id, period_id, room_id, lec_id)
                
                if self._backtrack(timetable, variables, var_idx + 1, graph):
                    return True
                    
                timetable.unassign(sec_id)
                
        return False
