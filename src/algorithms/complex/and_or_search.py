import time
from typing import Any, Dict, List, Optional, Tuple, Union
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

# Định nghĩa kiểu kế hoạch điều kiện (Contingency Plan)
# Một kế hoạch có thể là một chuỗi hành động hoặc một rẽ nhánh điều kiện dưới dạng dict
Plan = Union[str, Dict[str, Any], List[Any]]

class AndOrSearch(BaseSearchAlgorithm):
    """
    Thuật toán tìm kiếm AND-OR (AND-OR Search).
    Dùng trong xếp thời khóa biểu để lập kế hoạch dự phòng (contingency plans)
    khi một phòng học hoặc một ca bị sự cố.
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

        self.explored_nodes = 0
        sec_id = start
        if sec_id not in graph.sections:
            sec_id = list(graph.sections.keys())[0] if graph.sections else ""
            
        if not sec_id:
            execution_time = time.perf_counter() - start_time
            return None, 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # Tạo một timetable trống hoặc cơ sở
        timetable = Timetable()
        
        # Tìm kế hoạch hành động dự phòng cho lớp HP sec_id
        plan = self._and_or_search(graph, sec_id, timetable, set())
        
        # Đường đi mặc định (thành công lý tưởng)
        default_path = [sec_id]
        if isinstance(plan, dict) and "action" in plan:
            default_path.append(plan["action"])
        elif isinstance(plan, list) and plan:
            default_path.append(plan[0])

        execution_time = time.perf_counter() - start_time
        
        return default_path, 0.0, {
            "explored_nodes": self.explored_nodes,
            "execution_time": execution_time,
            "contingency_plan": plan
        }

    def _and_or_search(self, graph: TimetableProblem, sec_id: str, timetable: Timetable, visited_actions: set) -> Optional[Plan]:
        self.explored_nodes += 1
        
        domain = graph.get_domain_for_section(sec_id)
        if not domain:
            return None
            
        # Thử từng gán (Period, Room) -> đây là OR Node
        for period_id, room_id in domain[:5]: # Giới hạn 5 phương án
            action = f"{period_id} @ {room_id}"
            if action in visited_actions:
                continue
                
            # Kiểm tra ràng buộc cứng xem có gán được không
            is_valid, _ = ConstraintChecker.check_all_hard(
                timetable, sec_id, period_id, room_id,
                graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            if not is_valid:
                continue
                
            # Tìm phương án dự phòng khác cùng ca nhưng khác phòng (phòng dự phòng - AND Node)
            backup_options = [(p, r) for p, r in domain if p == period_id and r != room_id]
            
            plan_backup = None
            chosen_backup = None
            for p_b, r_b in backup_options:
                is_b_valid, _ = ConstraintChecker.check_all_hard(
                    timetable, sec_id, p_b, r_b,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_b_valid:
                    plan_backup = f"Divert to {p_b} @ {r_b}"
                    chosen_backup = f"{p_b} @ {r_b}"
                    break
                    
            if plan_backup:
                return {
                    "action": f"Assign to {action}",
                    "if_success": "Successfully scheduled",
                    "if_room_failed_divert_to": chosen_backup,
                    "backup_plan": plan_backup
                }
            else:
                return [f"Assign to {action}"]
                
        return None
