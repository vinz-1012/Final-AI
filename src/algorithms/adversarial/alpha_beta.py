import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class AlphaBetaSearch(BaseSearchAlgorithm):
    """
    Thuật toán Minimax có cắt tỉa Alpha-Beta (Alpha-Beta Pruning).
    Tối ưu hóa số lượng nút cần duyệt so với Minimax thông thường khi xếp TKB.
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
            
        depth = kwargs.get("depth", 2)
        self.explored_nodes = 0
        
        sec_id = start
        if sec_id not in graph.sections:
            sec_id = list(graph.sections.keys())[0] if graph.sections else ""
            
        if not sec_id:
            execution_time = time.perf_counter() - start_time
            return None, 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # Dựng TKB cơ sở cho các lớp khác
        timetable = Timetable()
        for s_id in graph.sections:
            if s_id != sec_id:
                domain = graph.get_domain_for_section(s_id)
                if domain:
                    lec_id = graph.sections[s_id].lecturer_id if s_id in graph.sections else None
                    timetable.assign(s_id, domain[0][0], domain[0][1], lec_id)

        domain = graph.get_domain_for_section(sec_id)
        best_val = -float('inf')
        best_move = None
        
        alpha = -float('inf')
        beta = float('inf')
        
        for period_id, room_id in domain[:15]:
            lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
            timetable.assign(sec_id, period_id, room_id, lec_id)
            
            val = self._min_value(timetable, graph, depth - 1, alpha, beta)
            if val > best_val:
                best_val = val
                best_move = (period_id, room_id)
                
            timetable.unassign(sec_id)
            
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

        execution_time = time.perf_counter() - start_time
        path = [sec_id, f"{best_move[0]}@{best_move[1]}"] if best_move else None
        
        return path, float(best_val), {
            "explored_nodes": self.explored_nodes,
            "execution_time": execution_time,
            "best_value": best_val,
            "next_step": f"{best_move[0]} @ {best_move[1]}" if best_move else None,
            "timetable": timetable
        }

    def _eval(self, timetable: Timetable, graph: TimetableProblem) -> float:
        return ConstraintChecker.compute_fitness(
            timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
        )

    def _max_value(self, timetable: Timetable, graph: TimetableProblem, depth: int, alpha: float, beta: float) -> float:
        self.explored_nodes += 1
        if depth == 0:
            return self._eval(timetable, graph)
        return self._eval(timetable, graph)

    def _min_value(self, timetable: Timetable, graph: TimetableProblem, depth: int, alpha: float, beta: float) -> float:
        self.explored_nodes += 1
        if depth == 0:
            return self._eval(timetable, graph)
            
        val = float('inf')
        for room in list(graph.rooms.values())[:5]:
            old_available = room.is_available
            room.is_available = False
            
            val = min(val, self._max_value(timetable, graph, depth - 1, alpha, beta))
            
            room.is_available = old_available
            
            if val <= alpha:
                return val
            beta = min(beta, val)
            
        return val
