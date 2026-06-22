import time
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class HillClimbingSearch(BaseSearchAlgorithm):
    """
    Thuật toán tìm kiếm leo đồi (Hill Climbing).
    Áp dụng cho bài toán tối ưu hóa thời khóa biểu (Course Timetabling).
    Mục tiêu: Tối đa hóa hàm fitness (cực tiểu hóa vi phạm ràng buộc cứng và mềm).
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

        max_steps = kwargs.get("max_steps", 1000)
        variables = list(graph.sections.keys())
        
        # 1. Khởi tạo một TKB ngẫu nhiên
        current_timetable = Timetable()
        for sec_id in variables:
            domain = graph.get_domain_for_section(sec_id)
            if domain:
                p_id, r_id = random.choice(domain)
                lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                current_timetable.assign(sec_id, p_id, r_id, lec_id)
                
        current_fitness = ConstraintChecker.compute_fitness(
            current_timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
        )
        
        explored_count = 0
        
        for step in range(max_steps):
            explored_count += 1
            improved = False
            
            # Sinh ra 30 lân cận ngẫu nhiên và chọn lân cận tốt nhất
            best_neighbor_timetable = None
            best_neighbor_fitness = current_fitness
            
            for _ in range(30):
                neighbor = current_timetable.copy()
                
                # Biến đổi ngẫu nhiên: 50% đổi ngẫu nhiên 1 môn, 50% đổi chỗ 2 môn
                if random.random() < 0.5 or len(variables) < 2:
                    sec_id = random.choice(variables)
                    domain = graph.get_domain_for_section(sec_id)
                    if domain:
                        p_id, r_id = random.choice(domain)
                        lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                        neighbor.assign(sec_id, p_id, r_id, lec_id)
                else:
                    sec1, sec2 = random.sample(variables, 2)
                    e1 = neighbor.get_entry(sec1)
                    e2 = neighbor.get_entry(sec2)
                    if e1 and e2:
                        lec1 = graph.sections[sec1].lecturer_id if sec1 in graph.sections else None
                        lec2 = graph.sections[sec2].lecturer_id if sec2 in graph.sections else None
                        neighbor.assign(sec1, e2.period_id, e2.room_id, lec1)
                        neighbor.assign(sec2, e1.period_id, e1.room_id, lec2)
                
                fit = ConstraintChecker.compute_fitness(
                    neighbor, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
                )
                
                if fit > best_neighbor_fitness:
                    best_neighbor_fitness = fit
                    best_neighbor_timetable = neighbor
                    
            if best_neighbor_timetable is not None:
                current_timetable = best_neighbor_timetable
                current_fitness = best_neighbor_fitness
                improved = True
                
            if not improved:
                # Đã đạt cực trị cục bộ
                break

        execution_time = time.perf_counter() - start_time
        
        hard_violations = ConstraintChecker.count_hard_violations(
            current_timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
        )
        soft_violations = ConstraintChecker.count_soft_violations(
            current_timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
        )
        cost = float(hard_violations + soft_violations["total"])
        
        return (
            list(current_timetable.entries.keys()),
            cost,
            {
                "explored_nodes": explored_count,
                "execution_time": execution_time,
                "timetable": current_timetable,
                "hard_violations": hard_violations,
                "soft_violations": soft_violations,
                "fitness": current_fitness
            }
        )
