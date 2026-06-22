import time
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class SimulatedAnnealingSearch(BaseSearchAlgorithm):
    """
    Thuật toán luyện kim mô phỏng (Simulated Annealing).
    Áp dụng cho bài toán tối ưu thời khóa biểu (Course Timetabling).
    Cho phép chấp nhận phương án gán xấu hơn với xác suất giảm dần theo nhiệt độ để thoát khỏi cực tiểu cục bộ.
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
            
        temp = kwargs.get("initial_temperature", 100.0)
        cooling_rate = kwargs.get("cooling_rate", 0.95)
        min_temp = kwargs.get("min_temperature", 0.01)
        iterations_per_temp = kwargs.get("iterations_per_temp", 20)
        
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
        
        best_timetable = current_timetable.copy()
        best_fitness = current_fitness
        
        explored_count = 0
        
        # Vòng lặp luyện kim
        while temp > min_temp:
            for _ in range(iterations_per_temp):
                explored_count += 1
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
                        
                neighbor_fitness = ConstraintChecker.compute_fitness(
                    neighbor, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
                )
                
                # delta = current_fitness - neighbor_fitness (vì ta muốn tối đa hóa fitness)
                delta = current_fitness - neighbor_fitness
                
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / temp)):
                    current_timetable = neighbor
                    current_fitness = neighbor_fitness
                    
                    if current_fitness > best_fitness:
                        best_timetable = current_timetable.copy()
                        best_fitness = current_fitness
            
            # Hạ nhiệt độ
            temp *= cooling_rate

        execution_time = time.perf_counter() - start_time
        
        hard_violations = ConstraintChecker.count_hard_violations(
            best_timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
        )
        soft_violations = ConstraintChecker.count_soft_violations(
            best_timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
        )
        cost = float(hard_violations + soft_violations["total"])
        
        return (
            list(best_timetable.entries.keys()),
            cost,
            {
                "explored_nodes": explored_count,
                "execution_time": execution_time,
                "timetable": best_timetable,
                "hard_violations": hard_violations,
                "soft_violations": soft_violations,
                "fitness": best_fitness
            }
        )
