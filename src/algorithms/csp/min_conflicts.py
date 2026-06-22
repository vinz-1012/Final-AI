import time
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class CSPMinConflictsSearch(BaseSearchAlgorithm):
    """
    Thuật toán cực tiểu hóa mâu thuẫn (Min-Conflicts).
    Là thuật toán tìm kiếm cục bộ giải quyết bài toán thỏa mãn ràng buộc (CSP).
    Xếp các lớp học phần vào ca và phòng học sao cho tổng mâu thuẫn (vi phạm ràng buộc cứng) bằng 0.
    """

    def _count_conflicts(self, timetable: Timetable, graph: TimetableProblem) -> int:
        return ConstraintChecker.count_hard_violations(
            timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
        )

    def _get_conflicted_variables(
        self, 
        timetable: Timetable, 
        graph: TimetableProblem
    ) -> List[str]:
        conflicted = set()
        entries = list(timetable.entries.values())
        for i, e1 in enumerate(entries):
            for e2 in entries[i+1:]:
                if e1.period_id == e2.period_id:
                    # HC2: Trùng phòng
                    if e1.room_id == e2.room_id:
                        conflicted.add(e1.section_id)
                        conflicted.add(e2.section_id)
                    # HC1: GV trùng ca
                    s1 = graph.sections.get(e1.section_id)
                    s2 = graph.sections.get(e2.section_id)
                    if s1 and s2 and s1.lecturer_id == s2.lecturer_id:
                        conflicted.add(e1.section_id)
                        conflicted.add(e2.section_id)
        for sid, e in timetable.entries.items():
            sec = graph.sections.get(sid)
            if not sec:
                continue
            room = graph.rooms.get(e.room_id)
            course = graph.courses.get(sec.course_id)
            lec = graph.lecturers.get(sec.lecturer_id)
            if room and course:
                if room.capacity < sec.student_count or room.room_type != course.room_type:
                    conflicted.add(sid)
            if lec and not lec.is_available(e.period_id):
                conflicted.add(sid)
        return list(conflicted)

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
            
        variables = list(graph.sections.keys())
        max_steps = kwargs.get("max_steps", 1000)
        
        # 1. Khởi tạo một lời giải đầy đủ ngẫu nhiên (Complete Assignment)
        timetable = Timetable()
        for sec_id in variables:
            domain = graph.get_domain_for_section(sec_id)
            if domain:
                p_id, r_id = random.choice(domain)
                lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                timetable.assign(sec_id, p_id, r_id, lec_id)
        
        explored_count = 0
        success = False
        
        for step in range(max_steps):
            explored_count += 1
            
            # Kiểm tra xem đã hết mâu thuẫn chưa
            total_conflict = self._count_conflicts(timetable, graph)
            if total_conflict == 0:
                success = True
                break
                
            # Chọn ngẫu nhiên một biến bị mâu thuẫn
            conflicted_vars = self._get_conflicted_variables(timetable, graph)
            if not conflicted_vars:
                # Không còn biến bị mâu thuẫn nào nhưng count_conflicts khác 0? (Trường hợp hiếm gặp)
                break
            var = random.choice(conflicted_vars)
            
            # Lấy domain của biến đó
            domain = graph.get_domain_for_section(var)
            if not domain:
                continue
                
            # Chọn giá trị giúp cực tiểu hóa mâu thuẫn
            current_val = timetable.get_entry(var)
            best_val = (current_val.period_id, current_val.room_id) if current_val else domain[0]
            min_c = total_conflict
            
            # Thử gán biến này vào từng giá trị khác trong domain
            for p_id, r_id in domain:
                if current_val and p_id == current_val.period_id and r_id == current_val.room_id:
                    continue
                
                # Gán thử
                lec_id = graph.sections[var].lecturer_id if var in graph.sections else None
                timetable.assign(var, p_id, r_id, lec_id)
                c = self._count_conflicts(timetable, graph)
                
                if c < min_c:
                    min_c = c
                    best_val = (p_id, r_id)
                elif c == min_c and random.random() < 0.5:
                    best_val = (p_id, r_id)
                    
            # Khôi phục gán phương án tốt nhất
            best_lec_id = graph.sections[var].lecturer_id if var in graph.sections else None
            timetable.assign(var, best_val[0], best_val[1], best_lec_id)

        execution_time = time.perf_counter() - start_time
        
        final_conflict = self._count_conflicts(timetable, graph)
        soft_violations = ConstraintChecker.count_soft_violations(
            timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
        )
        cost = float(final_conflict + soft_violations["total"])
        
        return (
            list(timetable.entries.keys()) if final_conflict == 0 else None,
            cost,
            {
                "explored_nodes": explored_count,
                "execution_time": execution_time,
                "timetable": timetable,
                "hard_violations": final_conflict,
                "soft_violations": soft_violations,
                "fitness": ConstraintChecker.compute_fitness(
                    timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
                ),
                "success": final_conflict == 0
            }
        )
