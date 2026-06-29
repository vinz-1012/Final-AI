# ── CÁCH HOẠT ĐỘNG ──
# 1. Khởi tạo: Gán ngẫu nhiên giá trị cho TẤT CẢ biến (complete assignment).
# 2. Lặp (tối đa max_steps):
#    a. Kiểm tra: Nếu không còn xung đột → THÀNH CÔNG.
#    b. Chọn ngẫu nhiên 1 biến đang bị xung đột (conflicted variable).
#    c. Thử gán biến đó sang TẤT CẢ giá trị trong domain.
#    d. Chọn giá trị gây ra ÍT XUNG ĐỘT NHẤT (min-conflicts heuristic).
#    e. Gán giá trị mới cho biến đó.
# 3. Nếu hết max_steps mà vẫn còn xung đột → thất bại.
# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# - Biến bị xung đột = lớp HP vi phạm ràng buộc cứng:
#   + HC1: GV dạy 2 lớp cùng ca (trùng GV).
#   + HC2: 2 lớp cùng phòng cùng ca (trùng phòng).
#   + HC3: Phòng không đủ chỗ hoặc sai loại.
#   + HC4: GV không khả dụng trong ca đó.
# - Min-Conflicts heuristic: Với biến bị xung đột, thử tất cả giá trị trong domain
#   và chọn giá trị gây tổng vi phạm cứng nhỏ nhất.

import time
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class CSPMinConflictsSearch(BaseSearchAlgorithm):
    """
    Min-Conflicts — Cực tiểu hóa mâu thuẫn cho bài toán CSP xếp TKB.

    Nguyên lý hoạt động:
      1. Khởi tạo TKB đầy đủ ngẫu nhiên (mỗi lớp HP đều có ca & phòng).
      2. Lặp: Chọn ngẫu nhiên 1 lớp đang bị xung đột → thử tất cả giá trị
         trong domain → chọn giá trị gây ít xung đột nhất → gán lại.
      3. Dừng khi hết xung đột (thành công) hoặc hết bước lặp (thất bại).

    Ưu điểm: Rất nhanh cho bài toán CSP lớn, đơn giản, dễ cài đặt.
    Nhược điểm: Không đảm bảo tìm được lời giải, phụ thuộc khởi tạo.
    """

    def _count_conflicts(self, timetable: Timetable, graph: TimetableProblem) -> int:
        """
        Đếm tổng số vi phạm ràng buộc cứng (hard violations) trong TKB.
        Đây là "hàm mục tiêu" của Min-Conflicts → mục tiêu: đưa về 0.
        """
        return ConstraintChecker.count_hard_violations(
            timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
        )

    def _get_conflicted_variables(
        self, 
        timetable: Timetable, 
        graph: TimetableProblem
    ) -> List[str]:
        """
        Tìm tất cả các biến (lớp HP) đang vi phạm ít nhất 1 ràng buộc cứng.

        Kiểm tra 4 loại ràng buộc cứng:
          - HC1: GV trùng ca (2 lớp cùng GV, cùng period).
          - HC2: Phòng trùng ca (2 lớp cùng phòng, cùng period).
          - HC3: Sức chứa phòng < số SV, hoặc loại phòng không phù hợp.
          - HC4: GV không khả dụng trong ca đó.

        Returns:
            List các section_id đang bị xung đột.
        """
        conflicted = set()
        entries = list(timetable.entries.values())

        # Kiểm tra HC1 (GV trùng ca) và HC2 (phòng trùng ca)
        # So sánh từng cặp entry để tìm xung đột
        for i, e1 in enumerate(entries):
            for e2 in entries[i+1:]:
                if e1.period_id == e2.period_id:  # Cùng ca học
                    # HC2: Trùng phòng → 2 lớp không thể ở cùng phòng cùng lúc
                    if e1.room_id == e2.room_id:
                        conflicted.add(e1.section_id)
                        conflicted.add(e2.section_id)
                    # HC1: Cùng GV → GV không thể dạy 2 lớp cùng lúc
                    s1 = graph.sections.get(e1.section_id)
                    s2 = graph.sections.get(e2.section_id)
                    if s1 and s2 and s1.lecturer_id == s2.lecturer_id:
                        conflicted.add(e1.section_id)
                        conflicted.add(e2.section_id)

        # Kiểm tra HC3 (sức chứa, loại phòng) và HC4 (GV khả dụng)
        for sid, e in timetable.entries.items():
            sec = graph.sections.get(sid)
            if not sec:
                continue
            room = graph.rooms.get(e.room_id)
            course = graph.courses.get(sec.course_id)
            lec = graph.lecturers.get(sec.lecturer_id)
            if room and course:
                # HC3: Phòng không đủ chỗ hoặc sai loại
                if room.capacity < sec.student_count or room.room_type != course.room_type:
                    conflicted.add(sid)
            if lec and not lec.is_available(e.period_id):
                # HC4: GV không khả dụng trong ca này
                conflicted.add(sid)
        return list(conflicted)

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện Min-Conflicts trên bài toán CSP xếp TKB.

        Args:
            graph: Đối tượng TimetableProblem.
            **kwargs: max_steps (int) — số bước sửa tối đa (mặc định 1000).
        """
        start_time = time.perf_counter()
        
        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}
            
        variables = list(graph.sections.keys())
        max_steps = kwargs.get("max_steps", 1000)
        
        # ══════════════════════════════════════════════════════════════
        # BƯỚC 1: KHỞI TẠO LỜI GIẢI ĐẦY ĐỦ NGẪU NHIÊN
        # ══════════════════════════════════════════════════════════════
        # Gán ngẫu nhiên MỖI lớp HP vào 1 cặp (ca, phòng) trong domain.
        # TKB này có thể (và thường) vi phạm nhiều ràng buộc cứng.
        timetable = Timetable()
        for sec_id in variables:
            domain = graph.get_domain_for_section(sec_id)
            if domain:
                p_id, r_id = random.choice(domain)
                lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                timetable.assign(sec_id, p_id, r_id, lec_id)
        
        explored_count = 0
        success = False
        
        # ══════════════════════════════════════════════════════════════
        # BƯỚC 2: VÒNG LẶP SỬA XUNG ĐỘT (MIN-CONFLICTS LOOP)
        # ══════════════════════════════════════════════════════════════
        for step in range(max_steps):
            explored_count += 1
            
            # ── Kiểm tra: Hết xung đột chưa? ──
            total_conflict = self._count_conflicts(timetable, graph)
            if total_conflict == 0:
                success = True  # TKB thỏa mãn tất cả ràng buộc cứng!
                break
                
            # ── Chọn ngẫu nhiên 1 biến đang bị xung đột ──
            # Random selection (thay vì deterministic) giúp tránh bị kẹt trong chu kỳ.
            conflicted_vars = self._get_conflicted_variables(timetable, graph)
            if not conflicted_vars:
                break
            var = random.choice(conflicted_vars)
            
            # ── Lấy domain của biến đó ──
            domain = graph.get_domain_for_section(var)
            if not domain:
                continue
                
            # ══════════════════════════════════════════════════════════
            # MIN-CONFLICTS HEURISTIC:
            # Thử gán biến vào TẤT CẢ giá trị trong domain,
            # chọn giá trị gây ra TỔNG XUNG ĐỘT NHỎ NHẤT.
            # ══════════════════════════════════════════════════════════
            current_val = timetable.get_entry(var)
            best_val = (current_val.period_id, current_val.room_id) if current_val else domain[0]
            min_c = total_conflict
            
            for p_id, r_id in domain:
                # Bỏ qua giá trị hiện tại (không tự gán lại chính mình)
                if current_val and p_id == current_val.period_id and r_id == current_val.room_id:
                    continue
                
                # Gán thử (trial assignment) và đếm xung đột
                lec_id = graph.sections[var].lecturer_id if var in graph.sections else None
                timetable.assign(var, p_id, r_id, lec_id)
                c = self._count_conflicts(timetable, graph)
                
                if c < min_c:
                    # Giá trị mới gây ít xung đột hơn → cập nhật best
                    min_c = c
                    best_val = (p_id, r_id)
                elif c == min_c and random.random() < 0.5:
                    # Tie-breaking ngẫu nhiên khi xung đột bằng nhau
                    # Giúp tránh bị kẹt trong chu kỳ lặp
                    best_val = (p_id, r_id)
                    
            # ── Gán giá trị tốt nhất cho biến ──
            best_lec_id = graph.sections[var].lecturer_id if var in graph.sections else None
            timetable.assign(var, best_val[0], best_val[1], best_lec_id)

        execution_time = time.perf_counter() - start_time
        
        # ══════════════════════════════════════════════════════════════
        # ĐÁNH GIÁ KẾT QUẢ
        # ══════════════════════════════════════════════════════════════
        final_conflict = self._count_conflicts(timetable, graph)
        soft_violations = ConstraintChecker.count_soft_violations(
            timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
        )
        cost = float(final_conflict + soft_violations["total"])
        
        return (
            # Chỉ trả về path nếu hết xung đột (thành công)
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
