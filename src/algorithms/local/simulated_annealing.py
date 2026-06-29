
# ── LỊCH TRÌNH HẠ NHIỆT (COOLING SCHEDULE) ──
# SA sử dụng lịch trình hạ nhiệt hình học (geometric cooling):
#     T_new = T_old × cooling_rate
# Với cooling_rate = 0.95 → nhiệt độ giảm 5% mỗi vòng.
# Quá trình dừng khi T < min_temperature (mặc định 0.01).
#
# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# - Trạng thái: Một TKB đầy đủ (giống Hill Climbing).
# - Hàm mục tiêu: fitness → CỰC ĐẠI HÓA.
# - Lân cận: Đổi ca/phòng 1 lớp hoặc hoán đổi 2 lớp (giống Hill Climbing).
# - Khác biệt: SA có thể chấp nhận TKB xấu hơn → thoát cực trị cục bộ.
# - Lưu lại TKB tốt nhất từng gặp (best_timetable) → trả về khi kết thúc.
import time
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class SimulatedAnnealingSearch(BaseSearchAlgorithm):
    """
    Simulated Annealing — Luyện kim mô phỏng cho bài toán tối ưu TKB.

    Nguyên lý hoạt động:
      1. Khởi tạo TKB ngẫu nhiên, đặt nhiệt độ T = T_init (cao).
      2. Lặp (cho đến khi T < T_min):
         a. Sinh 1 lân cận ngẫu nhiên.
         b. Nếu lân cận TỐT HƠN → chấp nhận ngay.
         c. Nếu lân cận XẤU HƠN → chấp nhận với xác suất P = e^(-Δ/T).
         d. Cập nhật TKB tốt nhất (best) nếu cần.
         e. Lặp lại iterations_per_temp lần tại mỗi mức nhiệt.
         f. Hạ nhiệt: T = T × cooling_rate.
      3. Trả về TKB tốt nhất từng gặp trong toàn bộ quá trình.

    Ưu điểm: Có khả năng thoát cực trị cục bộ, hội tụ về lời giải tốt.
    Nhược điểm: Cần tinh chỉnh tham số (T_init, cooling_rate, min_temp).
    """

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện Simulated Annealing trên bài toán xếp TKB.

        Args:
            graph: Đối tượng TimetableProblem.
            **kwargs:
                initial_temperature (float): Nhiệt độ ban đầu (mặc định 100.0).
                cooling_rate (float): Hệ số hạ nhiệt (mặc định 0.95).
                min_temperature (float): Nhiệt độ dừng (mặc định 0.01).
                iterations_per_temp (int): Số lần lặp tại mỗi mức nhiệt (mặc định 20).
        """
        start_time = time.perf_counter()
        
        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        # ── Tham số luyện kim ──
        temp = kwargs.get("initial_temperature", 100.0)     # T₀: Nhiệt độ ban đầu (cao → khám phá)
        cooling_rate = kwargs.get("cooling_rate", 0.95)      # α: Hệ số hạ nhiệt (0 < α < 1)
        min_temp = kwargs.get("min_temperature", 0.01)       # T_min: Ngưỡng dừng
        iterations_per_temp = kwargs.get("iterations_per_temp", 20)  # Số lần thử tại mỗi mức T
        
        variables = list(graph.sections.keys())
        
        # ══════════════════════════════════════════════════════════════
        # BƯỚC 1: KHỞI TẠO TKB NGẪU NHIÊN (giống Hill Climbing)
        # ══════════════════════════════════════════════════════════════
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
        
        # ── Lưu TKB tốt nhất từng gặp ──
        # SA có thể di chuyển sang trạng thái xấu hơn, nên cần lưu riêng
        # trạng thái tốt nhất để trả về cuối cùng.
        best_timetable = current_timetable.copy()
        best_fitness = current_fitness
        
        explored_count = 0
        
        # ══════════════════════════════════════════════════════════════
        # BƯỚC 2: VÒNG LẶP LUYỆN KIM
        # ══════════════════════════════════════════════════════════════
        # Vòng ngoài: Giảm nhiệt độ dần dần (T → T × cooling_rate)
        # Vòng trong: Thử iterations_per_temp lân cận tại mỗi mức nhiệt T
        while temp > min_temp:
            for _ in range(iterations_per_temp):
                explored_count += 1
                
                # Sinh 1 lân cận ngẫu nhiên (toán tử biến đổi giống Hill Climbing)
                neighbor = current_timetable.copy()
                
                # ── Toán tử biến đổi lân cận ──
                # 50% đổi ngẫu nhiên 1 môn, 50% hoán đổi 2 môn
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
                
                # ══════════════════════════════════════════════════════
                # TIÊU CHUẨN CHẤP NHẬN METROPOLIS
                # ══════════════════════════════════════════════════════
                # delta = current_fitness - neighbor_fitness
                #   - delta < 0 → lân cận TỐT HƠN (fitness cao hơn) → LUÔN chấp nhận.
                #   - delta > 0 → lân cận XẤU HƠN → chấp nhận với xác suất e^(-delta/T).
                # Khi T cao: e^(-delta/T) ≈ 1 → gần như luôn chấp nhận → khám phá rộng.
                # Khi T thấp: e^(-delta/T) ≈ 0 → hầu như chỉ chấp nhận cải thiện → hội tụ.
                delta = current_fitness - neighbor_fitness
                
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / temp)):
                    current_timetable = neighbor
                    current_fitness = neighbor_fitness
                    
                    # Cập nhật TKB tốt nhất nếu lân cận vừa chấp nhận là tốt nhất từ trước đến nay
                    if current_fitness > best_fitness:
                        best_timetable = current_timetable.copy()
                        best_fitness = current_fitness
            
            # ── Hạ nhiệt (Cooling) ──
            # T_new = T_old × α (geometric cooling schedule)
            # Với α = 0.95: 100 → 95 → 90.25 → ... → 0.01 (khoảng 180 vòng)
            temp *= cooling_rate

        execution_time = time.perf_counter() - start_time
        
        # ══════════════════════════════════════════════════════════════
        # ĐÁNH GIÁ KẾT QUẢ
        # ══════════════════════════════════════════════════════════════
        # Trả về TKB tốt nhất từng gặp (best), không phải TKB cuối cùng (current),
        # vì SA có thể đã di chuyển sang trạng thái xấu hơn ở các bước cuối.
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
