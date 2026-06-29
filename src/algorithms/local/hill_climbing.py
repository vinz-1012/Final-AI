# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
#   - Trạng thái: Một TKB đầy đủ (tất cả lớp HP đã được gán ca & phòng).
#   - Hàm mục tiêu: fitness = (số lớp đã xếp × 10) − vi phạm → CỰC ĐẠI HÓA.
#   - Lân cận (neighbor): Biến đổi nhỏ trên TKB hiện tại:
#       + 50% xác suất: Đổi 1 lớp HP sang ca/phòng ngẫu nhiên khác.
#       + 50% xác suất: Hoán đổi ca/phòng của 2 lớp HP với nhau.
#   - Thuật toán sinh 30 lân cận, chọn cái tốt nhất → steepest ascent variant.
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class HillClimbingSearch(BaseSearchAlgorithm):
    """
    Hill Climbing — Thuật toán leo đồi cho bài toán tối ưu Thời Khóa Biểu.

    Nguyên lý hoạt động:
      1. Khởi tạo một TKB ngẫu nhiên đầy đủ (tất cả lớp HP đều có ca & phòng).
      2. Lặp (tối đa max_steps bước):
         a. Sinh 30 lân cận ngẫu nhiên (đổi ca/phòng 1 lớp hoặc hoán đổi 2 lớp).
         b. Chọn lân cận có fitness cao nhất (steepest ascent).
         c. Nếu fitness lân cận > fitness hiện tại → di chuyển sang.
         d. Nếu không có lân cận nào tốt hơn → DỪNG (đạt cực trị cục bộ).

    Ưu điểm: Nhanh, đơn giản, tiết kiệm bộ nhớ.
    Nhược điểm: Dễ bị kẹt ở cực trị cục bộ, không đảm bảo tìm được lời giải tối ưu.
    """

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện Hill Climbing trên bài toán xếp TKB.

        Args:
            graph: Đối tượng TimetableProblem.
            **kwargs: max_steps (int) — số bước lặp tối đa (mặc định 1000).

        Returns:
            Tuple (path, cost, info) — luôn trả về kết quả (kể cả khi chưa tối ưu).
        """
        start_time = time.perf_counter()
        
        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        max_steps = kwargs.get("max_steps", 1000)
        variables = list(graph.sections.keys())
        
        # ══════════════════════════════════════════════════════════════
        # BƯỚC 1: KHỞI TẠO TKB NGẪU NHIÊN
        # ══════════════════════════════════════════════════════════════
        # Gán mỗi lớp HP vào một cặp (ca, phòng) ngẫu nhiên từ domain.
        # Lưu ý: TKB ban đầu có thể vi phạm ràng buộc — Hill Climbing sẽ cải thiện dần.
        current_timetable = Timetable()
        for sec_id in variables:
            domain = graph.get_domain_for_section(sec_id)
            if domain:
                p_id, r_id = random.choice(domain)  # Chọn ngẫu nhiên từ domain
                lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                current_timetable.assign(sec_id, p_id, r_id, lec_id)
                
        # Tính fitness ban đầu: fitness = (assigned × 10) − (hard + soft violations)
        current_fitness = ConstraintChecker.compute_fitness(
            current_timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
        )
        
        explored_count = 0
        
        # ══════════════════════════════════════════════════════════════
        # BƯỚC 2: VÒNG LẶP LEO ĐỒI (STEEPEST ASCENT)
        # ══════════════════════════════════════════════════════════════
        for step in range(max_steps):
            explored_count += 1
            improved = False
            
            # ── Sinh 30 lân cận ngẫu nhiên ──
            # Chọn số 30 là trade-off: đủ nhiều để tìm hướng leo tốt,
            # nhưng không quá lớn để giữ tốc độ nhanh.
            best_neighbor_timetable = None
            best_neighbor_fitness = current_fitness
            
            for _ in range(30):
                # Sao chép TKB hiện tại để tạo lân cận
                neighbor = current_timetable.copy()
                
                # ── Toán tử biến đổi lân cận ──
                # 50% xác suất: Đổi 1 lớp HP sang ca/phòng ngẫu nhiên khác
                #   → Khám phá vùng mới trong không gian tìm kiếm.
                # 50% xác suất: Hoán đổi ca/phòng giữa 2 lớp HP
                #   → Giữ nguyên tài nguyên, chỉ tráo đổi → ít gây xung đột mới.
                if random.random() < 0.5 or len(variables) < 2:
                    # Kiểu 1: Gán lại 1 lớp HP vào ca/phòng ngẫu nhiên
                    sec_id = random.choice(variables)
                    domain = graph.get_domain_for_section(sec_id)
                    if domain:
                        p_id, r_id = random.choice(domain)
                        lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                        neighbor.assign(sec_id, p_id, r_id, lec_id)
                else:
                    # Kiểu 2: Hoán đổi vị trí (ca, phòng) của 2 lớp HP
                    sec1, sec2 = random.sample(variables, 2)
                    e1 = neighbor.get_entry(sec1)
                    e2 = neighbor.get_entry(sec2)
                    if e1 and e2:
                        lec1 = graph.sections[sec1].lecturer_id if sec1 in graph.sections else None
                        lec2 = graph.sections[sec2].lecturer_id if sec2 in graph.sections else None
                        # Tráo đổi: sec1 lấy vị trí của sec2 và ngược lại
                        neighbor.assign(sec1, e2.period_id, e2.room_id, lec1)
                        neighbor.assign(sec2, e1.period_id, e1.room_id, lec2)
                
                # Đánh giá fitness của lân cận
                fit = ConstraintChecker.compute_fitness(
                    neighbor, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
                )
                
                # Chỉ chấp nhận lân cận TỐT HƠN NGHIÊM NGẶT (strictly better)
                if fit > best_neighbor_fitness:
                    best_neighbor_fitness = fit
                    best_neighbor_timetable = neighbor
                    
            # ── Di chuyển sang lân cận tốt nhất (nếu có) ──
            if best_neighbor_timetable is not None:
                current_timetable = best_neighbor_timetable
                current_fitness = best_neighbor_fitness
                improved = True
                
            if not improved:
                # ── DỪNG: Đã đạt cực trị cục bộ ──
                # Không có lân cận nào tốt hơn → đỉnh đồi hiện tại là tốt nhất.
                # Đây là nhược điểm chính của Hill Climbing.
                break

        execution_time = time.perf_counter() - start_time
        
        # ══════════════════════════════════════════════════════════════
        # ĐÁNH GIÁ KẾT QUẢ
        # ══════════════════════════════════════════════════════════════
        # Hill Climbing LUÔN trả về kết quả (TKB tốt nhất tìm được),
        # khác với BFS/DFS/A* có thể trả về None nếu không tìm được lời giải đầy đủ.
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
