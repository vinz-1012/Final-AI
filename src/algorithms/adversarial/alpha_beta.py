# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# Hoàn toàn giống Minimax nhưng thêm α và β để cắt tỉa:
# - MAX (Người xếp lịch) chọn ca/phòng tốt nhất.
# - MIN (Sự cố) mô phỏng phòng bị khóa.
# - Khi phát hiện nhánh chắc chắn không ảnh hưởng kết quả → bỏ qua.

import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class AlphaBetaSearch(BaseSearchAlgorithm):
    """
    Alpha-Beta Pruning — Minimax có cắt tỉa cho bài toán xếp TKB.

    Giống Minimax nhưng thêm cắt tỉa α-β để giảm số nút phải duyệt:
      - α (alpha): Giới hạn dưới của MAX (MAX đã đảm bảo được ≥ α).
      - β (beta): Giới hạn trên của MIN (MIN đã đảm bảo được ≤ β).
      - Khi α ≥ β → cắt nhánh (không cần duyệt thêm).

    Ưu điểm: Kết quả giống Minimax nhưng nhanh hơn nhiều nhờ cắt tỉa.
    Nhược điểm: Hiệu quả cắt tỉa phụ thuộc vào thứ tự duyệt nút.
    """

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện Alpha-Beta Pruning cho 1 lớp HP mục tiêu.
        """
        start_time = time.perf_counter()
        
        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}
            
        depth = kwargs.get("depth", 2)
        self.explored_nodes = 0
        
        # ── Xác định lớp HP mục tiêu ──
        sec_id = start
        if sec_id not in graph.sections:
            sec_id = list(graph.sections.keys())[0] if graph.sections else ""
            
        if not sec_id:
            execution_time = time.perf_counter() - start_time
            return None, 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ── Dựng TKB cơ sở (giống Minimax) ──
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
        
        # ── Khởi tạo α và β ──
        # α = -∞: MAX chưa tìm được gì → mọi giá trị đều chấp nhận được.
        # β = +∞: MIN chưa tìm được gì → mọi giá trị đều chấp nhận được.
        alpha = -float('inf')
        beta = float('inf')
        
        # ══════════════════════════════════════════════════════════════
        # NÚT GỐC MAX (với α-β pruning)
        # ══════════════════════════════════════════════════════════════
        for period_id, room_id in domain[:15]:
            lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
            timetable.assign(sec_id, period_id, room_id, lec_id)
            
            # Gọi MIN với α và β hiện tại
            val = self._min_value(timetable, graph, depth - 1, alpha, beta)
            if val > best_val:
                best_val = val
                best_move = (period_id, room_id)
                
            timetable.unassign(sec_id)
            
            # ── Cập nhật α và kiểm tra cắt tỉa ──
            # α = max(α, best_val): MAX đã đảm bảo được ít nhất best_val.
            alpha = max(alpha, best_val)
            # Nếu α ≥ β → CẮT: Các nhánh sau chắc chắn không cải thiện kết quả.
            if alpha >= beta:
                break  # Beta cut-off tại nút gốc MAX

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
        """Hàm đánh giá tại nút lá (giống Minimax)."""
        return ConstraintChecker.compute_fitness(
            timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
        )

    def _max_value(self, timetable: Timetable, graph: TimetableProblem, depth: int, alpha: float, beta: float) -> float:
        """
        Nút MAX (có α-β pruning):
        MAX chọn hành động cực đại hóa fitness, truyền α-β xuống con.
        """
        self.explored_nodes += 1
        if depth == 0:
            return self._eval(timetable, graph)
        return self._eval(timetable, graph)

    def _min_value(self, timetable: Timetable, graph: TimetableProblem, depth: int, alpha: float, beta: float) -> float:
        """
        Nút MIN (có α-β pruning):
        MIN cố gắng cực tiểu hóa fitness bằng cách mô phỏng sự cố phòng học.

        Cắt tỉa α (alpha cut-off):
          Nếu val ≤ α → CẮT: MAX đã có phương án tốt hơn (≥ α) ở nhánh khác,
          nên sẽ KHÔNG BAO GIỜ chọn nhánh này → không cần duyệt thêm.
        """
        self.explored_nodes += 1
        if depth == 0:
            return self._eval(timetable, graph)
            
        val = float('inf')
        for room in list(graph.rooms.values())[:5]:
            # Mô phỏng sự cố: Khóa phòng
            old_available = room.is_available
            room.is_available = False
            
            val = min(val, self._max_value(timetable, graph, depth - 1, alpha, beta))
            
            # Khôi phục phòng
            room.is_available = old_available
            
            # ══════════════════════════════════════════════════════
            # ALPHA CUT-OFF (Cắt tỉa α tại nút MIN)
            # ══════════════════════════════════════════════════════
            # Nếu val ≤ α: Giá trị MIN tìm được (val) đã nhỏ hơn hoặc bằng
            # giới hạn dưới của MAX (α). MAX sẽ không chọn nhánh này vì
            # đã có nhánh khác cho giá trị ≥ α.
            # → Không cần duyệt thêm các phòng còn lại → CẮT.
            if val <= alpha:
                return val  # Alpha cut-off!
            # Cập nhật β: MIN đảm bảo được nhiều nhất val
            beta = min(beta, val)
            
        return val
