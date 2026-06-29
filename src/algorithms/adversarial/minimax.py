# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# Mô phỏng đối kháng giữa:
#   - MAX = Người xếp lịch: Chọn ca/phòng tốt nhất cho 1 lớp HP mục tiêu.
#     Mục tiêu: Tối đa hóa fitness (chất lượng TKB).
#   - MIN = Sự cố (disruption): Mô phỏng các tình huống xấu nhất (phòng bị khóa,
#     GV bận đột xuất, ...) để làm giảm chất lượng TKB.
#
# Ý nghĩa: Minimax giúp tìm phương án "an toàn nhất" — phương án tốt nhất
# ngay cả khi mọi thứ xấu nhất xảy ra (worst-case optimal).
#
# ── CÁCH THỨC MÔ PHỎNG ──
# - MAX: Thử gán lớp HP mục tiêu vào các cặp (ca, phòng) trong domain.
# - MIN: Thử "khóa" lần lượt các phòng học (mô phỏng sự cố phòng học).
# - Hàm đánh giá: compute_fitness() → đánh giá chất lượng TKB tại nút lá.

import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class MinimaxSearch(BaseSearchAlgorithm):
    """
    Minimax — Thuật toán tìm kiếm đối kháng cho bài toán xếp TKB.

    Mô phỏng trò chơi đối kháng 2 bên:
      - MAX (Người xếp lịch): Chọn ca/phòng tối ưu cho 1 lớp HP.
      - MIN (Sự cố): Mô phỏng tình huống xấu nhất (phòng bị khóa).

    Kết quả: Phương án gán (ca, phòng) TỐT NHẤT trong TÌNH HUỐNG XẤU NHẤT.

    Ưu điểm: Tìm được chiến lược robust (chống chịu sự cố).
    Nhược điểm: Chậm (O(b^m)), cần giới hạn depth và branching factor.
    """

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện Minimax cho 1 lớp HP mục tiêu.

        Args:
            graph: Đối tượng TimetableProblem.
            start: section_id của lớp HP mục tiêu cần xếp.
            **kwargs: depth (int) — độ sâu cây trò chơi (mặc định 2).
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

        # ── Dựng TKB cơ sở cho các lớp khác ──
        # Gán tạm các lớp HP khác (trừ lớp mục tiêu) để tạo bối cảnh đánh giá.
        timetable = Timetable()
        for s_id in graph.sections:
            if s_id != sec_id:
                domain = graph.get_domain_for_section(s_id)
                if domain:
                    lec_id = graph.sections[s_id].lecturer_id if s_id in graph.sections else None
                    timetable.assign(s_id, domain[0][0], domain[0][1], lec_id)

        # ══════════════════════════════════════════════════════════════
        # NÚT GỐC MAX: THỬ CÁC PHƯƠNG ÁN GÁN CHO LỚP MỤC TIÊU
        # ══════════════════════════════════════════════════════════════
        domain = graph.get_domain_for_section(sec_id)
        best_val = -float('inf')
        best_move = None
        
        # Giới hạn 15 ca/phòng để tránh bùng nổ tổ hợp
        for period_id, room_id in domain[:15]:
            lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
            timetable.assign(sec_id, period_id, room_id, lec_id)
            
            # Gọi MIN: Đối thủ cố gắng làm giảm fitness
            val = self._min_value(timetable, graph, depth - 1)
            if val > best_val:
                best_val = val        # MAX chọn giá trị cao nhất
                best_move = (period_id, room_id)
                
            timetable.unassign(sec_id)  # Khôi phục để thử phương án tiếp

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
        """
        Hàm đánh giá (evaluation function) tại nút lá.
        Trả về fitness của TKB → MAX muốn cực đại, MIN muốn cực tiểu.
        """
        return ConstraintChecker.compute_fitness(
            timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
        )

    def _max_value(self, timetable: Timetable, graph: TimetableProblem, depth: int) -> float:
        """
        Nút MAX: Người xếp lịch chọn hành động CỰC ĐẠI HÓA fitness.
        Ở depth = 0 (nút lá) → trả về giá trị đánh giá trực tiếp.
        """
        self.explored_nodes += 1
        if depth == 0:
            return self._eval(timetable, graph)
        # Simplified: trả về eval tại nút này (không mở rộng thêm nhánh MAX)
        return self._eval(timetable, graph)

    def _min_value(self, timetable: Timetable, graph: TimetableProblem, depth: int) -> float:
        """
        Nút MIN: Sự cố cố gắng CỰC TIỂU HÓA fitness.
        Mô phỏng: MIN thử "khóa" từng phòng học → tìm tình huống xấu nhất.

        Giải thích: Nếu phòng bị sự cố (bảo trì, hỏng thiết bị, ...),
        TKB sẽ bị ảnh hưởng thế nào? MIN tìm phương án gây hại nhiều nhất.
        """
        self.explored_nodes += 1
        if depth == 0:
            return self._eval(timetable, graph)
            
        val = float('inf')
        # MIN mô phỏng sự cố: Thử khóa lần lượt 5 phòng học
        for room in list(graph.rooms.values())[:5]:
            # Tạm khóa phòng (mô phỏng phòng bị sự cố)
            old_available = room.is_available
            room.is_available = False
            
            # Đánh giá: fitness giảm bao nhiêu khi phòng này bị khóa?
            val = min(val, self._max_value(timetable, graph, depth - 1))
            
            # Khôi phục trạng thái phòng
            room.is_available = old_available
            
        return val  # MIN chọn giá trị THẤP NHẤT (tình huống xấu nhất)
