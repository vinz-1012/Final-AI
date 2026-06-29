
# Bài toán CSP gồm 3 thành phần:
#   - Variables (Biến): Các lớp học phần (section) cần xếp lịch.
#   - Domain (Miền giá trị): Các cặp (ca, phòng) mà mỗi lớp có thể xếp vào.
#   - Constraints (Ràng buộc): Các ràng buộc cứng (hard constraints).
#
# ── CÁCH HOẠT ĐỘNG ──
# 1. Chọn biến tiếp theo để gán (dùng MRV Heuristic).
# 2. Duyệt các giá trị trong domain theo thứ tự (dùng LCV Heuristic).
# 3. Với mỗi giá trị: kiểm tra ràng buộc cứng.
#    - Nếu hợp lệ → gán và đệ quy sang biến tiếp theo.
#    - Nếu đệ quy thất bại → GỠ GÁN (backtrack) và thử giá trị tiếp.
# 4. Nếu hết giá trị → quay lui (backtrack) về biến trước đó.
#
# ── CÁC HEURISTIC ĐƯỢC SỬ DỤNG ──
# 1. MRV (Minimum Remaining Values): Chọn biến có domain nhỏ nhất trước.
#    → Lý do: Biến có ít lựa chọn nhất dễ thất bại nhất → phát hiện sớm.
# 2. LCV (Least Constraining Value): Sắp xếp domain theo sức chứa phòng tăng dần.
#    → Lý do: Gán vào phòng nhỏ trước → giữ phòng lớn cho các lớp đông sinh viên.


import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

class CSPBacktrackingSearch(BaseSearchAlgorithm):
    """
    Backtracking Search — Tìm kiếm quay lui giải bài toán CSP xếp TKB.

    Nguyên lý hoạt động:
      1. Sắp xếp biến theo MRV (domain nhỏ trước).
      2. Đệ quy: Với mỗi biến, thử gán từng giá trị trong domain (LCV).
      3. Kiểm tra ràng buộc cứng TRƯỚC khi gán.
      4. Nếu hợp lệ → đệ quy tiếp. Nếu thất bại → gỡ gán (backtrack).
      5. Tất cả biến được gán hợp lệ → trả về lời giải.

    Ưu điểm: Đảm bảo tìm lời giải (nếu tồn tại), kiểm tra ràng buộc sớm.
    Nhược điểm: Worst-case vẫn exponential, chậm với bài toán lớn.
    """

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện Backtracking CSP trên bài toán xếp TKB.

        Args:
            graph: Đối tượng TimetableProblem.
            **kwargs: max_explored (int) — giới hạn số trạng thái duyệt tối đa.

        Returns:
            Tuple (path, cost, info).
        """
        start_time = time.perf_counter()
        
        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        # ── Variables (Biến CSP) ──
        # Mỗi biến = 1 lớp HP cần xếp lịch.
        # ── MRV Heuristic: Sắp xếp biến theo kích thước domain tăng dần ──
        # Biến có domain nhỏ nhất → dễ thất bại nhất → xếp trước → phát hiện dead-end sớm.
        variables = list(graph.sections.keys())
        variables.sort(key=lambda s: len(graph.get_domain_for_section(s)))
        
        timetable = Timetable()
        self.explored_states = 0
        self.max_explored = kwargs.get("max_explored", 2000)

        # ── Gọi hàm đệ quy backtrack ──
        success = self._backtrack(timetable, variables, 0, graph)
        
        execution_time = time.perf_counter() - start_time
        
        # Đánh giá kết quả
        hard_violations = ConstraintChecker.count_hard_violations(
            timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
        )
        soft_violations = ConstraintChecker.count_soft_violations(
            timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
        )
        cost = float(hard_violations + soft_violations["total"])
        
        # Xác định thông báo lỗi (nếu thất bại)
        if not success and self.explored_states >= self.max_explored:
            error_msg = f"Đạt giới hạn duyệt tối đa ({self.max_explored} trạng thái) mà chưa tìm được lời giải."
        else:
            error_msg = "Không tìm thấy lời giải hợp lệ hoàn chỉnh."

        return (
            list(timetable.entries.keys()) if success else None,
            cost,
            {
                "explored_nodes": self.explored_states,
                "execution_time": execution_time,
                "timetable": timetable,
                "hard_violations": hard_violations,
                "soft_violations": soft_violations,
                "error": None if success else error_msg,
                "fitness": ConstraintChecker.compute_fitness(
                    timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
                )
            }
        )

    def _backtrack(
        self, 
        timetable: Timetable, 
        variables: List[str], 
        var_idx: int, 
        graph: TimetableProblem
    ) -> bool:
        """
        Hàm đệ quy Backtracking.

        Quy trình tại mỗi lần gọi đệ quy:
          1. Kiểm tra giới hạn duyệt.
          2. Kiểm tra base case: tất cả biến đã gán → THÀNH CÔNG.
          3. Lấy biến tiếp theo (đã sắp MRV).
          4. Duyệt domain theo LCV → kiểm tra ràng buộc → gán → đệ quy.
          5. Nếu đệ quy thất bại → gỡ gán (backtrack) → thử giá trị tiếp.

        Args:
            timetable: TKB đang xây dựng (sửa trực tiếp, in-place).
            variables: Danh sách section_id đã sắp theo MRV.
            var_idx: Chỉ số biến đang xét (0-based).
            graph: Bài toán TimetableProblem.

        Returns:
            True nếu xếp thành công tất cả biến, False nếu thất bại.
        """
        self.explored_states += 1

        # Kiểm tra giới hạn duyệt — tránh chạy quá lâu
        if self.explored_states >= self.max_explored:
            return False
        
        # ── Base case: Tất cả biến đã được gán → TKB hoàn chỉnh ──
        if var_idx == len(variables):
            return True
            
        # ── Lấy biến (lớp HP) tiếp theo ──
        sec_id = variables[var_idx]
        domain = graph.get_domain_for_section(sec_id)
        
        # ── LCV Heuristic (Least Constraining Value) ──
        # Sắp xếp domain theo sức chứa phòng học tăng dần.
        # Lý do: Gán vào phòng nhỏ vừa đủ → giữ lại phòng lớn cho các lớp đông SV.
        # → Giảm khả năng gây xung đột với các biến chưa gán.
        domain.sort(key=lambda item: graph.rooms[item[1]].capacity)
        
        # ── Duyệt từng giá trị trong domain ──
        for period_id, room_id in domain:
            # Kiểm tra ràng buộc cứng TRƯỚC khi gán (forward checking)
            is_valid, _ = ConstraintChecker.check_all_hard(
                timetable, sec_id, period_id, room_id,
                graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            
            if is_valid:
                # ── Gán giá trị cho biến ──
                lec_id = graph.sections[sec_id].lecturer_id if sec_id in graph.sections else None
                timetable.assign(sec_id, period_id, room_id, lec_id)
                
                # ── Đệ quy: Tiếp tục gán biến tiếp theo ──
                if self._backtrack(timetable, variables, var_idx + 1, graph):
                    return True  # Tìm được lời giải!
                
                # ── BACKTRACK: Gỡ gán và thử giá trị khác ──
                # Nếu đệ quy thất bại (các biến sau không gán được) → quay lui.
                # Đây là bước quan trọng nhất: undo phép gán vừa thực hiện.
                timetable.unassign(sec_id)
                
        # Hết giá trị trong domain → trả về False (nhánh này thất bại hoàn toàn)
        return False
