# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# Mô phỏng tình huống: Phòng học có thể bị sự cố bất ngờ (bảo trì, mất điện, ...),
# nên cần lập kế hoạch dự phòng cho việc xếp lịch.
#
# Quy trình 2 bước:
#   BƯỚC 1: Xếp lịch toàn bộ bằng chiến lược tham lam (greedy assignment).
#   BƯỚC 2: Lập kế hoạch dự phòng cho 1 lớp HP mục tiêu.
#     - OR Node: Thử gán lớp HP vào các cặp (ca, phòng) khác nhau.
#     - AND Node: Với mỗi phương án gán, tìm phòng dự phòng (backup room)
#       cùng ca nhưng khác phòng → nếu phòng chính sự cố, chuyển sang phòng backup.
#
# Kế hoạch dự phòng (contingency plan) có dạng:
#   {
#     "action": "Assign to P1 @ R1",          ← Phương án chính
#     "if_success": "Xếp thành công",         ← Nếu phòng R1 OK
#     "if_room_failed_divert_to": "P1 @ R2",  ← Nếu phòng R1 sự cố → chuyển R2
#     "backup_plan": "Chuyển sang phòng R2"    ← Mô tả kế hoạch dự phòng
#   }

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

# Định nghĩa kiểu kế hoạch điều kiện (Contingency Plan)
# Plan có thể là:
#   - str: Hành động đơn giản.
#   - Dict: Kế hoạch có điều kiện (if-then-else).
#   - List: Danh sách hành động tuần tự.
Plan = Union[str, Dict[str, Any], List[Any]]

class AndOrSearch(BaseSearchAlgorithm):
    """
    AND-OR Search — Tìm kiếm trong môi trường không xác định cho TKB.

    Nguyên lý hoạt động:
      Bước 1: Xếp lịch tham lam (greedy) → tạo TKB đầy đủ nhanh chóng.
      Bước 2: Lập kế hoạch dự phòng (contingency plan) cho lớp HP mục tiêu:
        - OR Node: Thử các phương án gán (ca, phòng).
        - AND Node: Tìm phòng dự phòng (backup) cho mỗi phương án.
        - Kết quả: Kế hoạch "nếu phòng OK → dùng, nếu sự cố → chuyển backup".

    Ưu điểm: Lập kế hoạch robust, xử lý được tình huống bất ngờ.
    Nhược điểm: Chỉ lập kế hoạch cho 1 lớp HP, phức tạp khi mở rộng.
    """

    def search(
        self,
        graph: Graph,
        start: str,
        goal: str,
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện AND-OR Search cho bài toán xếp TKB.

        Args:
            graph: Đối tượng TimetableProblem.
            start: section_id của lớp HP mục tiêu cần lập kế hoạch dự phòng.

        Returns:
            Tuple gồm (default_path, cost, info).
            info["contingency_plan"] chứa kế hoạch dự phòng dạng AND-OR.
        """
        start_time = time.perf_counter()

        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        self.explored_nodes = 0

        # Xác định lớp HP mục tiêu để lập kế hoạch dự phòng
        sec_id = start
        if sec_id not in graph.sections:
            sec_id = list(graph.sections.keys())[0] if graph.sections else ""

        if not sec_id:
            execution_time = time.perf_counter() - start_time
            return None, 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ══════════════════════════════════════════════════════════════
        # BƯỚC 1: XẾP LỊCH THAM LAM (GREEDY ASSIGNMENT)
        # ══════════════════════════════════════════════════════════════
        # Với mỗi lớp HP, duyệt domain theo thứ tự → gán ngay giá trị hợp lệ đầu tiên.
        # Ưu điểm: Nhanh (O(n×d)), tạo được TKB đầy đủ để đánh giá.
        # Nhược điểm: Không đảm bảo tối ưu (tham lam).
        timetable = Timetable()
        assigned_count = 0
        for s_id in graph.sections:
            domain = graph.get_domain_for_section(s_id)
            self.explored_nodes += 1
            for period_id, room_id in domain:
                # Kiểm tra ràng buộc cứng trước khi gán
                is_valid, _ = ConstraintChecker.check_all_hard(
                    timetable, s_id, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    lec_id = graph.sections[s_id].lecturer_id
                    timetable.assign(s_id, period_id, room_id, lec_id)
                    assigned_count += 1
                    break  # Tham lam: chọn giá trị hợp lệ đầu tiên

        # ══════════════════════════════════════════════════════════════
        # BƯỚC 2: LẬP KẾ HOẠCH DỰ PHÒNG AND-OR
        # ══════════════════════════════════════════════════════════════
        # Gọi _and_or_search để tạo contingency plan cho lớp HP mục tiêu.
        plan = self._and_or_search(graph, sec_id, timetable)

        # Trích xuất đường đi mặc định (default path) từ kế hoạch
        default_path = [sec_id]
        if isinstance(plan, dict) and "action" in plan:
            default_path.append(plan["action"])
        elif isinstance(plan, list) and plan:
            default_path.append(plan[0])

        # ── Thống kê kết quả ──
        hard_violations = ConstraintChecker.count_hard_violations(
            timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
        )
        soft_violations = ConstraintChecker.count_soft_violations(
            timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
        )
        fitness = ConstraintChecker.compute_fitness(
            timetable, graph.sections, graph.courses, graph.rooms, graph.periods, graph.lecturers
        )
        cost = float(hard_violations + soft_violations["total"])

        execution_time = time.perf_counter() - start_time

        return default_path, cost, {
            "explored_nodes": self.explored_nodes,
            "execution_time": execution_time,
            "timetable": timetable,
            "hard_violations": hard_violations,
            "soft_violations": soft_violations,
            "fitness": fitness,
            "contingency_plan": plan,      # Kế hoạch dự phòng AND-OR
            "next_step": default_path[1] if len(default_path) > 1 else None
        }

    def _and_or_search(self, graph: TimetableProblem, sec_id: str, timetable: Timetable) -> Optional[Plan]:
        """
        Thuật toán AND-OR Search để lập kế hoạch dự phòng.

        Cấu trúc cây AND-OR:
        ┌─────────────────────────────────────────────────┐
        │  OR Node (Lớp HP mục tiêu)                      │
        │  ├── Action 1: Gán vào P1@R1                    │ ← OR: chọn 1
        │  │   └── AND Node: Kết quả                      │
        │  │       ├── if_success → Thành công             │ ← AND: cả 2 phải xử lý
        │  │       └── if_room_failed → Backup P1@R2       │
        │  ├── Action 2: Gán vào P2@R3                    │ ← OR: hoặc chọn cái này
        │  │   └── ...                                     │
        │  └── ...                                         │
        └─────────────────────────────────────────────────┘

        Args:
            graph: Bài toán TKB.
            sec_id: Lớp HP cần lập kế hoạch.
            timetable: TKB hiện tại (đã xếp greedy).

        Returns:
            Plan: Kế hoạch dự phòng dạng dict/list/str, hoặc None nếu thất bại.
        """
        self.explored_nodes += 1

        domain = graph.get_domain_for_section(sec_id)
        if not domain:
            return None

        # Lấy vị trí hiện tại của lớp HP (nếu đã được xếp trong Bước 1)
        current_entry = timetable.get_entry(sec_id)

        # ── OR Node: Thử từng phương án gán (Period, Room) ──
        # Giới hạn 8 phương án để kiểm soát thời gian chạy.
        for period_id, room_id in domain[:8]:
            action = f"Assign to {period_id} @ {room_id}"

            # Tạm thời bỏ gán lớp HP này để kiểm tra khả thi từ đầu
            timetable.unassign(sec_id)
            is_valid, _ = ConstraintChecker.check_all_hard(
                timetable, sec_id, period_id, room_id,
                graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            # Khôi phục gán ban đầu
            if current_entry:
                timetable.assign(sec_id, current_entry.period_id, current_entry.room_id, current_entry.lecturer_id)

            if not is_valid:
                continue  # Phương án không khả thi → thử phương án OR tiếp theo

            # ── AND Node: Tìm phương án dự phòng (backup room) ──
            # Tìm phòng khác CÙNG CA nhưng KHÁC PHÒNG → nếu phòng chính sự cố,
            # có thể chuyển sang phòng backup ngay mà không cần đổi ca.
            backup_options = [(p, r) for p, r in domain if p == period_id and r != room_id]

            chosen_backup = None
            for p_b, r_b in backup_options:
                # Kiểm tra khả thi của phòng backup
                timetable.unassign(sec_id)
                is_b_valid, _ = ConstraintChecker.check_all_hard(
                    timetable, sec_id, p_b, r_b,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if current_entry:
                    timetable.assign(sec_id, current_entry.period_id, current_entry.room_id, current_entry.lecturer_id)
                if is_b_valid:
                    chosen_backup = f"{p_b} @ {r_b}"
                    break

            if chosen_backup:
                # ── Trả về kế hoạch dự phòng AND ──
                # Kế hoạch bao gồm CẢ trường hợp thành công VÀ thất bại (AND).
                return {
                    "action": action,                                    # Hành động chính
                    "if_success": f"Xếp lịch thành công tại {period_id} @ {room_id}",  # Kết quả tốt
                    "if_room_failed_divert_to": chosen_backup,           # Phòng dự phòng
                    "backup_plan": f"Chuyển sang phòng dự phòng: {chosen_backup}"       # Mô tả
                }
            else:
                # Không tìm được phòng backup → trả về hành động đơn (không có dự phòng)
                return [action]

        return None  # Không tìm được phương án nào khả thi
