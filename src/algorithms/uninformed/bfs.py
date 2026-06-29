# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# Mỗi trạng thái (node) là một TKB bán hoàn chỉnh, trong đó một số lớp HP
# đã được gán vào (ca, phòng) và các lớp còn lại chưa xếp.
# - Trạng thái gốc (root): TKB rỗng (chưa có lớp nào được xếp).
# - Mỗi nhánh con (child): Là kết quả của việc gán thêm 1 lớp HP vào 1 cặp
#   (ca học, phòng học) hợp lệ.
# - Trạng thái đích (goal): TKB đầy đủ, tất cả lớp HP đều đã được xếp.


import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker


class BreadthFirstSearch(BaseSearchAlgorithm):
    """
    BFS — Tìm kiếm theo chiều rộng cho bài toán xếp Thời Khóa Biểu.

    Nguyên lý hoạt động:
      1. Khởi tạo hàng đợi FIFO với trạng thái gốc (TKB rỗng).
      2. Lặp: Lấy trạng thái đầu hàng đợi ra → kiểm tra hoàn thành →
         nếu chưa, sinh tất cả trạng thái con hợp lệ và đẩy vào cuối hàng đợi.
      3. Dừng khi tìm được TKB đầy đủ hoặc hết trạng thái để duyệt.

    Ưu điểm: Đảm bảo tìm lời giải nông nhất (completeness + optimality theo bước).
    Nhược điểm: Tốn bộ nhớ rất lớn O(b^d), không phù hợp bài toán lớn.
    """

    def search(
        self,
        graph: Graph,
        start: str,
        goal: str,
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện tìm kiếm BFS trên không gian trạng thái TKB.

        Args:
            graph: Đối tượng TimetableProblem chứa dữ liệu bài toán (sections, rooms, periods, ...).
            start: Tham số bắt buộc của interface (không sử dụng trong BFS cho TKB).
            goal: Tham số bắt buộc của interface (không sử dụng trong BFS cho TKB).
            **kwargs: max_explored (int) — giới hạn số trạng thái tối đa được duyệt.

        Returns:
            Tuple gồm:
              - path (List[str] | None): Danh sách section_id đã xếp, hoặc None nếu thất bại.
              - cost (float): Tổng vi phạm (hard + soft) của TKB tìm được.
              - info (Dict): Thông tin chi tiết (explored_nodes, execution_time, timetable, ...).
        """
        start_time = time.perf_counter()

        # Kiểm tra đầu vào: BFS chỉ hoạt động với TimetableProblem
        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        # ── MRV Heuristic (Minimum Remaining Values) ──
        # Sắp xếp các lớp HP theo kích thước domain tăng dần.
        # Lớp HP có ít lựa chọn (ca, phòng) hơn → xếp trước.
        # Lý do: Giảm branching factor ở các mức đầu, phát hiện xung đột sớm hơn.
        sections_list = sorted(list(graph.sections.keys()), key=lambda s: len(graph.get_domain_for_section(s)))
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ── Khởi tạo trạng thái gốc ──
        # TKB rỗng: chưa có lớp HP nào được gán vào ca/phòng.
        start_state = TimetableState("", set(), Timetable(), 0.0)

        # ── Hàng đợi FIFO (deque) ──
        # Dùng deque thay list: popleft() có độ phức tạp O(1) thay vì O(n) của list.pop(0).
        queue: deque[TimetableState] = deque([start_state])

        # ── Tập visited ──
        # Lưu các trạng thái đã duyệt dưới dạng frozenset để tránh mở rộng lại
        # cùng một tổ hợp gán lớp HP → tiết kiệm thời gian đáng kể.
        visited: set = set()

        explored_count = 0
        success_state: Optional[TimetableState] = None

        # Giới hạn số trạng thái duyệt tối đa (tránh BFS chạy vô tận với bài toán lớn)
        max_explored = kwargs.get("max_explored", 2000)

        # ══════════════════════════════════════════════════════════════
        # VÒNG LẶP CHÍNH CỦA BFS
        # ══════════════════════════════════════════════════════════════
        while queue:
            # Lấy trạng thái ở đầu hàng đợi (FIFO → duyệt theo chiều rộng)
            current_state = queue.popleft()
            explored_count += 1

            # ── Kiểm tra điều kiện dừng: TKB đã xếp đủ tất cả lớp HP ──
            if current_state.is_complete(len(sections_list)):
                success_state = current_state
                break

            # Kiểm tra giới hạn duyệt
            if 0 < max_explored <= explored_count:
                break

            # ── Kiểm tra trùng lặp trạng thái ──
            # Tạo key duy nhất cho trạng thái hiện tại (dựa trên tổ hợp gán)
            # Nếu đã duyệt rồi → bỏ qua (tránh lặp vô ích).
            state_key = self._state_key(current_state)
            if state_key in visited:
                continue
            visited.add(state_key)

            # ── Chọn lớp HP tiếp theo để xếp lịch ──
            # Dựa vào số lớp đã xếp → lấy lớp kế tiếp trong danh sách đã sắp MRV.
            next_sec_idx = len(current_state.assigned_sections)
            if next_sec_idx >= len(sections_list):
                continue
            next_sec = sections_list[next_sec_idx]

            # ── Lấy domain (tập giá trị khả thi) của lớp HP ──
            # Domain = danh sách các cặp (period_id, room_id) mà lớp này có thể xếp vào.
            domain = graph.get_domain_for_section(next_sec)

            # ── Pruning: domain rỗng → nhánh này chắc chắn thất bại ──
            if not domain:
                continue

            # ── Sinh các trạng thái con (mở rộng nút) ──
            # Với mỗi cặp (ca, phòng) trong domain:
            #   1. Kiểm tra ràng buộc cứng (hard constraints)
            #   2. Nếu hợp lệ → tạo trạng thái con mới và đẩy vào hàng đợi
            for period_id, room_id in domain:
                # Kiểm tra ràng buộc cứng: GV trùng ca, phòng trùng ca, sức chứa, loại phòng, ...
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    # Tạo trạng thái con: sao chép TKB hiện tại + gán thêm lớp HP mới
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id,
                        lecturer_id=graph.sections[next_sec].lecturer_id
                        if next_sec in graph.sections else None
                    )
                    # Đẩy trạng thái con vào CUỐI hàng đợi (đặc trưng FIFO của BFS)
                    queue.append(child_state)

        execution_time = time.perf_counter() - start_time

        # ══════════════════════════════════════════════════════════════
        # ĐÁNH GIÁ KẾT QUẢ
        # ══════════════════════════════════════════════════════════════
        if success_state:
            # Đếm vi phạm ràng buộc cứng (hard) và mềm (soft) của TKB tìm được
            hard_violations = ConstraintChecker.count_hard_violations(
                success_state.timetable,
                graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            soft_violations = ConstraintChecker.count_soft_violations(
                success_state.timetable,
                graph.sections, graph.rooms, graph.periods, graph.lecturers
            )
            # Chi phí = tổng vi phạm cứng + vi phạm mềm (thấp hơn = tốt hơn)
            cost = float(hard_violations + soft_violations["total"])
            return (
                list(success_state.timetable.entries.keys()),
                cost,
                {
                    "explored_nodes": explored_count,
                    "execution_time": execution_time,
                    "timetable": success_state.timetable,
                    "hard_violations": hard_violations,
                    "soft_violations": soft_violations,
                    "fitness": success_state.fitness,
                }
            )
        else:
            # Không tìm thấy lời giải (hết trạng thái hoặc chạm giới hạn duyệt)
            return None, 0.0, {
                "explored_nodes": explored_count,
                "execution_time": execution_time,
                "error": "Không tìm thấy lời giải hợp lệ hoàn chỉnh."
            }

    @staticmethod
    def _state_key(state: TimetableState) -> frozenset:
        """
        Tạo khóa duy nhất (hashable) cho mỗi trạng thái TKB.

        Khóa = frozenset của các bộ (section_id, period_id, room_id) đã gán.
        Dùng frozenset vì:
          - Hashable → có thể đưa vào set/dict để tra cứu O(1).
          - Không phụ thuộc thứ tự → hai trạng thái có cùng tổ hợp gán
            nhưng khác thứ tự gán sẽ được coi là giống nhau.
        """
        entries = state.timetable.entries  # dict: section_id -> Entry
        return frozenset(
            (sec_id, entry.period_id, entry.room_id)
            for sec_id, entry in entries.items()
        )