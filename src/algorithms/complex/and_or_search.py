import time
from typing import Any, Dict, List, Optional, Tuple, Union
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.timetable_entities import Timetable, ConstraintChecker

# Định nghĩa kiểu kế hoạch điều kiện (Contingency Plan)
Plan = Union[str, Dict[str, Any], List[Any]]

class AndOrSearch(BaseSearchAlgorithm):
    """
    Thuật toán tìm kiếm AND-OR (AND-OR Search).
    Bước 1: Xếp lịch toàn bộ bằng chiến lược tham lam (greedy) để tạo TKB đầy đủ.
    Bước 2: Lập kế hoạch dự phòng (contingency plan) cho lớp học phần mục tiêu.
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

        self.explored_nodes = 0

        # Xác định lớp học phần mục tiêu để lập kế hoạch dự phòng
        sec_id = start
        if sec_id not in graph.sections:
            sec_id = list(graph.sections.keys())[0] if graph.sections else ""

        if not sec_id:
            execution_time = time.perf_counter() - start_time
            return None, 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ── BƯỚC 1: Xếp lịch toàn bộ bằng phương pháp tham lam ──
        timetable = Timetable()
        assigned_count = 0
        for s_id in graph.sections:
            domain = graph.get_domain_for_section(s_id)
            self.explored_nodes += 1
            for period_id, room_id in domain:
                is_valid, _ = ConstraintChecker.check_all_hard(
                    timetable, s_id, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    lec_id = graph.sections[s_id].lecturer_id
                    timetable.assign(s_id, period_id, room_id, lec_id)
                    assigned_count += 1
                    break

        # ── BƯỚC 2: Lập kế hoạch dự phòng AND-OR cho lớp mục tiêu ──
        plan = self._and_or_search(graph, sec_id, timetable)

        # Đường đi mặc định
        default_path = [sec_id]
        if isinstance(plan, dict) and "action" in plan:
            default_path.append(plan["action"])
        elif isinstance(plan, list) and plan:
            default_path.append(plan[0])

        # Tính toán thống kê kết quả
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
            "contingency_plan": plan,
            "next_step": default_path[1] if len(default_path) > 1 else None
        }

    def _and_or_search(self, graph: TimetableProblem, sec_id: str, timetable: Timetable) -> Optional[Plan]:
        """Tìm kế hoạch dự phòng cho lớp học phần mục tiêu."""
        self.explored_nodes += 1

        domain = graph.get_domain_for_section(sec_id)
        if not domain:
            return None

        # Lấy vị trí hiện tại của lớp (nếu đã được xếp)
        current_entry = timetable.get_entry(sec_id)

        # Thử từng gán (Period, Room) → đây là OR Node
        for period_id, room_id in domain[:8]:
            action = f"Assign to {period_id} @ {room_id}"

            # Tạm thời bỏ gán lớp này để kiểm tra khả thi
            timetable.unassign(sec_id)
            is_valid, _ = ConstraintChecker.check_all_hard(
                timetable, sec_id, period_id, room_id,
                graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            # Khôi phục gán ban đầu
            if current_entry:
                timetable.assign(sec_id, current_entry.period_id, current_entry.room_id, current_entry.lecturer_id)

            if not is_valid:
                continue

            # Tìm phương án dự phòng khác cùng ca nhưng khác phòng (AND Node)
            backup_options = [(p, r) for p, r in domain if p == period_id and r != room_id]

            chosen_backup = None
            for p_b, r_b in backup_options:
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
                return {
                    "action": action,
                    "if_success": f"Xếp lịch thành công tại {period_id} @ {room_id}",
                    "if_room_failed_divert_to": chosen_backup,
                    "backup_plan": f"Chuyển sang phòng dự phòng: {chosen_backup}"
                }
            else:
                return [action]

        return None

