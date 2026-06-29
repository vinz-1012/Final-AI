# ── ÁP DỤNG CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU ──
# - Trạng thái: TKB bán hoàn chỉnh (giống BFS/DFS/A*).
# - Heuristic cơ bản: Đếm xung đột tiềm năng của lớp HP chưa xếp.
# - Learning: Cập nhật H table sau mỗi bước gán lớp HP.
# - Online/Dynamic: Mô phỏng thay đổi môi trường (phòng đóng/mở bất ngờ)
#   → agent phải thích nghi real-time.
#
# ── MÔ PHỎNG THAY ĐỔI MÔI TRƯỜNG ──
# Với xác suất 10% tại mỗi bước, 1 phòng học ngẫu nhiên sẽ bị đổi trạng thái
# (mở → đóng hoặc đóng → mở). Điều này mô phỏng:
#   - Phòng bảo trì đột xuất.
#   - Phòng hết bảo trì và khả dụng lại.
#   - Sự cố thiết bị / mất điện.
# Agent phải tiếp tục xếp lịch trong điều kiện biến động này.
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Set
from src.core.base_search import BaseSearchAlgorithm
from src.core.graph import Graph, TimetableProblem
from src.core.state import TimetableState
from src.core.timetable_entities import Timetable, ConstraintChecker

class LrtaStarSearch(BaseSearchAlgorithm):
    """
    LRTA* — Learning Real-Time A* cho bài toán xếp TKB trong môi trường biến động.

    Nguyên lý hoạt động:
      1. Bắt đầu từ TKB rỗng.
      2. Tại mỗi bước:
         a. Tính heuristic cho trạng thái hiện tại (hoặc tra H table).
         b. Chọn lớp HP tiếp theo → duyệt domain → tìm successor tốt nhất.
         c. Cập nhật H[current] = min(f(successors)).
         d. Di chuyển sang successor tốt nhất.
         e. [10% xác suất] Mô phỏng thay đổi môi trường (phòng đóng/mở).
      3. Dừng khi TKB đầy đủ hoặc hết bước lặp.

    Ưu điểm: Real-time, adaptive, xử lý được môi trường biến động.
    Nhược điểm: Có thể không tìm được lời giải nếu thay đổi quá nhiều.
    """

    def __init__(self):
        # ── Bảng Heuristic đã học ──
        # H table: {hash(state) → h_value}
        # Ban đầu rỗng → heuristic được tính lần đầu bằng _get_heuristic().
        # Sau đó cập nhật dần qua các bước → "học" từ kinh nghiệm.
        # H table PERSISTS giữa các lần gọi search() → học qua nhiều episode.
        self.H: Dict[int, float] = {}

    def _get_heuristic(self, state: TimetableState, graph: TimetableProblem) -> float:
        """
        Hàm heuristic cơ bản h(n): Ước lượng xung đột còn lại.
        Được dùng khi trạng thái CHƯA CÓ trong H table (chưa học).
        Giống hàm heuristic của A* và Greedy.
        """
        unassigned = [s for s in graph.sections if s not in state.assigned_sections]
        h_val = 0.0
        for s in unassigned:
            h_val += graph.count_remaining_conflicts(s, state.assigned_sections)
        return h_val

    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện LRTA* trên bài toán xếp TKB với mô phỏng thay đổi môi trường.

        Args:
            graph: Đối tượng TimetableProblem.
            **kwargs:
                max_steps (int): Số bước tối đa (mặc định 200).
                dynamic_changes (bool): Bật/tắt mô phỏng thay đổi môi trường (mặc định True).
        """
        start_time = time.perf_counter()
        
        if not isinstance(graph, TimetableProblem):
            return None, 0.0, {"error": "Graph must be a TimetableProblem"}

        sections_list = list(graph.sections.keys())
        if not sections_list:
            execution_time = time.perf_counter() - start_time
            return [], 0.0, {"explored_nodes": 0, "execution_time": execution_time}

        # ── Trạng thái ban đầu: TKB rỗng ──
        current_state = TimetableState("", set(), Timetable(), 0.0)
        steps = 0
        max_steps = kwargs.get("max_steps", 200)
        dynamic_changes = kwargs.get("dynamic_changes", True)  # Bật mô phỏng thay đổi
        
        # ══════════════════════════════════════════════════════════════
        # VÒNG LẶP CHÍNH CỦA LRTA*
        # ══════════════════════════════════════════════════════════════
        # Mỗi bước = 1 quyết định real-time (gán 1 lớp HP vào ca/phòng).
        while not current_state.is_complete(len(sections_list)) and steps < max_steps:
            steps += 1

            # ── Bước 1: Tra hoặc tính heuristic cho trạng thái hiện tại ──
            curr_hash = hash(current_state)
            if curr_hash not in self.H:
                # Trạng thái mới (chưa học) → tính h bằng heuristic cơ bản
                self.H[curr_hash] = self._get_heuristic(current_state, graph)
                
            # ── Bước 2: Chọn lớp HP tiếp theo để xếp ──
            next_sec_idx = len(current_state.assigned_sections)
            next_sec = sections_list[next_sec_idx]
            
            # ── Bước 3: Duyệt domain → sinh successor hợp lệ ──
            domain = graph.get_domain_for_section(next_sec)
            valid_successors = []
            
            for period_id, room_id in domain:
                is_valid, _ = ConstraintChecker.check_all_hard(
                    current_state.timetable, next_sec, period_id, room_id,
                    graph.sections, graph.courses, graph.rooms, graph.lecturers
                )
                if is_valid:
                    # Tính chi phí chuyển đổi (cost) = vi phạm mềm mới
                    temp_timetable = current_state.timetable.copy()
                    lec_id = graph.sections[next_sec].lecturer_id if next_sec in graph.sections else None
                    temp_timetable.assign(next_sec, period_id, room_id, lec_id)
                    soft_violations = ConstraintChecker.count_soft_violations(
                        temp_timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
                    )
                    g_val = soft_violations["total"]
                    
                    # Tạo trạng thái con
                    child_state = current_state.copy_with_assignment(
                        next_sec, period_id, room_id, extra_violations=g_val - current_state.violation_score,
                        lecturer_id=lec_id
                    )
                    valid_successors.append((period_id, room_id, child_state))
                    
            if not valid_successors:
                break  # Không có successor hợp lệ → kẹt
                
            # ── Bước 4: Tìm successor tối thiểu hóa f(s') = cost + H(s') ──
            # Đây là quy tắc quyết định cốt lõi của LRTA*.
            best_successor = None
            min_f = float('inf')
            
            for p_id, r_id, child in valid_successors:
                child_hash = hash(child)
                if child_hash not in self.H:
                    # Tính heuristic cho trạng thái chưa học
                    self.H[child_hash] = self._get_heuristic(child, graph)
                # f(s') = cost(s → s') + H(s')
                cost = child.violation_score - current_state.violation_score
                f_val = cost + self.H[child_hash]
                
                if f_val < min_f:
                    min_f = f_val
                    best_successor = child
                    
            # ── Bước 5: CẬP NHẬT HEURISTIC (LEARNING) ──
            # H(current) ← min_f
            # Ý nghĩa: "Từ trạng thái hiện tại, chi phí tối thiểu ước lượng là min_f."
            # Đây là bước QUAN TRỌNG NHẤT phân biệt LRTA* với các thuật toán khác.
            # Qua nhiều lần giải, H hội tụ dần về giá trị thực → tìm đường tốt hơn.
            self.H[curr_hash] = min_f
            
            # ── Bước 6: Di chuyển sang successor tốt nhất ──
            if best_successor:
                current_state = best_successor
            else:
                break
                
            # ══════════════════════════════════════════════════════════
            # MÔ PHỎNG THAY ĐỔI MÔI TRƯỜNG (DYNAMIC ENVIRONMENT)
            # ══════════════════════════════════════════════════════════
            # Với xác suất 10%, 1 phòng học ngẫu nhiên thay đổi trạng thái.
            # Mô phỏng tình huống:
            #   - Phòng đang mở → đóng cửa (sự cố, bảo trì).
            #   - Phòng đang đóng → mở lại (hết bảo trì).
            # Điều này tạo ra môi trường KHÔNG XÁC ĐỊNH (non-deterministic)
            # mà agent phải thích nghi trong thời gian thực.
            if dynamic_changes and random.random() < 0.1:
                rooms_list = list(graph.rooms.values())
                if rooms_list:
                    r = random.choice(rooms_list)
                    r.is_available = not r.is_available  # Toggle trạng thái phòng

        execution_time = time.perf_counter() - start_time
        success = current_state.is_complete(len(sections_list))
        
        # ── Khôi phục trạng thái phòng về ban đầu ──
        # Quan trọng: Đảm bảo không ảnh hưởng đến các thuật toán khác chạy sau.
        for room in graph.rooms.values():
            room.is_available = True
            
        # ══════════════════════════════════════════════════════════════
        # ĐÁNH GIÁ KẾT QUẢ
        # ══════════════════════════════════════════════════════════════
        if success:
            hard_violations = ConstraintChecker.count_hard_violations(
                current_state.timetable, graph.sections, graph.courses, graph.rooms, graph.lecturers
            )
            soft_violations = ConstraintChecker.count_soft_violations(
                current_state.timetable, graph.sections, graph.rooms, graph.periods, graph.lecturers
            )
            cost = float(hard_violations + soft_violations["total"])
            return (
                list(current_state.timetable.entries.keys()),
                cost,
                {
                    "explored_nodes": steps,
                    "execution_time": execution_time,
                    "timetable": current_state.timetable,
                    "hard_violations": hard_violations,
                    "soft_violations": soft_violations,
                    "fitness": current_state.fitness,
                    "learned_heuristics_count": len(self.H)  # Số trạng thái đã học
                }
            )
        else:
            return None, 0.0, {
                "explored_nodes": steps,
                "execution_time": execution_time,
                "learned_heuristics_count": len(self.H),
                "error": "Không tìm thấy thời khóa biểu trực tuyến do kẹt hoặc thay đổi môi trường."
            }
