from typing import Dict, List, Optional, Set, Tuple

class DeliveryState:
    """Đại diện cho trạng thái hiện tại của xe giao hàng tại một thời điểm."""
    def __init__(
        self, 
        current_node: str, 
        remaining_capacity: float, 
        visited_customers: Set[str],
        accumulated_time: float = 0.0
    ):
        self.current_node = current_node                  # Nút xe đang đứng
        self.remaining_capacity = remaining_capacity      # Tải trọng còn lại của xe
        self.visited_customers = visited_customers        # Tập hợp khách hàng đã giao xong
        self.accumulated_time = accumulated_time          # Thời gian tích lũy đã chạy

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeliveryState):
            return False
        return (self.current_node == other.current_node and 
                self.remaining_capacity == other.remaining_capacity and 
                self.visited_customers == other.visited_customers)

    def __hash__(self) -> int:
        return hash((self.current_node, self.remaining_capacity, frozenset(self.visited_customers)))

    def __repr__(self) -> str:
        return (f"DeliveryState(At={self.current_node}, Capacity={self.remaining_capacity}, "
                f"Visited={len(self.visited_customers)}, Time={self.accumulated_time:.2f})")


# ══════════════════════════════════════════════════════════════
# TIMETABLE STATE — Trạng thái Xếp Thời Khóa Biểu
# Tương đương: DeliveryState nhưng cho bài toán TKB
# ══════════════════════════════════════════════════════════════
from src.core.timetable_entities import Timetable, ConstraintChecker

class TimetableState:
    """Trạng thái hiện tại trong quá trình xây dựng thời khóa biểu.
    
    Tương đương: DeliveryState trong bài toán giao hàng.
    
    Ánh xạ khái niệm:
      current_node      → current_section  (Lớp HP đang xét xếp lịch)
      remaining_capacity → unassigned_count (Số lớp HP chưa xếp lịch)
      visited_customers → assigned_sections (Tập lớp HP đã được xếp lịch)
      accumulated_time  → violation_score  (Tổng điểm vi phạm ràng buộc)
    """

    def __init__(
        self,
        current_section: str,
        assigned_sections: Set[str],
        timetable: Optional[Timetable] = None,
        violation_score: float = 0.0
    ):
        self.current_section = current_section            # Lớp HP đang xét
        self.assigned_sections = assigned_sections        # Tập lớp HP đã xếp
        self.timetable = timetable or Timetable()         # Đối tượng Timetable lưu gán
        self.violation_score = violation_score            # Tổng điểm vi phạm (thấp hơn = tốt hơn)

    @property
    def assigned_count(self) -> int:
        """Số lớp HP đã được xếp lịch."""
        return len(self.assigned_sections)

    @property
    def fitness(self) -> float:
        """Hàm đánh giá chất lượng thời khóa biểu (cao hơn = tốt hơn).
        Tương đương: -Route Cost trong bài toán giao hàng.
        """
        return self.assigned_count * 10.0 - self.violation_score

    def is_complete(self, total_sections: int) -> bool:
        """Kiểm tra thời khóa biểu đã xếp xong toàn bộ các lớp HP chưa."""
        return len(self.assigned_sections) == total_sections

    def copy_with_assignment(
        self,
        section_id: str,
        period_id: str,
        room_id: str,
        extra_violations: float = 0.0,
        lecturer_id: Optional[str] = None
    ) -> "TimetableState":
        """Tạo trạng thái mới sau khi xếp lớp này vào (period, room).
        Tương đương: Tạo Node mới trong lộ trình giao hàng.
        """
        new_assigned = set(self.assigned_sections) | {section_id}
        new_timetable = self.timetable.copy()
        new_timetable.assign(section_id, period_id, room_id, lecturer_id)
        return TimetableState(
            current_section=section_id,
            assigned_sections=new_assigned,
            timetable=new_timetable,
            violation_score=self.violation_score + extra_violations
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimetableState):
            return False
        return (self.current_section == other.current_section and
                self.assigned_sections == other.assigned_sections and
                self.timetable.to_dict() == other.timetable.to_dict())

    def __hash__(self) -> int:
        return hash((self.current_section, frozenset(self.assigned_sections)))

    def __repr__(self) -> str:
        return (f"TimetableState(Current={self.current_section}, "
                f"Assigned={self.assigned_count} lớp, "
                f"Violations={self.violation_score:.2f})")
