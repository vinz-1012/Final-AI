"""
timetable_entities.py
----------------------
Định nghĩa các thực thể cốt lõi của Hệ thống Xếp Thời Khóa Biểu Tự động.

Chuẩn Việt Nam: mỗi học phần gặp 1 buổi/tuần, mỗi buổi 3 tiết liên tục.
Lưới thời gian: Thứ 2–6, 4 ca/ngày = 20 ca học/tuần.

Ánh xạ khái niệm từ bài toán giao hàng:
  Customer Node  → Section     (Lớp học phần cần được xếp ca)
  Vehicle        → Room        (Phòng học là tài nguyên)
  Route          → Timetable   (Thời khóa biểu hoàn chỉnh)
  Time Window    → Period      (Ca học trong tuần)
  Capacity       → Room.capacity
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────
# HẰNG SỐ LƯỚI THỜI GIAN
# ─────────────────────────────────────────────────
DAYS = ["MON", "TUE", "WED", "THU", "FRI"]
DAY_NAMES_VI = {
    "MON": "Thứ 2", "TUE": "Thứ 3", "WED": "Thứ 4",
    "THU": "Thứ 5", "FRI": "Thứ 6"
}
# 4 ca mỗi ngày (chuẩn trường ĐH Việt Nam)
SLOT_TIMES = {
    1: {"label": "Ca 1 (Sáng 1)", "start": "07:00", "end": "09:25",  "periods": "Tiết 1–3"},
    2: {"label": "Ca 2 (Sáng 2)", "start": "09:35", "end": "12:00",  "periods": "Tiết 4–6"},
    3: {"label": "Ca 3 (Chiều 1)", "start": "12:35", "end": "15:00", "periods": "Tiết 7–9"},
    4: {"label": "Ca 4 (Chiều 2)", "start": "15:10", "end": "17:35", "periods": "Tiết 10–12"},
}

# ─────────────────────────────────────────────────
# 1. CÁC THỰC THỂ CƠ BẢN
# ─────────────────────────────────────────────────

@dataclass
class Course:
    """Học phần (môn học).

    Tương đương: Customer trong bài toán giao hàng.
    Đây là 'loại biến' cần được xếp lịch.
    """
    course_id: str              # VD: "CS101"
    name: str                   # VD: "Nhập môn Lập trình"
    credit: int = 3             # Số tín chỉ
    room_type: str = "lecture"  # "lecture" | "lab" | "seminar"
    department: str = ""        # Khoa phụ trách
    color: str = "#4A90D9"      # Màu hiển thị trên TKB

    def __hash__(self) -> int:
        return hash(self.course_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Course) and self.course_id == other.course_id

    def __repr__(self) -> str:
        return f"Course({self.course_id}, '{self.name}', {self.credit}TC)"


@dataclass
class Section:
    """Lớp học phần — nhóm sinh viên học cùng 1 ca.

    Tương đương: một 'đơn hàng' (delivery order) trong bài toán giao hàng.
    Đây là 'biến' (variable) trong bài toán CSP cần được gán giá trị.

    Mỗi Section cần được gán đúng 1 (Period, Room) mỗi tuần.
    """
    section_id: str             # VD: "CS101_N01" (Môn CS101, Nhóm 01)
    course_id: str              # Thuộc học phần nào
    lecturer_id: str            # Giảng viên phụ trách
    student_count: int = 40     # Số sinh viên
    group_name: str = "Nhóm 1"  # Tên nhóm

    def __hash__(self) -> int:
        return hash(self.section_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Section) and self.section_id == other.section_id

    def __repr__(self) -> str:
        return f"Section({self.section_id}, GV={self.lecturer_id}, {self.student_count}SV)"


@dataclass
class Lecturer:
    """Giảng viên.

    Ràng buộc cứng: Một GV không dạy 2 lớp cùng ca.
    """
    lecturer_id: str                                            # VD: "GV001"
    name: str
    department: str = ""
    unavailable_periods: List[str] = field(default_factory=list)  # Ca không thể dạy
    max_periods_per_day: int = 2                                # Tối đa ca/ngày
    max_periods_per_week: int = 12                              # Tối đa ca/tuần

    def is_available(self, period_id: str) -> bool:
        return period_id not in self.unavailable_periods

    def __hash__(self) -> int:
        return hash(self.lecturer_id)

    def __repr__(self) -> str:
        return f"Lecturer({self.lecturer_id}, '{self.name}')"


@dataclass
class Room:
    """Phòng học — tài nguyên không gian.

    Tương đương: Vehicle (xe) trong bài toán giao hàng.
    Đây là phần 'domain' trong bài toán CSP.
    """
    room_id: str                # VD: "A101"
    capacity: int               # Số chỗ ngồi
    room_type: str = "lecture"  # "lecture" | "lab" | "seminar"
    building: str = "A"         # Tòa nhà
    floor: int = 1
    is_available: bool = True

    def can_host(self, section: Section, course: Course) -> bool:
        """Kiểm tra phòng có thể tổ chức lớp học phần này không."""
        return (self.is_available and
                self.room_type == course.room_type and
                self.capacity >= section.student_count)

    def __hash__(self) -> int:
        return hash(self.room_id)

    def __repr__(self) -> str:
        return f"Room({self.room_id}, {self.capacity} chỗ, {self.room_type})"


@dataclass
class Period:
    """Ca học trong tuần — tài nguyên thời gian.

    Tương đương: Time Window trong bài toán giao hàng.
    Lưới: 5 ngày × 4 ca = 20 ca học/tuần.
    """
    period_id: str      # VD: "MON_1" (Thứ 2, Ca 1)
    day: str            # "MON" | "TUE" | "WED" | "THU" | "FRI"
    day_index: int      # 1–5 (Thứ 2=1, Thứ 6=5)
    slot: int           # 1–4 (Ca sáng 1, Ca sáng 2, Ca chiều 1, Ca chiều 2)
    start_time: str     # VD: "07:00"
    end_time: str       # VD: "09:25"
    session: str        # "morning" | "afternoon"

    @property
    def day_name(self) -> str:
        return DAY_NAMES_VI.get(self.day, self.day)

    @property
    def slot_label(self) -> str:
        return SLOT_TIMES.get(self.slot, {}).get("label", f"Ca {self.slot}")

    @property
    def display_label(self) -> str:
        return f"{self.day_name}\n{self.slot_label}"

    def __hash__(self) -> int:
        return hash(self.period_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Period) and self.period_id == other.period_id

    def __repr__(self) -> str:
        return f"Period({self.period_id}: {self.day_name} {self.start_time}–{self.end_time})"


# ─────────────────────────────────────────────────
# 2. PHÂN CÔNG & THỜI KHÓA BIỂU
# ─────────────────────────────────────────────────

@dataclass
class TimetableEntry:
    """Một ô trong thời khóa biểu: lớp học phần → (ca, phòng).

    Tương đương: Assignment trong CSP / Node trong lộ trình giao hàng.
    """
    section_id: str
    period_id: str
    room_id: str
    lecturer_id: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.section_id, self.period_id, self.room_id, self.lecturer_id))

    def __repr__(self) -> str:
        return f"Entry({self.section_id} → {self.period_id} @ {self.room_id} (GV: {self.lecturer_id}))"


class Timetable:
    """Thời khóa biểu hoàn chỉnh — lặp lại hàng tuần.

    Tương đương: Route (lộ trình) trong bài toán giao hàng.
    Đây là 'lời giải' của bài toán CSP.

    Mỗi lớp học phần được gán đúng 1 (Period, Room) mỗi tuần.
    """

    def __init__(self):
        # Ánh xạ: section_id → TimetableEntry
        self.entries: Dict[str, TimetableEntry] = {}
        # Bảng tra cứu nhanh phục vụ tối ưu O(1) kiểm tra ràng buộc
        self.room_occupancy: Dict[Tuple[str, str], str] = {}      # (period_id, room_id) -> section_id
        self.lecturer_occupancy: Dict[Tuple[str, str], str] = {}  # (period_id, lecturer_id) -> section_id

    def assign(self, section_id: str, period_id: str, room_id: str, lecturer_id: Optional[str] = None) -> None:
        """Xếp lớp học phần vào (ca, phòng)."""
        # Hủy gán cũ nếu có
        old_entry = self.entries.get(section_id)
        if old_entry:
            self.room_occupancy.pop((old_entry.period_id, old_entry.room_id), None)
            if old_entry.lecturer_id:
                self.lecturer_occupancy.pop((old_entry.period_id, old_entry.lecturer_id), None)

        self.entries[section_id] = TimetableEntry(section_id, period_id, room_id, lecturer_id)
        self.room_occupancy[(period_id, room_id)] = section_id
        if lecturer_id:
            self.lecturer_occupancy[(period_id, lecturer_id)] = section_id

    def unassign(self, section_id: str) -> None:
        """Hủy xếp lịch của lớp học phần."""
        old_entry = self.entries.pop(section_id, None)
        if old_entry:
            self.room_occupancy.pop((old_entry.period_id, old_entry.room_id), None)
            if old_entry.lecturer_id:
                self.lecturer_occupancy.pop((old_entry.period_id, old_entry.lecturer_id), None)

    def get_entry(self, section_id: str) -> Optional[TimetableEntry]:
        return self.entries.get(section_id)

    def is_assigned(self, section_id: str) -> bool:
        return section_id in self.entries

    @property
    def assigned_count(self) -> int:
        return len(self.entries)

    def get_period_id(self, section_id: str) -> Optional[str]:
        e = self.entries.get(section_id)
        return e.period_id if e else None

    def get_room_id(self, section_id: str) -> Optional[str]:
        e = self.entries.get(section_id)
        return e.room_id if e else None

    def sections_in_period(self, period_id: str) -> List[str]:
        return [sid for sid, e in self.entries.items() if e.period_id == period_id]

    def sections_in_room_period(self, room_id: str, period_id: str) -> List[str]:
        return [sid for sid, e in self.entries.items()
                if e.room_id == room_id and e.period_id == period_id]

    def copy(self) -> "Timetable":
        t = Timetable()
        for sid, e in self.entries.items():
            t.entries[sid] = TimetableEntry(e.section_id, e.period_id, e.room_id, e.lecturer_id)
        t.room_occupancy = dict(self.room_occupancy)
        t.lecturer_occupancy = dict(self.lecturer_occupancy)
        return t

    def to_dict(self) -> Dict:
        return {
            sid: {"period_id": e.period_id, "room_id": e.room_id}
            for sid, e in self.entries.items()
        }

    def __repr__(self) -> str:
        return f"Timetable({self.assigned_count}/{self.assigned_count} lớp đã xếp)"


# ─────────────────────────────────────────────────
# 3. KIỂM TRA RÀNG BUỘC
# ─────────────────────────────────────────────────

class ConstraintChecker:
    """Kiểm tra ràng buộc của thời khóa biểu.

    Hard Constraints (PHẢI tuân thủ):
      HC1: GV không dạy 2 lớp cùng ca
      HC2: Phòng chỉ có 1 lớp trong 1 ca
      HC3: Phòng đủ sức chứa cho lớp
      HC4: Loại phòng phù hợp với học phần (lab/lecture/seminar)
      HC5: GV phải có mặt (không nằm trong unavailable_periods)

    Soft Constraints (NÊN tuân thủ):
      SC1: GV không dạy quá max_periods_per_day ca/ngày
      SC2: Tránh tiết trống (gaps) trong lịch GV
      SC3: Phân bổ ca học đều trong tuần
      SC4: Ưu tiên cùng tòa nhà nếu GV dạy liên tiếp
    """

    # ── HARD CONSTRAINTS ──

    @staticmethod
    def hc1_lecturer_conflict(
        timetable: Timetable,
        section_id: str, period_id: str,
        sections: Dict[str, Section]
    ) -> bool:
        """HC1: GV không dạy 2 lớp cùng ca. True = hợp lệ."""
        sec = sections.get(section_id)
        if not sec:
            return True
            
        # Sử dụng lookup table O(1) nếu khả dụng
        if hasattr(timetable, 'lecturer_occupancy') and timetable.lecturer_occupancy:
            occupied_sec_id = timetable.lecturer_occupancy.get((period_id, sec.lecturer_id))
            if occupied_sec_id is not None and occupied_sec_id != section_id:
                return False
            return True
            
        # Fallback O(N)
        for sid, e in timetable.entries.items():
            if e.period_id == period_id and sid != section_id:
                other = sections.get(sid)
                if other and other.lecturer_id == sec.lecturer_id:
                    return False
        return True

    @staticmethod
    def hc2_room_conflict(
        timetable: Timetable,
        period_id: str, room_id: str
    ) -> bool:
        """HC2: Phòng chỉ có 1 lớp trong 1 ca. True = hợp lệ."""
        # Sử dụng lookup table O(1) nếu khả dụng
        if hasattr(timetable, 'room_occupancy') and timetable.room_occupancy:
            return (period_id, room_id) not in timetable.room_occupancy
            
        # Fallback O(N)
        for e in timetable.entries.values():
            if e.period_id == period_id and e.room_id == room_id:
                return False
        return True

    @staticmethod
    def hc3_room_capacity(
        section: Section, room: Room
    ) -> bool:
        """HC3: Phòng đủ sức chứa. True = hợp lệ."""
        return room.capacity >= section.student_count

    @staticmethod
    def hc4_room_type(
        course: Course, room: Room
    ) -> bool:
        """HC4: Loại phòng phù hợp. True = hợp lệ."""
        return room.room_type == course.room_type

    @staticmethod
    def hc5_lecturer_available(
        lecturer: Lecturer, period_id: str
    ) -> bool:
        """HC5: GV không bận vào ca này. True = hợp lệ."""
        return lecturer.is_available(period_id)

    @classmethod
    def check_all_hard(
        cls,
        timetable: Timetable,
        section_id: str,
        period_id: str,
        room_id: str,
        sections: Dict[str, Section],
        courses: Dict[str, Course],
        rooms: Dict[str, Room],
        lecturers: Dict[str, Lecturer],
    ) -> Tuple[bool, List[str]]:
        """Kiểm tra toàn bộ ràng buộc cứng cho 1 phân công mới."""
        violations: List[str] = []
        sec = sections.get(section_id)
        if not sec:
            return False, ["Section not found"]
        course = courses.get(sec.course_id)
        room = rooms.get(room_id)
        lec = lecturers.get(sec.lecturer_id)
        if not course or not room:
            return False, ["Course or Room not found"]

        if not cls.hc1_lecturer_conflict(timetable, section_id, period_id, sections):
            violations.append("HC1: GV đang dạy lớp khác cùng ca")
        if not cls.hc2_room_conflict(timetable, period_id, room_id):
            violations.append("HC2: Phòng đã có lớp khác")
        if not cls.hc3_room_capacity(sec, room):
            violations.append(f"HC3: Phòng chỉ chứa {room.capacity} < {sec.student_count} SV")
        if not cls.hc4_room_type(course, room):
            violations.append(f"HC4: Phòng type '{room.room_type}' ≠ '{course.room_type}'")
        if lec and not cls.hc5_lecturer_available(lec, period_id):
            violations.append("HC5: GV bận vào ca này")

        return len(violations) == 0, violations

    @classmethod
    def count_hard_violations(
        cls,
        timetable: Timetable,
        sections: Dict[str, Section],
        courses: Dict[str, Course],
        rooms: Dict[str, Room],
        lecturers: Dict[str, Lecturer],
    ) -> int:
        """Đếm tổng số vi phạm ràng buộc cứng trong toàn bộ TKB."""
        violations = 0
        
        # Kiểm tra trùng phòng và trùng giảng viên theo O(N) sử dụng hash map
        room_occupancy = {}
        lecturer_occupancy = {}
        
        for e in timetable.entries.values():
            # Trùng phòng
            key_room = (e.period_id, e.room_id)
            room_occupancy[key_room] = room_occupancy.get(key_room, 0) + 1
            
            # Trùng giảng viên
            sec = sections.get(e.section_id)
            if sec:
                key_lec = (e.period_id, sec.lecturer_id)
                lecturer_occupancy[key_lec] = lecturer_occupancy.get(key_lec, 0) + 1
                
        for count in room_occupancy.values():
            if count > 1:
                violations += (count - 1)
                
        for count in lecturer_occupancy.values():
            if count > 1:
                violations += (count - 1)
                
        # HC3, HC4, HC5
        for sid, e in timetable.entries.items():
            sec = sections.get(sid)
            if not sec:
                continue
            room = rooms.get(e.room_id)
            course = courses.get(sec.course_id)
            lec = lecturers.get(sec.lecturer_id)
            if room and course:
                if not cls.hc3_room_capacity(sec, room):
                    violations += 1
                if not cls.hc4_room_type(course, room):
                    violations += 1
            if lec and not cls.hc5_lecturer_available(lec, e.period_id):
                violations += 1
        return violations

    @classmethod
    def count_soft_violations(
        cls,
        timetable: Timetable,
        sections: Dict[str, Section],
        rooms: Dict[str, Room],
        periods: Dict[str, Period],
        lecturers: Dict[str, Lecturer],
    ) -> Dict[str, float]:
        """Đếm vi phạm ràng buộc mềm (Tối ưu hóa chạy trong 1 vòng lặp duy nhất)."""
        v: Dict[str, float] = {"SC1": 0, "SC2": 0, "SC3": 0, "total": 0.0}

        lec_day_count: Dict[Tuple[str, int], int] = {}
        lec_slots: Dict[str, Dict[int, List[int]]] = {}
        day_counts: Dict[int, int] = {}

        for sid, e in timetable.entries.items():
            sec = sections.get(sid)
            period = periods.get(e.period_id)
            if not period:
                continue
            
            # SC3: Phân bổ ca học đều trong tuần
            day_counts[period.day_index] = day_counts.get(period.day_index, 0) + 1

            if sec:
                lec_id = sec.lecturer_id
                day_idx = period.day_index
                
                # SC1
                key_sc1 = (lec_id, day_idx)
                lec_day_count[key_sc1] = lec_day_count.get(key_sc1, 0) + 1
                
                # SC2
                if lec_id not in lec_slots:
                    lec_slots[lec_id] = {}
                if day_idx not in lec_slots[lec_id]:
                    lec_slots[lec_id][day_idx] = []
                lec_slots[lec_id][day_idx].append(period.slot)

        # Hậu xử lý SC1
        for (lec_id, day_idx), count in lec_day_count.items():
            lec = lecturers.get(lec_id)
            if lec and count > lec.max_periods_per_day:
                v["SC1"] += count - lec.max_periods_per_day

        # Hậu xử lý SC2
        for lec_id, day_dict in lec_slots.items():
            for day, slots in day_dict.items():
                if len(slots) >= 2:
                    slots.sort()
                    gaps = sum(slots[i+1] - slots[i] - 1 for i in range(len(slots)-1))
                    v["SC2"] += gaps

        # Hậu xử lý SC3
        if day_counts:
            avg = sum(day_counts.values()) / len(day_counts)
            v["SC3"] = sum(abs(c - avg) for c in day_counts.values())

        v["total"] = v["SC1"] * 2 + v["SC2"] * 1 + v["SC3"] * 0.5
        return v

    @classmethod
    def compute_fitness(
        cls,
        timetable: Timetable,
        sections: Dict[str, Section],
        courses: Dict[str, Course],
        rooms: Dict[str, Room],
        periods: Dict[str, Period],
        lecturers: Dict[str, Lecturer],
        w_hard: float = 1000.0,
        w_soft: float = 1.0,
    ) -> float:
        """Tính fitness của TKB (càng cao càng tốt).

        fitness = coverage × 10 - hard_penalty × w_hard - soft_penalty × w_soft
        Tương đương: -Route Cost trong bài toán giao hàng.
        """
        hard = cls.count_hard_violations(timetable, sections, courses, rooms, lecturers)
        soft = cls.count_soft_violations(timetable, sections, rooms, periods, lecturers)
        return timetable.assigned_count * 10.0 - hard * w_hard - soft["total"] * w_soft
