from typing import Dict, List, Optional, Tuple, Set
import itertools

class Node:
    """Đại diện cho một nút giao thông, kho hàng hoặc điểm giao hàng.
    
    [Lập lịch thi] Tương đương: ExamSlot = (môn × ca × phòng) trong không gian trạng thái.
    """
    def __init__(
        self, 
        node_id: str, 
        x: float, 
        y: float, 
        node_type: str = "intersection", 
        demand: float = 0.0, 
        time_window: Optional[Tuple[float, float]] = None
    ):
        self.id = node_id
        self.x = x
        self.y = y
        self.type = node_type  # "depot", "customer", "intersection"
        self.demand = demand    # Lượng hàng cần giao (dành cho bài toán CSP / VRP)
        self.time_window = time_window  # Khung giờ nhận hàng (start_hour, end_hour)

    def __repr__(self) -> str:
        return f"Node({self.id}, type={self.type}, pos=({self.x}, {self.y}))"


class Edge:
    """Đường nối giữa hai nút giao thông kèm chi phí di chuyển."""
    def __init__(
        self, 
        from_node: str, 
        to_node: str, 
        weight: float, 
        travel_time: float, 
        is_blocked: bool = False
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight            # Độ dài quãng đường thực tế (chi phí)
        self.travel_time = travel_time  # Thời gian di chuyển thực tế
        self.is_blocked = is_blocked    # Trạng thái tắc đường (Complex Environment)

    def __repr__(self) -> str:
        status = " [BLOCKED]" if self.is_blocked else ""
        return f"Edge({self.from_node} -> {self.to_node}, weight={self.weight}{status})"


class Graph:
    """Lớp quản lý đồ thị mạng lưới giao thông."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.adjacency_list: Dict[str, List[Edge]] = {}

    def add_node(self, node: Node) -> None:
        """Thêm một nút vào đồ thị."""
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self.adjacency_list[node.id] = []

    def add_edge(
        self, 
        from_node: str, 
        to_node: str, 
        weight: float, 
        travel_time: float, 
        is_blocked: bool = False
    ) -> None:
        """Thêm cạnh có hướng từ from_node đến to_node."""
        if from_node in self.nodes and to_node in self.nodes:
            edge = Edge(from_node, to_node, weight, travel_time, is_blocked)
            self.adjacency_list[from_node].append(edge)

    def get_neighbors(self, node_id: str) -> List[Edge]:
        """Lấy danh sách các cạnh đi ra từ một nút."""
        return self.adjacency_list.get(node_id, [])

    def get_heuristic(self, node_a_id: str, node_b_id: str) -> float:
        """Tính khoảng cách Euclid giữa 2 nút (Dùng làm heuristic)."""
        node_a = self.nodes.get(node_a_id)
        node_b = self.nodes.get(node_b_id)
        if not node_a or not node_b:
            return 0.0
        import math
        return math.sqrt((node_a.x - node_b.x) ** 2 + (node_a.y - node_b.y) ** 2)


# ══════════════════════════════════════════════════════════════
# PHẦN MỞ RỘNG: TIMETABLE PROBLEM — Không gian bài toán Xếp TKB
# Kế thừa Graph để tái sử dụng toàn bộ cơ sở hạ tầng cũ
# ══════════════════════════════════════════════════════════════

class TimetableProblem(Graph):
    """Biểu diễn không gian bài toán Xếp Thời Khóa Biểu dưới dạng Graph.
    
    Ánh xạ khái niệm:
      Graph       → TimetableProblem (Không gian bài toán)
      Node        → Section          (Lớp học phần cần được xếp ca)
      Edge        → Conflict         (Xung đột giữa 2 lớp học phần - chung GV)
      Edge.weight → Conflict Score   (Mức độ xung đột)
    """

    def __init__(self):
        super().__init__()
        self.courses: Dict[str, object] = {}      # course_id → Course
        self.sections: Dict[str, object] = {}     # section_id → Section
        self.rooms: Dict[str, object] = {}        # room_id → Room
        self.periods: Dict[str, object] = {}      # period_id → Period
        self.lecturers: Dict[str, object] = {}    # lecturer_id → Lecturer

        # Ma trận xung đột: conflict_matrix[s1][s2] = 1 nếu chung GV, 0 otherwise
        self.conflict_matrix: Dict[str, Dict[str, int]] = {}

    def build_conflict_graph(self) -> None:
        """Xây dựng đồ thị xung đột giữa các lớp học phần.
        
        Hai lớp HP xung đột nếu có cùng giảng viên phụ trách.
        Node = Lớp học phần, Edge = xung đột (weight = 1.0)
        """
        sec_list = list(self.sections.values())

        # Thêm mỗi lớp học phần như một Node trong đồ thị
        for sec in sec_list:
            n = Node(
                node_id=sec.section_id,
                x=0.0,
                y=0.0,
                node_type="section",
                demand=float(sec.student_count)  # Tái dùng 'demand' = số SV
            )
            self.add_node(n)

        # Tính xung đột giữa từng cặp lớp học phần
        for s1, s2 in itertools.combinations(sec_list, 2):
            # Xung đột nếu chung giảng viên
            overlap = 1 if s1.lecturer_id == s2.lecturer_id else 0

            # Khởi tạo ma trận xung đột
            if s1.section_id not in self.conflict_matrix:
                self.conflict_matrix[s1.section_id] = {}
            if s2.section_id not in self.conflict_matrix:
                self.conflict_matrix[s2.section_id] = {}

            self.conflict_matrix[s1.section_id][s2.section_id] = overlap
            self.conflict_matrix[s2.section_id][s1.section_id] = overlap

            if overlap > 0:
                # Thêm cạnh xung đột (hai chiều)
                self.add_edge(s1.section_id, s2.section_id, float(overlap), float(overlap))
                self.add_edge(s2.section_id, s1.section_id, float(overlap), float(overlap))

    def get_conflict_score(self, sec_id_a: str, sec_id_b: str) -> int:
        return self.conflict_matrix.get(sec_id_a, {}).get(sec_id_b, 0)

    def get_heuristic(self, sec_id_a: str, sec_id_b: str) -> float:  # type: ignore[override]
        return float(self.get_conflict_score(sec_id_a, sec_id_b))

    def get_domain_for_section(self, section_id: str) -> List[Tuple[str, str]]:
        """Lấy danh sách (period_id, room_id) khả thi cho một lớp học phần (Tối ưu cắt tỉa ca bận của giảng viên)."""
        sec = self.sections.get(section_id)
        if not sec:
            return []
        course = self.courses.get(sec.course_id)
        room_list = list(self.rooms.values())
        if not course:
            return []

        lec = self.lecturers.get(sec.lecturer_id)
        domain = []
        for p_id in self.periods:
            # Tối ưu cắt tỉa: Nếu giảng viên bận ca này, bỏ qua ngay
            if lec and not lec.is_available(p_id):
                continue
            for room in room_list:
                if room.can_host(sec, course):
                    domain.append((p_id, room.room_id))
        return domain

    def count_remaining_conflicts(self, section_id: str, assigned_sections: Set[str]) -> int:
        """Đếm số xung đột của lớp này với các lớp CHƯA được xếp lịch."""
        count = 0
        for other_id, overlap in self.conflict_matrix.get(section_id, {}).items():
            if other_id not in assigned_sections and overlap > 0:
                count += 1
        return count
