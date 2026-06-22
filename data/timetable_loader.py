import json
from src.core.graph import TimetableProblem
from src.core.timetable_entities import Course, Section, Lecturer, Room, Period

class TimetableLoader:
    """Đọc và chuyển đổi dữ liệu thời khóa biểu dạng JSON thành cấu trúc TimetableProblem."""

    @staticmethod
    def load_from_json(file_path: str) -> TimetableProblem:
        """Đọc file JSON và trả về một đối tượng TimetableProblem hoàn chỉnh."""
        problem = TimetableProblem()
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Đọc periods
        for p_data in data.get("periods", []):
            period = Period(
                period_id=p_data["period_id"],
                day=p_data["day"],
                day_index=p_data["day_index"],
                slot=p_data["slot"],
                start_time=p_data["start_time"],
                end_time=p_data["end_time"],
                session=p_data["session"]
            )
            problem.periods[period.period_id] = period

        # 2. Đọc rooms
        for r_data in data.get("rooms", []):
            room = Room(
                room_id=r_data["room_id"],
                capacity=r_data["capacity"],
                room_type=r_data.get("room_type", "lecture"),
                building=r_data.get("building", "A"),
                floor=r_data.get("floor", 1),
                is_available=r_data.get("is_available", True)
            )
            problem.rooms[room.room_id] = room

        # 3. Đọc lecturers
        for l_data in data.get("lecturers", []):
            lecturer = Lecturer(
                lecturer_id=l_data["lecturer_id"],
                name=l_data["name"],
                department=l_data.get("department", ""),
                unavailable_periods=l_data.get("unavailable_periods", []),
                max_periods_per_day=l_data.get("max_periods_per_day", 2),
                max_periods_per_week=l_data.get("max_periods_per_week", 12)
            )
            problem.lecturers[lecturer.lecturer_id] = lecturer

        # 4. Đọc courses
        for c_data in data.get("courses", []):
            course = Course(
                course_id=c_data["course_id"],
                name=c_data["name"],
                credit=c_data.get("credit", 3),
                room_type=c_data.get("room_type", "lecture"),
                department=c_data.get("department", ""),
                color=c_data.get("color", "#4A90D9")
            )
            problem.courses[course.course_id] = course

        # 5. Đọc sections
        for s_data in data.get("sections", []):
            section = Section(
                section_id=s_data["section_id"],
                course_id=s_data["course_id"],
                lecturer_id=s_data["lecturer_id"],
                student_count=s_data.get("student_count", 40),
                group_name=s_data.get("group_name", "Nhóm 1")
            )
            problem.sections[section.section_id] = section

        # Xây dựng đồ thị xung đột giữa các lớp học phần
        problem.build_conflict_graph()

        return problem
