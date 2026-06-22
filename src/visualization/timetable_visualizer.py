import pandas as pd
from typing import Dict, List, Optional
from src.core.timetable_entities import Timetable, DAYS, SLOT_TIMES, DAY_NAMES_VI
from src.core.graph import TimetableProblem

class TimetableVisualizer:
    """Hỗ trợ trực quan hóa thời khóa biểu dưới dạng bảng điều khiển và in ấn."""

    @staticmethod
    def to_dataframe(timetable: Timetable, graph: TimetableProblem) -> pd.DataFrame:
        """Chuyển đổi thời khóa biểu thành Pandas DataFrame dạng lưới (Ca học x Thứ)."""
        # Khởi tạo ma trận trống
        grid = {day: {slot: "" for slot in SLOT_TIMES.keys()} for day in DAYS}
        
        for sec_id, entry in timetable.entries.items():
            sec = graph.sections.get(sec_id)
            if not sec:
                continue
            course = graph.courses.get(sec.course_id)
            lec = graph.lecturers.get(sec.lecturer_id)
            period = graph.periods.get(entry.period_id)
            if not period:
                continue
                
            course_name = course.name if course else sec.course_id
            lec_name = lec.name if lec else sec.lecturer_id
            
            # Nội dung ô
            cell_text = f"**{sec_id}**<br>{course_name}<br>GV: {lec_name}<br>P: {entry.room_id}"
            
            grid[period.day][period.slot] = cell_text
            
        # Dựng DataFrame
        df_data = {}
        for day in DAYS:
            col_name = DAY_NAMES_VI.get(day, day)
            df_data[col_name] = [grid[day][slot] for slot in SLOT_TIMES.keys()]
            
        df = pd.DataFrame(df_data, index=[SLOT_TIMES[slot]["label"] for slot in SLOT_TIMES.keys()])
        return df

    @staticmethod
    def render_html_table(timetable: Timetable, graph: TimetableProblem) -> str:
        """Tạo bảng HTML trực quan hóa thời khóa biểu với màu sắc sinh động."""
        df = TimetableVisualizer.to_dataframe(timetable, graph)
        
        # Tạo style CSS cho bảng thời khóa biểu cực kỳ cao cấp và chuyên nghiệp
        html = """
        <style>
            .timetable-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
                font-family: 'Inter', sans-serif;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            .timetable-table th {
                background-color: #2E5BFF;
                color: white;
                text-align: center;
                padding: 12px;
                font-weight: 600;
            }
            .timetable-table td {
                border: 1px solid #E2E8F0;
                padding: 10px;
                text-align: center;
                vertical-align: top;
                background-color: #F8FAFC;
                height: 100px;
                width: 18%;
            }
            .timetable-table tr:nth-child(even) td {
                background-color: #EDF2F7;
            }
            .timetable-cell {
                background-color: white;
                border-left: 4px solid #4A90D9;
                border-radius: 4px;
                padding: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                font-size: 12px;
                text-align: left;
                word-break: break-word;
            }
            .timetable-cell-empty {
                color: #A0AEC0;
                font-style: italic;
                padding-top: 35px;
            }
        </style>
        """
        
        html += "<table class='timetable-table'><thead><tr><th>Ca học</th>"
        for col in df.columns:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"
        
        for slot_label in df.index:
            html += f"<tr><td style='font-weight: 600; background-color: #EDF2F7; width: 10%; vertical-align: middle;'>{slot_label}</td>"
            for col in df.columns:
                content = df.loc[slot_label, col]
                if content:
                    # Lấy màu của môn học nếu có
                    color = "#4A90D9"
                    # Trích xuất section_id từ content
                    parts = content.split("**")
                    if len(parts) >= 3:
                        sec_id = parts[1]
                        sec = graph.sections.get(sec_id)
                        if sec:
                            course = graph.courses.get(sec.course_id)
                            if course and course.color:
                                color = course.color
                    
                    html += f"<td><div class='timetable-cell' style='border-left-color: {color};'>{content}</div></td>"
                else:
                    html += "<td><div class='timetable-cell-empty'>Trống</div></td>"
            html += "</tr>"
            
        html += "</tbody></table>"
        return html

    @staticmethod
    def print_console(timetable: Timetable, graph: TimetableProblem) -> None:
        """In thời khóa biểu dạng text ra màn hình console."""
        for sec_id, entry in sorted(timetable.entries.items()):
            sec = graph.sections.get(sec_id)
            course_name = graph.courses[sec.course_id].name if sec and sec.course_id in graph.courses else sec_id
            lec_name = graph.lecturers[sec.lecturer_id].name if sec and sec.lecturer_id in graph.lecturers else "Chưa rõ"
            print(f"  [+] Lớp {sec_id} ({course_name}) -> Ca {entry.period_id} tại Phòng {entry.room_id} (GV: {lec_name})")
