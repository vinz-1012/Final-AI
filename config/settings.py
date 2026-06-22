# Cấu hình chung cho ứng dụng
import os

# Đường dẫn gốc của project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình dữ liệu thời khóa biểu
DEFAULT_TIMETABLE_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_data", "timetable_data.json")
DEFAULT_SMALL_TIMETABLE_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_data", "small_timetable.json")
DEFAULT_MEDIUM_TIMETABLE_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_data", "medium_timetable.json")


# Kích thước khung hình hiển thị
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
