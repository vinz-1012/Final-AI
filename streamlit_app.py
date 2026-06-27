import io
import os
import sys
import time
import random
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# --- Streamlit ---
# pyrefly: ignore [missing-import]
import streamlit as st

# --- Page config ---
st.set_page_config(
    page_title="AI Course Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Third-party ---
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Project imports ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import DEFAULT_TIMETABLE_DATA_PATH, DEFAULT_SMALL_TIMETABLE_DATA_PATH, DEFAULT_MEDIUM_TIMETABLE_DATA_PATH
from data.timetable_loader import TimetableLoader
from src.core.timetable_entities import Timetable, ConstraintChecker, SLOT_TIMES, DAYS, DAY_NAMES_VI
from src.algorithms.uninformed.bfs import BreadthFirstSearch
from src.algorithms.uninformed.dfs import DepthFirstSearch
from src.algorithms.informed.greedy import GreedyBestFirstSearch
from src.algorithms.informed.astar import AStarSearch
from src.algorithms.local.hill_climbing import HillClimbingSearch
from src.algorithms.local.simulated_annealing import SimulatedAnnealingSearch
from src.algorithms.complex.and_or_search import AndOrSearch
from src.algorithms.complex.online_search import LrtaStarSearch
from src.algorithms.csp.backtracking import CSPBacktrackingSearch
from src.algorithms.csp.min_conflicts import CSPMinConflictsSearch
from src.algorithms.adversarial.minimax import MinimaxSearch
from src.algorithms.adversarial.alpha_beta import AlphaBetaSearch
from src.visualization.timetable_visualizer import TimetableVisualizer

# --- Algorithm constants ---
ALGO_OPTIONS = {
    "🔍 Uninformed Search": {
        "Breadth-First Search (BFS)": "bfs",
        "Depth-First Search (DFS)": "dfs",
    },
    "💡 Informed Search": {
        "Greedy Best-First Search": "greedy",
        "A* Search": "astar",
    },
    "⛰️ Local Search": {
        "Hill Climbing": "hill_climbing",
        "Simulated Annealing": "simulated_annealing",
    },
    "📋 Constraint Satisfaction (CSP)": {
        "CSP Backtracking": "csp_backtracking",
        "CSP Min-Conflicts": "csp_min_conflicts",
    },
    "🌐 Complex Search": {
        "AND-OR Search (Dự phòng)": "and_or",
        "LRTA* (Online Search)": "lrta_star",
    },
    "⚔️ Adversarial Search": {
        "Minimax Search": "minimax",
        "Alpha-Beta Pruning": "alpha_beta",
    }
}

ALGO_FLAT = {}
for category, algos in ALGO_OPTIONS.items():
    ALGO_FLAT.update(algos)

# --- CSS Injection for premium UI ---
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0c10 0%, #1f2833 100%);
    }
    
    /* Banner glassmorphism */
    .hero-banner {
        background: rgba(31, 40, 51, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #66fcf1, #45a29e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #c5c6c7;
        margin-top: 5px;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(31, 40, 51, 0.6);
        border: 1px solid rgba(102, 252, 241, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #66fcf1;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #a5a5a5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Title Header ---
st.markdown("""
<div class="hero-banner">
    <span style="background: rgba(102, 252, 241, 0.15); border: 1px solid #66fcf1; color: #66fcf1; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">AI Timetabling Scheduler</span>
    <h1 class="hero-title">HỆ THỐNG XẾP THỜI KHÓA BIỂU TỰ ĐỘNG AI</h1>
    <p class="hero-subtitle">Tối ưu hóa phòng học, giảng viên và ca học bằng các thuật toán tìm kiếm thông minh.</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "timetable" not in st.session_state:
    st.session_state.timetable = None
if "run_stats" not in st.session_state:
    st.session_state.run_stats = {}
if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = "Dữ liệu nhỏ (5 lớp HP)"

# --- Sidebar Inputs ---
st.sidebar.markdown("### ⚙️ Cấu hình Hệ thống")
dataset_choice = st.sidebar.selectbox("📂 Chọn Bộ dữ liệu:", ["Dữ liệu nhỏ (5 lớp HP)", "Dữ liệu vừa (8 lớp HP)", "Dữ liệu chuẩn (40 lớp HP)"])

if st.session_state.last_dataset != dataset_choice:
    st.session_state.timetable = None
    st.session_state.run_stats = {}
    st.session_state.last_dataset = dataset_choice

# Load data based on choice
if dataset_choice.startswith("Dữ liệu nhỏ"):
    data_path = DEFAULT_SMALL_TIMETABLE_DATA_PATH
elif dataset_choice.startswith("Dữ liệu vừa"):
    data_path = DEFAULT_MEDIUM_TIMETABLE_DATA_PATH
else:
    data_path = DEFAULT_TIMETABLE_DATA_PATH
graph = TimetableLoader.load_from_json(data_path)

# Algorithm Choice
st.sidebar.markdown("### 🧠 Chọn Thuật toán AI")
selected_category = st.sidebar.selectbox("Phân loại:", list(ALGO_OPTIONS.keys()))
selected_algo_name = st.sidebar.selectbox("Thuật toán:", list(ALGO_OPTIONS[selected_category].keys()))
algo_key = ALGO_OPTIONS[selected_category][selected_algo_name]

# Settings dynamically displayed based on algorithm group
st.sidebar.markdown("### ⚙️ Tham số Thuật toán")
max_explored = st.sidebar.slider("Giới hạn số nút duyệt:", 100, 100000, 2000, step=100)

algo_params = {}
selected_section = ""
if algo_key == "min_conflicts":
    algo_params["max_steps"] = st.sidebar.slider("Số bước lặp tối đa (Max steps):", 50, 2000, 1000)
elif algo_key in ["hill_climbing", "simulated_annealing"]:
    algo_params["max_steps"] = st.sidebar.slider("Số bước lặp tối đa:", 50, 2000, 1000)
    if algo_key == "simulated_annealing":
        algo_params["initial_temperature"] = st.sidebar.slider("Nhiệt độ ban đầu:", 10.0, 500.0, 100.0)
        algo_params["cooling_rate"] = st.sidebar.slider("Tốc độ hạ nhiệt (Cooling rate):", 0.80, 0.99, 0.95, step=0.01)
elif algo_key in ["minimax", "alpha_beta", "and_or"]:
    selected_section = st.sidebar.selectbox("Chọn Lớp HP cần thử nghiệm:", list(graph.sections.keys()))
    if algo_key in ["minimax", "alpha_beta"]:
        algo_params["depth"] = st.sidebar.slider("Độ sâu trò chơi (Depth):", 1, 4, 2)

# Cảnh báo nếu chọn thuật toán hệ thống trên dữ liệu chuẩn
if dataset_choice.startswith("Dữ liệu chuẩn") and algo_key in ["bfs", "dfs", "greedy", "astar"]:
    st.sidebar.warning(
        "⚠️ Cảnh báo: Thuật toán tìm kiếm hệ thống (BFS, DFS, Greedy, A*) có độ phức tạp lũy thừa rất lớn. "
        "Chạy trên dữ liệu chuẩn 40 lớp HP dễ bị kẹt, chạy rất lâu hoặc quá tải bộ nhớ gây tải lại trang. "
        "Khuyên dùng: CSP (Min-Conflicts) hoặc Local Search để có kết quả tức thì!"
    )

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶ XẾP THỜI KHÓA BIỂU", use_container_width=True)

# --- Run Algorithm ---
if run_button:
    with st.spinner("Đang chạy xếp thời khóa biểu bằng thuật toán AI..."):
        time.sleep(0.2) # Thêm micro-delay cho trải nghiệm
        
        # Instantiate and run
        searcher = None
        if algo_key == "bfs":
            searcher = BreadthFirstSearch()
        elif algo_key == "dfs":
            searcher = DepthFirstSearch()
        elif algo_key == "greedy":
            searcher = GreedyBestFirstSearch()
        elif algo_key == "astar":
            searcher = AStarSearch()
        elif algo_key == "hill_climbing":
            searcher = HillClimbingSearch()
        elif algo_key == "simulated_annealing":
            searcher = SimulatedAnnealingSearch()
        elif algo_key == "csp_backtracking":
            searcher = CSPBacktrackingSearch()
        elif algo_key == "csp_min_conflicts":
            searcher = CSPMinConflictsSearch()
        elif algo_key == "and_or":
            searcher = AndOrSearch()
        elif algo_key == "lrta_star":
            searcher = LrtaStarSearch()
        elif algo_key == "minimax":
            searcher = MinimaxSearch()
        elif algo_key == "alpha_beta":
            searcher = AlphaBetaSearch()
            
        # Run search
        start_node = selected_section if algo_key in ["minimax", "alpha_beta", "and_or"] else ""
        
        kwargs = {"max_explored": max_explored}
        kwargs.update(algo_params)
        
        try:
            path, cost, stats = searcher.search(graph, start_node, "", **kwargs)
            
            st.session_state.timetable = stats.get("timetable", None)
            st.session_state.run_stats = {
                "success": path is not None,
                "cost": cost,
                "explored_nodes": stats.get("explored_nodes", 0),
                "execution_time_ms": stats.get("execution_time", 0.0) * 1000,
                "hard_violations": stats.get("hard_violations", 0),
                "soft_violations": stats.get("soft_violations", {}).get("total", 0.0) if isinstance(stats.get("soft_violations"), dict) else 0.0,
                "fitness": stats.get("fitness", 0.0),
                "error": stats.get("error", None),
                "contingency_plan": stats.get("contingency_plan", None),
                "next_step": stats.get("next_step", None)
            }
        except Exception as e:
            st.session_state.timetable = None
            st.session_state.run_stats = {
                "success": False,
                "error": f"Lỗi trong quá trình xếp lịch: {str(e)}"
            }

# --- Main Page Layout ---
tab_view, tab_data, tab_bench, tab_adv = st.tabs([
    "📅 Xem Thời Khóa Biểu", 
    "📊 Quản lý Dữ liệu gốc", 
    "📈 So sánh Hiệu năng", 
    "🛡️ So sánh từng nhóm thuật toán"
])

# ----------------------------------------------------
# TAB 1: VIEW TIMETABLE
# ----------------------------------------------------
with tab_view:
    if st.session_state.timetable is not None:
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⏱️</div>
                <div class="metric-value">{st.session_state.run_stats['execution_time_ms']:.2f} ms</div>
                <div class="metric-label">Thời gian thực thi</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔍</div>
                <div class="metric-value">{st.session_state.run_stats['explored_nodes']:,}</div>
                <div class="metric-label">Số trạng thái đã duyệt</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⚠️</div>
                <div class="metric-value" style="color: {'#e74c3c' if st.session_state.run_stats['hard_violations'] > 0 else '#2ecc71'};">
                    {st.session_state.run_stats['hard_violations']}
                </div>
                <div class="metric-label">Vi phạm Ràng buộc cứng</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-value">{st.session_state.run_stats['fitness']:.1f}</div>
                <div class="metric-label">Chất lượng (Fitness Score)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Display styled interactive HTML table
        st.markdown("### 🗓️ Lưới Thời Khóa Biểu Tuần")
        
        # Filter options
        filter_type = st.radio("Lọc hiển thị theo:", ["Tất cả", "Phòng học", "Giảng viên", "Học phần"], horizontal=True)
        
        timetable_to_render = st.session_state.timetable
        
        if filter_type == "Phòng học":
            room_id = st.selectbox("Chọn Phòng học:", list(graph.rooms.keys()))
            # Filter entries
            filtered_timetable = Timetable()
            for s_id, e in timetable_to_render.entries.items():
                if e.room_id == room_id:
                    filtered_timetable.assign(s_id, e.period_id, e.room_id)
            timetable_to_render = filtered_timetable
        elif filter_type == "Giảng viên":
            lec_id = st.selectbox("Chọn Giảng viên:", [f"{l_id} - {l.name}" for l_id, l in graph.lecturers.items()])
            lec_id_clean = lec_id.split(" - ")[0]
            filtered_timetable = Timetable()
            for s_id, e in timetable_to_render.entries.items():
                sec = graph.sections.get(s_id)
                if sec and sec.lecturer_id == lec_id_clean:
                    filtered_timetable.assign(s_id, e.period_id, e.room_id)
            timetable_to_render = filtered_timetable
        elif filter_type == "Học phần":
            course_id = st.selectbox("Chọn Học phần:", [f"{c_id} - {c.name}" for c_id, c in graph.courses.items()])
            course_id_clean = course_id.split(" - ")[0]
            filtered_timetable = Timetable()
            for s_id, e in timetable_to_render.entries.items():
                sec = graph.sections.get(s_id)
                if sec and sec.course_id == course_id_clean:
                    filtered_timetable.assign(s_id, e.period_id, e.room_id)
            timetable_to_render = filtered_timetable

        html_table = TimetableVisualizer.render_html_table(timetable_to_render, graph)
        st.markdown(html_table, unsafe_allow_html=True)
        
    else:
        if st.session_state.run_stats.get("error"):
            st.error(f"❌ Không thể xếp lịch: {st.session_state.run_stats['error']}")
            st.info("💡 Gợi ý: Các thuật toán tìm kiếm hệ thống (BFS, DFS, Greedy, A*) hoặc CSP Backtracking dễ bị kẹt/quá tải trên dữ liệu lớn. Hãy thử chuyển sang thuật toán CSP Min-Conflicts hoặc Tìm kiếm cục bộ (Hill Climbing, Simulated Annealing) để xếp lịch nhanh chóng chỉ trong vài mili-giây.")
        else:
            st.info("💡 Chọn cấu hình thuật toán bên thanh bên trái và bấm 'Xếp thời khóa biểu' để bắt đầu sinh lịch tự động.")

# ----------------------------------------------------
# TAB 2: DATA MANAGEMENT
# ----------------------------------------------------
with tab_data:
    st.markdown("### 📑 Chi tiết Dữ liệu đầu vào")
    
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["📚 Lớp Học Phần", "🏫 Phòng Học", "👨‍🏫 Giảng Viên", "📅 Ca học"])
    
    with sub_tab1:
        sec_data = []
        for s_id, sec in graph.sections.items():
            course = graph.courses.get(sec.course_id)
            lec = graph.lecturers.get(sec.lecturer_id)
            sec_data.append({
                "Mã nhóm": s_id,
                "Học phần": course.name if course else sec.course_id,
                "Giảng viên": lec.name if lec else sec.lecturer_id,
                "Sĩ số SV": sec.student_count
            })
        st.dataframe(pd.DataFrame(sec_data), use_container_width=True)
        
    with sub_tab2:
        room_data = []
        for r_id, r in graph.rooms.items():
            room_data.append({
                "Phòng": r_id,
                "Sức chứa": r.capacity,
                "Loại phòng": r.room_type,
                "Tòa nhà": r.building,
                "Khả dụng": "✓ Sẵn sàng" if r.is_available else "✗ Đang sửa"
            })
        st.dataframe(pd.DataFrame(room_data), use_container_width=True)
        
    with sub_tab3:
        lec_data = []
        for l_id, l in graph.lecturers.items():
            lec_data.append({
                "Mã GV": l_id,
                "Tên Giảng Viên": l.name,
                "Khoa": l.department,
                "Ca bận": ", ".join(l.unavailable_periods) if l.unavailable_periods else "Không bận",
                "Ca tối đa/ngày": l.max_periods_per_day,
                "Ca tối đa/tuần": l.max_periods_per_week
            })
        st.dataframe(pd.DataFrame(lec_data), use_container_width=True)
        
    with sub_tab4:
        period_data = []
        for p_id, p in graph.periods.items():
            period_data.append({
                "Mã Ca": p_id,
                "Ngày học": DAY_NAMES_VI.get(p.day, p.day),
                "Tiết học": p.slot_label,
                "Bắt đầu": p.start_time,
                "Kết thúc": p.end_time,
                "Buổi": "Sáng" if p.session == "morning" else "Chiều"
            })
        st.dataframe(pd.DataFrame(period_data), use_container_width=True)

# ----------------------------------------------------
# TAB 3: BENCHMARK
# ----------------------------------------------------
with tab_bench:
    st.markdown("### 📊 So sánh Hiệu năng thuật toán (Benchmark)")
    st.write("Chạy so sánh thời gian chạy và điểm tối ưu (Fitness) giữa các thuật toán phổ biến.")
    
    if st.button("🚀 CHẠY SO SÁNH BENCHMARK", use_container_width=True):
        algorithms_to_compare = {
            "BFS": BreadthFirstSearch(),
            "DFS": DepthFirstSearch(),
            "Greedy": GreedyBestFirstSearch(),
            "A*": AStarSearch(),
            "Hill Climbing": HillClimbingSearch(),
            "Simulated Annealing": SimulatedAnnealingSearch(),
            "CSP Backtracking": CSPBacktrackingSearch(),
            "CSP Min-Conflicts": CSPMinConflictsSearch(),
            "LRTA* (Online)": LrtaStarSearch()
        }
        
        bench_results = []
        
        with st.spinner("Đang chạy benchmark toàn bộ 9 thuật toán..."):
            for name, algo in algorithms_to_compare.items():
                t_start = time.perf_counter()
                path, cost, stats = algo.search(graph, "", "")
                t_end = time.perf_counter()
                
                bench_results.append({
                    "Thuật toán": name,
                    "Xếp thành công": "✓ Có" if path is not None else "✗ Không",
                    "Thời gian (ms)": (t_end - t_start) * 1000,
                    "Nút đã duyệt": stats.get("explored_nodes", 0),
                    "Điểm phạt vi phạm": cost,
                    "Chất lượng (Fitness)": stats.get("fitness", 0.0)
                })
                
        df_bench = pd.DataFrame(bench_results)
        st.dataframe(df_bench, use_container_width=True)
        
        # Plotting charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#0f172a')
            ax.bar(df_bench["Thuật toán"], df_bench["Thời gian (ms)"], color="#66fcf1")
            ax.set_title("Thời gian thực thi (ms) - Ít hơn là tốt hơn", color="white")
            ax.tick_params(colors="white")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#0f172a')
            ax.bar(df_bench["Thuật toán"], df_bench["Chất lượng (Fitness)"], color="#45a29e")
            ax.set_title("Chất lượng TKB (Fitness) - Cao hơn là tốt hơn", color="white")
            ax.tick_params(colors="white")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

# ----------------------------------------------------
# TAB 4: INTRA-GROUP ALGORITHM COMPARISON
# ----------------------------------------------------
with tab_adv:
    st.markdown("### ⚖️ Đối sánh Thuật toán trong Cùng Nhóm")
    st.write("Thực hiện chạy so sánh trực tiếp hiệu năng, độ phức tạp và chất lượng lời giải giữa các thuật toán trong 6 nhóm AI cốt lõi.")

    # Initialize session state for group benchmark results
    if "group_bench_results" not in st.session_state:
        st.session_state.group_bench_results = None

    if st.button("🚀 CHẠY SO SÁNH 6 NHÓM THUẬT TOÁN", use_container_width=True):
        groups_to_run = {
            "🔍 Uninformed Search": {
                "BFS (Breadth-First Search)": BreadthFirstSearch(),
                "DFS (Depth-First Search)": DepthFirstSearch()
            },
            "💡 Informed Search": {
                "Greedy Best-First Search": GreedyBestFirstSearch(),
                "A* Search": AStarSearch()
            },
            "⛰️ Local Search": {
                "Hill Climbing": HillClimbingSearch(),
                "Simulated Annealing": SimulatedAnnealingSearch()
            },
            "📋 Constraint Satisfaction (CSP)": {
                "CSP Backtracking": CSPBacktrackingSearch(),
                "CSP Min-Conflicts": CSPMinConflictsSearch()
            },
            "🌐 Complex Search": {
                "AND-OR Search": AndOrSearch(),
                "LRTA* (Online Search)": LrtaStarSearch()
            },
            "⚔️ Adversarial Search": {
                "Minimax Search": MinimaxSearch(),
                "Alpha-Beta Pruning": AlphaBetaSearch()
            }
        }

        results = {}
        with st.spinner("Đang chạy đối sánh song song 6 nhóm thuật toán trên bộ dữ liệu hiện tại..."):
            for group_name, algos in groups_to_run.items():
                results[group_name] = {}
                for name, searcher in algos.items():
                    t_start = time.perf_counter()
                    
                    # Xác định nút bắt đầu nếu cần thiết
                    start_node = ""
                    if name in ["Minimax Search", "Alpha-Beta Pruning", "AND-OR Search"]:
                        start_node = list(graph.sections.keys())[0] if graph.sections else ""
                    
                    try:
                        path, cost, stats = searcher.search(graph, start_node, "")
                        t_end = time.perf_counter()
                        duration = (t_end - t_start) * 1000  # ms
                        
                        results[group_name][name] = {
                            "success": path is not None or name in ["Minimax Search", "Alpha-Beta Pruning"],
                            "duration_ms": duration,
                            "explored_nodes": stats.get("explored_nodes", 0),
                            "cost": cost,
                            "fitness": stats.get("fitness", 0.0) if "fitness" in stats else (stats.get("best_value", 0.0) if "best_value" in stats else 0.0)
                        }
                    except Exception as e:
                        results[group_name][name] = {
                            "success": False,
                            "duration_ms": 0.0,
                            "explored_nodes": 0,
                            "cost": 0.0,
                            "fitness": 0.0,
                            "error": str(e)
                        }
            st.session_state.group_bench_results = results
            st.success("Đã hoàn thành phân tích đối sánh 6 nhóm thuật toán!")

    if st.session_state.group_bench_results is not None:
        results = st.session_state.group_bench_results
        
        group_tabs = st.tabs(list(results.keys()))
        
        # 1. Uninformed Search
        with group_tabs[0]:
            g_name = "🔍 Uninformed Search"
            st.markdown("#### Đối sánh: Breadth-First Search (BFS) vs Depth-First Search (DFS)")
            st.info("💡 **Lý thuyết**: BFS duyệt theo chiều rộng (từng cấp một) đảm bảo tính tối ưu (tìm thấy giải pháp ngắn nhất trước) nhưng tốn bộ nhớ khủng khiếp. DFS đi sâu nhất có thể trên một nhánh, tốn ít bộ nhớ hơn nhưng dễ đi vào nhánh cụt hoặc không tối ưu.")
            
            algo1, algo2 = list(results[g_name].keys())[0], list(results[g_name].keys())[1]
            r1, r2 = results[g_name][algo1], results[g_name][algo2]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"##### 📊 Số liệu kỹ thuật")
                df_compare = pd.DataFrame({
                    "Thuộc tính": ["Trạng thái thành công", "Thời gian thực thi (ms)", "Số nút đã duyệt", "Điểm phạt (Cost)", "Điểm Fitness"],
                    algo1: ["✓ Có" if r1["success"] else "✗ Không", f"{r1['duration_ms']:.2f} ms", f"{r1['explored_nodes']:,}", f"{r1['cost']:.1f}", f"{r1['fitness']:.1f}"],
                    algo2: ["✓ Có" if r2["success"] else "✗ Không", f"{r2['duration_ms']:.2f} ms", f"{r2['explored_nodes']:,}", f"{r2['cost']:.1f}", f"{r2['fitness']:.1f}"]
                })
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#0f172a')
                ax.bar([algo1, algo2], [r1['duration_ms'], r2['duration_ms']], color=["#66fcf1", "#45a29e"])
                ax.set_title("Thời gian thực thi (ms) - Ít hơn là tốt hơn", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                st.pyplot(fig)
                
        # 2. Informed Search
        with group_tabs[1]:
            g_name = "💡 Informed Search"
            st.markdown("#### Đối sánh: Greedy Best-First Search vs A* Search")
            st.info("💡 **Lý thuyết**: Greedy chỉ sử dụng hàm đánh giá heuristic $h(n)$ để đi nhanh nhất tới đích mà bỏ qua chi phí thực tế $g(n)$, dễ bị tối ưu cục bộ. A* kết hợp cả hai $f(n) = g(n) + h(n)$ để vừa tìm kiếm nhanh vừa đảm bảo tính tối ưu tuyệt đối.")
            
            algo1, algo2 = list(results[g_name].keys())[0], list(results[g_name].keys())[1]
            r1, r2 = results[g_name][algo1], results[g_name][algo2]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"##### 📊 Số liệu kỹ thuật")
                df_compare = pd.DataFrame({
                    "Thuộc tính": ["Trạng thái thành công", "Thời gian thực thi (ms)", "Số nút đã duyệt", "Điểm phạt (Cost)", "Điểm Fitness"],
                    algo1: ["✓ Có" if r1["success"] else "✗ Không", f"{r1['duration_ms']:.2f} ms", f"{r1['explored_nodes']:,}", f"{r1['cost']:.1f}", f"{r1['fitness']:.1f}"],
                    algo2: ["✓ Có" if r2["success"] else "✗ Không", f"{r2['duration_ms']:.2f} ms", f"{r2['explored_nodes']:,}", f"{r2['cost']:.1f}", f"{r2['fitness']:.1f}"]
                })
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#0f172a')
                ax.bar([algo1, algo2], [r1['duration_ms'], r2['duration_ms']], color=["#66fcf1", "#45a29e"])
                ax.set_title("Thời gian thực thi (ms) - Ít hơn là tốt hơn", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                st.pyplot(fig)

        # 3. Local Search
        with group_tabs[2]:
            g_name = "⛰️ Local Search"
            st.markdown("#### Đối sánh: Hill Climbing vs Simulated Annealing")
            st.info("💡 **Lý thuyết**: Hill Climbing luôn đi lên hướng tốt nhất cận kề nên rất nhanh nhưng cực kỳ dễ mắc kẹt tại cực trị địa phương (local optima). Simulated Annealing cho phép chấp nhận các bước đi tệ hơn với xác suất giảm dần theo nhiệt độ, giúp nó vượt qua thung lũng để tìm cực trị toàn cục.")
            
            algo1, algo2 = list(results[g_name].keys())[0], list(results[g_name].keys())[1]
            r1, r2 = results[g_name][algo1], results[g_name][algo2]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"##### 📊 Số liệu kỹ thuật")
                df_compare = pd.DataFrame({
                    "Thuộc tính": ["Trạng thái thành công", "Thời gian thực thi (ms)", "Số nút đã duyệt", "Điểm phạt (Cost)", "Điểm Fitness"],
                    algo1: ["✓ Có" if r1["success"] else "✗ Không", f"{r1['duration_ms']:.2f} ms", f"{r1['explored_nodes']:,}", f"{r1['cost']:.1f}", f"{r1['fitness']:.1f}"],
                    algo2: ["✓ Có" if r2["success"] else "✗ Không", f"{r2['duration_ms']:.2f} ms", f"{r2['explored_nodes']:,}", f"{r2['cost']:.1f}", f"{r2['fitness']:.1f}"]
                })
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#0f172a')
                ax.bar([algo1, algo2], [r1['duration_ms'], r2['duration_ms']], color=["#66fcf1", "#45a29e"])
                ax.set_title("Thời gian thực thi (ms) - Ít hơn là tốt hơn", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                st.pyplot(fig)

        # 4. CSP
        with group_tabs[3]:
            g_name = "📋 Constraint Satisfaction (CSP)"
            st.markdown("#### Đối sánh: CSP Backtracking vs CSP Min-Conflicts")
            st.info("💡 **Lý thuyết**: CSP Backtracking gán biến tuần tự và quay lui khi vi phạm ràng buộc (tìm kiếm hệ thống). CSP Min-Conflicts khởi tạo một trạng thái đầy đủ (nhưng có lỗi) rồi liên tục sửa lỗi bằng cách thay đổi giá trị của biến sao cho số xung đột là tối thiểu, cực kỳ hiệu quả cho dữ liệu lớn.")
            
            algo1, algo2 = list(results[g_name].keys())[0], list(results[g_name].keys())[1]
            r1, r2 = results[g_name][algo1], results[g_name][algo2]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"##### 📊 Số liệu kỹ thuật")
                df_compare = pd.DataFrame({
                    "Thuộc tính": ["Trạng thái thành công", "Thời gian thực thi (ms)", "Số nút đã duyệt", "Điểm phạt (Cost)", "Điểm Fitness"],
                    algo1: ["✓ Có" if r1["success"] else "✗ Không", f"{r1['duration_ms']:.2f} ms", f"{r1['explored_nodes']:,}", f"{r1['cost']:.1f}", f"{r1['fitness']:.1f}"],
                    algo2: ["✓ Có" if r2["success"] else "✗ Không", f"{r2['duration_ms']:.2f} ms", f"{r2['explored_nodes']:,}", f"{r2['cost']:.1f}", f"{r2['fitness']:.1f}"]
                })
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#0f172a')
                ax.bar([algo1, algo2], [r1['duration_ms'], r2['duration_ms']], color=["#66fcf1", "#45a29e"])
                ax.set_title("Thời gian thực thi (ms) - Ít hơn là tốt hơn", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                st.pyplot(fig)

        # 5. Complex Search
        with group_tabs[4]:
            g_name = "🌐 Complex Search"
            st.markdown("#### Đối sánh: AND-OR Search vs LRTA* (Online Search)")
            st.info("💡 **Lý thuyết**: AND-OR search xây dựng cây kế hoạch dự phòng đối phó với môi trường không chắc chắn (nếu A xảy ra thì làm X, nếu B xảy ra thì làm Y). LRTA* là thuật toán tìm kiếm thời gian thực học hỏi (Learning Real-Time A*), liên tục cập nhật giá trị heuristic trong khi di chuyển thực tế.")
            
            algo1, algo2 = list(results[g_name].keys())[0], list(results[g_name].keys())[1]
            r1, r2 = results[g_name][algo1], results[g_name][algo2]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"##### 📊 Số liệu kỹ thuật")
                df_compare = pd.DataFrame({
                    "Thuộc tính": ["Trạng thái thành công", "Thời gian thực thi (ms)", "Số nút đã duyệt", "Điểm phạt (Cost)", "Điểm Fitness"],
                    algo1: ["✓ Có" if r1["success"] else "✗ Không", f"{r1['duration_ms']:.2f} ms", f"{r1['explored_nodes']:,}", f"{r1['cost']:.1f}", f"{r1['fitness']:.1f}"],
                    algo2: ["✓ Có" if r2["success"] else "✗ Không", f"{r2['duration_ms']:.2f} ms", f"{r2['explored_nodes']:,}", f"{r2['cost']:.1f}", f"{r2['fitness']:.1f}"]
                })
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#0f172a')
                ax.bar([algo1, algo2], [r1['duration_ms'], r2['duration_ms']], color=["#66fcf1", "#45a29e"])
                ax.set_title("Thời gian thực thi (ms) - Ít hơn là tốt hơn", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                st.pyplot(fig)

        # 6. Adversarial Search
        with group_tabs[5]:
            g_name = "⚔️ Adversarial Search"
            st.markdown("#### Đối sánh: Minimax Search vs Alpha-Beta Pruning")
            st.info("💡 **Lý thuyết**: Cả hai đều tìm kiếm quyết định robust tối ưu chống lại nước đi tệ nhất của đối thủ (sự cố môi trường). Minimax duyệt toàn bộ cây trò chơi nên rất chậm. Alpha-Beta Pruning tối ưu hóa bằng cách cắt bỏ các nhánh chắc chắn không ảnh hưởng đến kết quả cuối cùng, cho ra cùng kết quả nhưng duyệt ít nút hơn đáng kể.")
            
            algo1, algo2 = list(results[g_name].keys())[0], list(results[g_name].keys())[1]
            r1, r2 = results[g_name][algo1], results[g_name][algo2]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"##### 📊 Số liệu kỹ thuật")
                df_compare = pd.DataFrame({
                    "Thuộc tính": ["Trạng thái thành công", "Thời gian thực thi (ms)", "Số nút đã duyệt", "Giá trị Tiện ích (Utility)", "Điểm Fitness"],
                    algo1: ["✓ Có" if r1["success"] else "✗ Không", f"{r1['duration_ms']:.2f} ms", f"{r1['explored_nodes']:,}", f"{r1['cost']:.1f}", f"{r1['fitness']:.1f}"],
                    algo2: ["✓ Có" if r2["success"] else "✗ Không", f"{r2['duration_ms']:.2f} ms", f"{r2['explored_nodes']:,}", f"{r2['cost']:.1f}", f"{r2['fitness']:.1f}"]
                })
                st.dataframe(df_compare, use_container_width=True, hide_index=True)
                
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#0f172a')
                ax.bar([algo1, algo2], [r1['explored_nodes'], r2['explored_nodes']], color=["#66fcf1", "#45a29e"])
                ax.set_title("Số nút đã duyệt (Càng ít càng tốt do cắt tỉa nhánh)", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                st.pyplot(fig)
    else:
        st.info("💡 Hãy nhấn nút '🚀 CHẠY SO SÁNH 6 NHÓM THUẬT TOÁN' ở trên để tiến hành đối sánh trực tiếp các thuật toán trong từng nhóm.")

