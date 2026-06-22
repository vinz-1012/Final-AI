from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from src.core.graph import Graph

class BaseSearchAlgorithm(ABC):
    """Lớp trừu tượng cho mọi thuật toán tìm kiếm đường đi."""
    
    @abstractmethod
    def search(
        self, 
        graph: Graph, 
        start: str, 
        goal: str, 
        **kwargs: Any
    ) -> Tuple[Optional[List[str]], float, Dict[str, Any]]:
        """
        Thực hiện tìm kiếm trên đồ thị từ start đến goal.

        Args:
            graph: Đối tượng đồ thị Graph.
            start: ID của nút bắt đầu.
            goal: ID của nút đích.
            kwargs: Các tham số tùy chọn khác của từng thuật toán.

        Returns:
            Tuple bao gồm:
            - List[str] hoặc None: Danh sách đường đi các Node ID (hoặc None nếu không tìm thấy).
            - float: Tổng chi phí đường đi (cost).
            - dict: Các thông số đo lường hiệu suất (ví dụ số node đã duyệt, thời gian chạy, v.v.).
        """
        pass
