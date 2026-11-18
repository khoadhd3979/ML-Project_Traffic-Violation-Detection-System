import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.data.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("TVDS_API_URL", "http://localhost:8000")
COLUMNS = ["ID", "Track ID", "Time", "Plate", "Code", "Violation", "Evidence"]


class APIClient:
    """
    Client gọi RESTful API từ FastAPI backend.
    Tự động fallback về DatabaseManager nếu backend chưa bật để UI không bị gián đoạn.
    """

    def __init__(self, base_url: str = API_BASE_URL, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.db = DatabaseManager()

    def is_api_online(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=1.0)
            return r.status_code == 200 and r.json().get("status") == "healthy"
        except Exception:
            return False

    def fetch_violations(
        self,
        page: int = 1,
        page_size: int = 100,
        rule_code: Optional[str] = None,
        plate: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str, Optional[str]]:
        """
        Lấy danh sách vi phạm. Trả về: (DataFrame, Nguồn dữ liệu, Thông báo lỗi nếu có).
        """
        # 1. Thử gọi qua FastAPI
        try:
            params = {"page": page, "page_size": page_size}
            if rule_code and rule_code != "Tất cả loại lỗi":
                params["rule_code"] = rule_code
            if plate and plate.strip():
                params["plate"] = plate.strip()
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            res = requests.get(f"{self.base_url}/api/v1/violations", params=params, timeout=self.timeout)
            if res.status_code == 200:
                data = res.json().get("data", [])
                if not data:
                    return pd.DataFrame(columns=COLUMNS), "FastAPI (Trống)", None

                rows = []
                for item in data:
                    # Ghép URL đầy đủ cho ảnh bằng chứng
                    ev_url = item.get("evidence_url")
                    full_ev = f"{self.base_url}{ev_url}" if ev_url else item.get("evidence_image_path")

                    rows.append({
                        "ID": item["id"],
                        "Track ID": item.get("track_id") or item.get("frame_number", 0),
                        "Time": pd.to_datetime(item.get("detected_at")),
                        "Plate": item.get("license_plate", "UNKNOWN"),
                        "Code": item.get("rule_code", "VIOLATION"),
                        "Violation": item.get("rule_name", "Vi phạm giao thông"),
                        "Evidence": full_ev
                    })
                df = pd.DataFrame(rows).reindex(columns=COLUMNS)
                return df, "FastAPI Backend (Cổng 8000)", None
        except Exception as e:
            logger.warning(f"FastAPI offline hoặc lỗi kết nối: {e}. Đang chuyển sang đọc CSDL cục bộ.")

        # 2. Fallback sang đọc CSDL SQLite trực tiếp
        try:
            records = self.db.get_violations(
                limit=page_size,
                offset=(page - 1) * page_size,
                rule_code=rule_code,
                plate=plate,
                start_date=start_date,
                end_date=end_date
            )
            if not records:
                return pd.DataFrame(columns=COLUMNS), "Cơ sở dữ liệu (Trống)", None

            rows = []
            for r in records:
                rows.append({
                    "ID": r["id"],
                    "Track ID": r.get("track_id") or r.get("frame_number", 0),
                    "Time": pd.to_datetime(r.get("detected_at")),
                    "Plate": r.get("license_plate", "UNKNOWN"),
                    "Code": r.get("rule_code", "VIOLATION"),
                    "Violation": r.get("rule_name", "Vi phạm giao thông"),
                    "Evidence": r.get("evidence_image_path")
                })
            df = pd.DataFrame(rows).reindex(columns=COLUMNS)
            return df, "Cơ sở dữ liệu SQLite (Fallback)", None
        except Exception as db_err:
            return pd.DataFrame(columns=COLUMNS), "Lỗi kết nối", str(db_err)

    def fetch_statistics(self) -> Dict[str, Any]:
        """Lấy dữ liệu thống kê tổng quan."""
        try:
            res = requests.get(f"{self.base_url}/api/v1/stats/overview", timeout=self.timeout)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

        # Fallback SQLite
        return self.db.get_statistics()

    def lookup_plate(self, plate: str) -> Optional[Dict[str, Any]]:
        """Tra cứu biển số qua API hoặc DB."""
        try:
            res = requests.get(f"{self.base_url}/api/v1/vehicles/{plate.strip().upper()}", timeout=self.timeout)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

        veh = self.db.get_vehicle_by_plate(plate)
        if not veh:
            return None
        violations = self.db.get_violations(limit=50, plate=plate)
        return {
            "license_plate": veh["license_plate"],
            "total_violations": len(violations),
            "violations": violations
        }
