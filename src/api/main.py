import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Đảm bảo đường dẫn gốc project nằm trong sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.database_manager import DatabaseManager
from src.api.schemas import (
    HourStatItem,
    RuleStatItem,
    StatsOverviewResponse,
    VehicleHistoryResponse,
    ViolationListResponse,
    ViolationOut,
)

# Thử import hàm core ML, fallback an toàn nếu model chưa nạp
try:
    from src.models.detection import track_vehicles_and_helmets
except Exception:
    def track_vehicles_and_helmets(image):
        return []

app = FastAPI(
    title="Traffic Violation Detection API (TVDS)",
    description="RESTful API phục vụ hệ thống phát hiện vi phạm giao thông và nhận diện biển số xe",
    version="1.0.0"
)

# 1. Cấu hình CORS để Frontend Streamlit (cổng 8501) gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Mount thư mục tĩnh data/evidence để Frontend truy cập ảnh bằng chứng qua URL
evidence_path = os.path.join(ROOT_DIR, "data", "evidence")
os.makedirs(evidence_path, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=evidence_path), name="evidence")

db_manager = DatabaseManager()


# ------------------------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "TVDS API",
        "database": os.path.exists(db_manager.db_path)
    }


# ------------------------------------------------------------------------------
# VIOLATION ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/api/v1/violations", response_model=ViolationListResponse, tags=["Violations"])
def get_violations(
    page: int = Query(1, ge=1, description="Số trang (bắt đầu từ 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Số bản ghi mỗi trang"),
    rule_code: Optional[str] = Query(None, description="Mã lỗi (NO_HELMET, WRONG_LANE...)"),
    plate: Optional[str] = Query(None, description="Biển số xe cần lọc"),
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)")
):
    """Lấy danh sách các sự kiện vi phạm giao thông kèm bộ lọc và phân trang."""
    clean_page = page if isinstance(page, int) and page >= 1 else 1
    clean_page_size = page_size if isinstance(page_size, int) and page_size >= 1 else 20
    clean_rule = rule_code if isinstance(rule_code, str) else None
    clean_plate = plate if isinstance(plate, str) else None
    clean_start = start_date if isinstance(start_date, str) else None
    clean_end = end_date if isinstance(end_date, str) else None

    offset = (clean_page - 1) * clean_page_size
    records = db_manager.get_violations(
        limit=clean_page_size,
        offset=offset,
        rule_code=clean_rule,
        plate=clean_plate,
        start_date=clean_start,
        end_date=clean_end
    )
    total = db_manager.count_violations(
        rule_code=clean_rule,
        plate=clean_plate,
        start_date=clean_start,
        end_date=clean_end
    )

    data = []
    for r in records:
        # Chuẩn hóa URL ảnh bằng chứng
        ev_path = r.get("evidence_image_path")
        ev_url = None
        if ev_path:
            clean_path = ev_path.replace("\\", "/").replace("data/evidence/", "")
            ev_url = f"/evidence/{clean_path}"

        item = ViolationOut(
            id=r["id"],
            track_id=r.get("track_id"),
            frame_number=r.get("frame_number", 0),
            detected_at=r.get("detected_at"),
            license_plate=r.get("license_plate", "UNKNOWN"),
            vehicle_type=r.get("vehicle_type", "motorbike"),
            rule_code=r.get("rule_code"),
            rule_name=r.get("rule_name", "Vi phạm giao thông"),
            severity=r.get("severity", "medium"),
            penalty_amount=r.get("penalty_amount", 0),
            evidence_image_path=ev_path,
            evidence_url=ev_url,
            status=r.get("status", "pending"),
            confidence=r.get("confidence", 0.0),
            source_type=r.get("source_type")
        )
        data.append(item)

    return ViolationListResponse(
        success=True,
        total=total,
        page=page,
        page_size=page_size,
        data=data
    )


@app.get("/api/v1/violations/{violation_id}", response_model=ViolationOut, tags=["Violations"])
def get_violation_detail(violation_id: int):
    """Lấy thông tin chi tiết của một vụ vi phạm theo ID."""
    r = db_manager.get_violation_by_id(violation_id)
    if not r:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi vi phạm.")

    ev_path = r.get("evidence_image_path")
    ev_url = None
    if ev_path:
        clean_path = ev_path.replace("\\", "/").replace("data/evidence/", "")
        ev_url = f"/evidence/{clean_path}"

    return ViolationOut(
        id=r["id"],
        track_id=r.get("track_id"),
        frame_number=r.get("frame_number", 0),
        detected_at=r.get("detected_at"),
        license_plate=r.get("license_plate", "UNKNOWN"),
        vehicle_type=r.get("vehicle_type", "motorbike"),
        rule_code=r.get("rule_code"),
        rule_name=r.get("rule_name", "Vi phạm giao thông"),
        severity=r.get("severity", "medium"),
        penalty_amount=r.get("penalty_amount", 0),
        evidence_image_path=ev_path,
        evidence_url=ev_url,
        status=r.get("status", "pending"),
        confidence=r.get("confidence", 0.0),
        source_type=r.get("source_type")
    )


# ------------------------------------------------------------------------------
# STATS ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/api/v1/stats/overview", response_model=StatsOverviewResponse, tags=["Dashboard"])
def get_stats_overview():
    """Lấy dữ liệu thống kê tổng hợp phục vụ hiển thị KPI Cards và biểu đồ Dashboard."""
    raw_stats = db_manager.get_statistics()
    by_rule = [RuleStatItem(label=r["label"], count=r["count"]) for r in raw_stats.get("by_rule", [])]
    by_hour = [HourStatItem(hour=h["hour"], count=h["count"]) for h in raw_stats.get("by_hour", [])]

    return StatsOverviewResponse(
        success=True,
        total_violations=raw_stats["total_violations"],
        violations_today=raw_stats["violations_today"],
        unique_vehicles=raw_stats["unique_vehicles"],
        plates_identified=raw_stats["plates_identified"],
        by_rule=by_rule,
        by_hour=by_hour
    )


# ------------------------------------------------------------------------------
# VEHICLES & CHATBOT LOOKUP
# ------------------------------------------------------------------------------
@app.get("/api/v1/vehicles/{plate}", response_model=VehicleHistoryResponse, tags=["Vehicles"])
def lookup_vehicle(plate: str):
    """Tra cứu lịch sử vi phạm của một phương tiện theo biển số (phục vụ Chatbot)."""
    veh = db_manager.get_vehicle_by_plate(plate)
    if not veh:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy phương tiện với biển số: {plate}")

    records = db_manager.get_violations(limit=50, plate=plate)
    violations = []
    for r in records:
        ev_path = r.get("evidence_image_path")
        ev_url = None
        if ev_path:
            clean_path = ev_path.replace("\\", "/").replace("data/evidence/", "")
            ev_url = f"/evidence/{clean_path}"

        violations.append(
            ViolationOut(
                id=r["id"],
                track_id=r.get("track_id"),
                frame_number=r.get("frame_number", 0),
                detected_at=r.get("detected_at"),
                license_plate=r.get("license_plate", plate),
                vehicle_type=r.get("vehicle_type", "motorbike"),
                rule_code=r.get("rule_code"),
                rule_name=r.get("rule_name", "Vi phạm giao thông"),
                severity=r.get("severity", "medium"),
                penalty_amount=r.get("penalty_amount", 0),
                evidence_image_path=ev_path,
                evidence_url=ev_url,
                status=r.get("status", "pending"),
                confidence=r.get("confidence", 0.0),
                source_type=r.get("source_type")
            )
        )

    return VehicleHistoryResponse(
        success=True,
        license_plate=veh["license_plate"],
        vehicle_type=veh.get("vehicle_type", "motorbike"),
        first_detected_at=veh.get("first_detected_at"),
        last_detected_at=veh.get("last_detected_at"),
        total_violations=len(violations),
        violations=violations
    )


# ------------------------------------------------------------------------------
# ML DETECTION ENDPOINT (Single Frame)
# ------------------------------------------------------------------------------
@app.post("/api/v1/detect_frame", tags=["ML Inference"])
async def detect_frame(file: UploadFile = File(...)):
    """Endpoint nhận ảnh upload, decode sang OpenCV BGR và trả về kết quả nhận diện từ core ML."""
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi đọc file: {str(e)}")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="File không hợp lệ hoặc không phải ảnh được hỗ trợ.")

    try:
        detections = track_vehicles_and_helmets(image)
        return {
            "success": True,
            "filename": file.filename,
            "data": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình xử lý model: {str(e)}")
