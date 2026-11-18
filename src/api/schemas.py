from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ViolationRuleOut(BaseModel):
    id: int
    rule_code: str
    rule_name: str
    severity: str
    penalty_amount: int
    description: Optional[str] = None


class ViolationOut(BaseModel):
    id: int
    track_id: Optional[int] = None
    frame_number: Optional[int] = 0
    detected_at: Optional[datetime] = None
    license_plate: str
    vehicle_type: Optional[str] = "motorbike"
    rule_code: Optional[str] = None
    rule_name: Optional[str] = "Vi phạm giao thông"
    severity: Optional[str] = "medium"
    penalty_amount: Optional[int] = 0
    evidence_image_path: Optional[str] = None
    evidence_url: Optional[str] = None
    status: Optional[str] = "pending"
    confidence: Optional[float] = 0.0
    source_type: Optional[str] = None


class ViolationListResponse(BaseModel):
    success: bool = True
    total: int
    page: int
    page_size: int
    data: List[ViolationOut]


class RuleStatItem(BaseModel):
    label: str
    count: int


class HourStatItem(BaseModel):
    hour: int
    count: int


class StatsOverviewResponse(BaseModel):
    success: bool = True
    total_violations: int
    violations_today: int
    unique_vehicles: int
    plates_identified: int
    by_rule: List[RuleStatItem]
    by_hour: List[HourStatItem]


class VehicleHistoryResponse(BaseModel):
    success: bool = True
    license_plate: str
    vehicle_type: str
    first_detected_at: Optional[datetime] = None
    last_detected_at: Optional[datetime] = None
    total_violations: int
    violations: List[ViolationOut]
