-- ==============================================================================
-- SCHEMA CƠ SỞ DỮ LIỆU HỆ THỐNG PHÁT HIỆN VI PHẠM GIAO THÔNG (TVDS)
-- Chuẩn hóa: 3NF / BCNF
-- Cơ sở dữ liệu: SQLite 3
-- ==============================================================================

PRAGMA foreign_keys = ON;

-- 1. Bảng quy tắc vi phạm (Violation Rules)
CREATE TABLE IF NOT EXISTS violation_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_code TEXT NOT NULL UNIQUE,          -- NO_HELMET, WRONG_LANE, WRONG_WAY, RED_LIGHT
    rule_name TEXT NOT NULL,                 -- Tên lỗi tiếng Việt
    description TEXT,                        -- Mô tả chi tiết hành vi vi phạm
    severity TEXT DEFAULT 'medium',          -- low, medium, high, critical
    penalty_amount INTEGER DEFAULT 0,        -- Mức phạt tiền quy định (VNĐ)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Bảng phiên xử lý camera/video (Processing Sessions)
CREATE TABLE IF NOT EXISTS processing_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT,                       -- Tên phiên (vd: Live Camera Cổng 1, Test Video)
    source_type TEXT NOT NULL,               -- webcam, video_file, rtsp_stream
    source_path TEXT,                        -- Đường dẫn video hoặc URL stream
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    total_frames INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'             -- active, completed, stopped, error
);

-- 3. Bảng phương tiện giao thông (Vehicles)
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license_plate TEXT NOT NULL UNIQUE,      -- Biển số xe (ví dụ: 59A1-12345, hoặc UNKNOWN)
    vehicle_type TEXT DEFAULT 'motorbike',   -- motorbike, car, truck, bus
    first_detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- 4. Bảng ghi nhận sự kiện vi phạm (Violations)
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    vehicle_id INTEGER NOT NULL,
    rule_id INTEGER,
    track_id INTEGER,                        -- ID của đối tượng từ tracker (ByteTrack)
    frame_number INTEGER DEFAULT 0,          -- Số thứ tự khung hình trong video/session
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence REAL DEFAULT 0.0,             -- Độ tin cậy của phát hiện AI (0.0 -> 1.0)
    evidence_image_path TEXT,                -- Đường dẫn tương đối tới ảnh bằng chứng (data/evidence/...)
    evidence_video_clip_path TEXT,           -- Đường dẫn clip ngắn (nếu có)
    status TEXT DEFAULT 'pending',           -- pending, verified, rejected, fined
    notes TEXT,
    FOREIGN KEY (session_id) REFERENCES processing_sessions(id) ON DELETE SET NULL,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE,
    FOREIGN KEY (rule_id) REFERENCES violation_rules(id) ON DELETE RESTRICT
);

-- ==============================================================================
-- INDEXES TỐI ƯU TRUY VẤN
-- ==============================================================================
CREATE INDEX IF NOT EXISTS idx_violations_detected_at ON violations(detected_at);
CREATE INDEX IF NOT EXISTS idx_violations_vehicle_id ON violations(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_violations_rule_id ON violations(rule_id);
CREATE INDEX IF NOT EXISTS idx_violations_session_id ON violations(session_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_license_plate ON vehicles(license_plate);

-- ==============================================================================
-- VIEWS PHỤC VỤ DASHBOARD & BÁO CÁO NHANH
-- ==============================================================================
CREATE VIEW IF NOT EXISTS v_violation_details AS
SELECT 
    v.id AS id,
    v.track_id AS track_id,
    v.frame_number AS frame_number,
    v.detected_at AS detected_at,
    veh.license_plate AS license_plate,
    veh.vehicle_type AS vehicle_type,
    r.rule_code AS rule_code,
    COALESCE(r.rule_name, v.notes, 'Vi phạm giao thông') AS rule_name,
    r.severity AS severity,
    r.penalty_amount AS penalty_amount,
    v.evidence_image_path AS evidence_image_path,
    v.status AS status,
    v.confidence AS confidence,
    s.source_type AS source_type
FROM violations v
JOIN vehicles veh ON veh.id = v.vehicle_id
LEFT JOIN violation_rules r ON r.id = v.rule_id
LEFT JOIN processing_sessions s ON s.id = v.session_id;

-- ==============================================================================
-- DỮ LIỆU KHỞI TẠO DANH MỤC LỖI (Mặc định)
-- ==============================================================================
INSERT OR IGNORE INTO violation_rules (id, rule_code, rule_name, severity, penalty_amount, description) VALUES
(1, 'NO_HELMET', 'Không đội mũ bảo hiểm', 'low', 500000, 'Người điều khiển hoặc người ngồi sau xe máy không đội mũ bảo hiểm hoặc đội không cài quai đúng quy cách'),
(2, 'WRONG_LANE', 'Sai làn đường', 'medium', 500000, 'Đi không đúng làn đường quy định cho loại phương tiện'),
(3, 'WRONG_WAY', 'Đi ngược chiều', 'high', 1500000, 'Đi ngược chiều của đường một chiều hoặc đi ngược chiều trên đường có biển cấm'),
(4, 'RED_LIGHT', 'Vượt đèn đỏ', 'critical', 900000, 'Không chấp hành hiệu lệnh của đèn tín hiệu giao thông');
