-- =====================================================================
-- schema.sql
-- Hệ thống phát hiện & xử lý vi phạm giao thông
-- Database: SQLite 3
-- =====================================================================

PRAGMA foreign_keys = ON;

-- =====================================================================
-- 1) processing_sessions
--    Mỗi lần hệ thống chạy model để xử lý 1 video / luồng camera / thư
--    mục ảnh là một "phiên xử lý". Bảng này lưu metadata + thống kê
--    tổng hợp của phiên đó.
-- =====================================================================
CREATE TABLE IF NOT EXISTS processing_sessions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name                TEXT NOT NULL,
    source_type                 TEXT NOT NULL
                                CHECK (source_type IN ('video_file', 'rtsp_stream', 'image_folder', 'webcam')),
    source_path                 TEXT,                       -- đường dẫn video / URL RTSP / thư mục ảnh
    camera_id                   TEXT,                       -- mã định danh camera
    location                    TEXT,                       -- vị trí lắp camera / ngã tư
    model_name                  TEXT,                       -- ví dụ: 'YOLOv8n-traffic'
    model_version               TEXT,
    config_json                 TEXT,                       -- tham số detection (JSON string)
    status                      TEXT NOT NULL DEFAULT 'running'
                                CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    total_frames_processed      INTEGER DEFAULT 0,
    total_vehicles_detected     INTEGER DEFAULT 0,
    total_violations_detected   INTEGER DEFAULT 0,
    error_message                TEXT,                       -- log lỗi nếu status = 'failed'
    started_at                  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at                    TIMESTAMP,
    created_at                  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON processing_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON processing_sessions(started_at);


-- =====================================================================
-- 2) vehicles
--    Mỗi biển số xe là DUY NHẤT trong bảng này (registry). Một xe có
--    thể xuất hiện/vi phạm ở nhiều session khác nhau -> tách bảng để
--    tránh trùng lặp thông tin xe và để tra cứu lịch sử vi phạm theo
--    biển số.
-- =====================================================================
CREATE TABLE IF NOT EXISTS vehicles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    license_plate       TEXT NOT NULL UNIQUE,      -- đã chuẩn hoá (viết hoa, bỏ khoảng trắng)
    plate_confidence    REAL,                       -- độ tin cậy OCR biển số lần gần nhất
    vehicle_type        TEXT
                        CHECK (vehicle_type IN ('car', 'motorbike', 'truck', 'bus', 'other')),
    color               TEXT,
    brand               TEXT,                       -- hãng xe (nếu model nhận diện được)
    first_seen_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_violations    INTEGER DEFAULT 0,          -- đếm nhanh (denormalized), cập nhật qua trigger
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(license_plate);
CREATE INDEX IF NOT EXISTS idx_vehicles_type ON vehicles(vehicle_type);


-- =====================================================================
-- 3) violation_rules
--    Danh mục các loại luật vi phạm giao thông. Tách riêng để:
--      - thêm/sửa/vô hiệu hoá luật mà không phải sửa code xử lý
--      - lưu mức phạt & căn cứ pháp lý tập trung một chỗ
-- =====================================================================
CREATE TABLE IF NOT EXISTS violation_rules (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_code           TEXT NOT NULL UNIQUE,       -- vd: 'RED_LIGHT', 'SPEEDING', 'WRONG_LANE'
    rule_name           TEXT NOT NULL,              -- vd: 'Vượt đèn đỏ'
    description         TEXT,
    severity            TEXT NOT NULL DEFAULT 'medium'
                        CHECK (severity IN ('low', 'medium', 'high')),
    fine_amount_min     INTEGER,                    -- mức phạt tối thiểu (VND)
    fine_amount_max     INTEGER,                    -- mức phạt tối đa (VND)
    legal_reference     TEXT,                       -- vd: 'Nghị định 100/2019/NĐ-CP, Điều 5'
    is_active           INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rules_code ON violation_rules(rule_code);
CREATE INDEX IF NOT EXISTS idx_rules_active ON violation_rules(is_active);


-- =====================================================================
-- 4) violations
--    Bảng trung tâm: mỗi record là MỘT vi phạm cụ thể được phát hiện,
--    liên kết tới xe vi phạm, luật bị vi phạm, và phiên xử lý đã sinh
--    ra nó. Đây là bảng phục vụ tra cứu / báo cáo / duyệt vi phạm.
-- =====================================================================
CREATE TABLE IF NOT EXISTS violations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id              INTEGER NOT NULL REFERENCES vehicles(id) ON DELETE RESTRICT,
    rule_id                 INTEGER NOT NULL REFERENCES violation_rules(id) ON DELETE RESTRICT,
    session_id              INTEGER REFERENCES processing_sessions(id) ON DELETE SET NULL,

    detected_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    camera_id               TEXT,
    location                TEXT,

    confidence_score        REAL,                   -- độ tin cậy của model khi phát hiện vi phạm
    frame_number            INTEGER,                 -- số thứ tự frame trong video (nếu có)
    speed_kmh               REAL,                    -- chỉ dùng cho vi phạm tốc độ
    speed_limit_kmh         REAL,                    -- tốc độ giới hạn tại vị trí đó

    evidence_image_path     TEXT,                    -- đường dẫn ảnh trong data/evidence/
    evidence_video_clip_path TEXT,                   -- đường dẫn clip ngắn (nếu có), có thể NULL

    status                  TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'confirmed', 'rejected', 'processed')),
    reviewed_by             TEXT,                    -- người/tài khoản duyệt
    reviewed_at             TIMESTAMP,
    notes                   TEXT,

    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_violations_vehicle ON violations(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_violations_rule ON violations(rule_id);
CREATE INDEX IF NOT EXISTS idx_violations_session ON violations(session_id);
CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status);
CREATE INDEX IF NOT EXISTS idx_violations_detected_at ON violations(detected_at);


-- =====================================================================
-- Triggers phụ trợ (giữ dữ liệu denormalized cho vehicles luôn đồng bộ)
-- =====================================================================

-- Cập nhật last_seen_at + total_violations khi có vi phạm mới
CREATE TRIGGER IF NOT EXISTS trg_violations_after_insert
AFTER INSERT ON violations
BEGIN
    UPDATE vehicles
    SET total_violations = total_violations + 1,
        last_seen_at = NEW.detected_at,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.vehicle_id;
END;

-- Tự động cập nhật updated_at của vehicles khi sửa record
CREATE TRIGGER IF NOT EXISTS trg_vehicles_before_update
AFTER UPDATE ON vehicles
BEGIN
    UPDATE vehicles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;


-- =====================================================================
-- View tiện dụng cho báo cáo (JOIN sẵn 4 bảng)
-- =====================================================================
CREATE VIEW IF NOT EXISTS v_violation_details AS
SELECT
    v.id                    AS violation_id,
    v.detected_at,
    v.status,
    v.confidence_score,
    v.speed_kmh,
    v.speed_limit_kmh,
    v.evidence_image_path,
    veh.license_plate,
    veh.vehicle_type,
    veh.color,
    r.rule_code,
    r.rule_name,
    r.severity,
    r.fine_amount_min,
    r.fine_amount_max,
    s.session_name,
    s.camera_id            AS session_camera_id,
    s.location              AS session_location
FROM violations v
JOIN vehicles veh        ON veh.id = v.vehicle_id
JOIN violation_rules r   ON r.id = v.rule_id
LEFT JOIN processing_sessions s ON s.id = v.session_id;
