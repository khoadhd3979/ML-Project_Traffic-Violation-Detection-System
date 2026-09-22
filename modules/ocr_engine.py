import re
import cv2
import logging
import numpy as np
from paddleocr import PaddleOCR as POCR

log = logging.getLogger("ocr_engine")

class PlateOCR:
    """
    Class nhận dạng ký tự biển số xe Việt Nam sử dụng PaddleOCR.
    Hỗ trợ cả biển dài (1 dòng) và biển vuông (2 dòng).
    """

    # Vị trí SỐ: sửa chữ cái trông giống số → thành số
    CHAR_TO_NUM = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'L': '1',
        'Z': '2',
        'S': '5',
        'G': '6',
        'T': '7',
        'B': '8',
    }

    # Vị trí CHỮ: sửa số trông giống chữ → thành chữ
    NUM_TO_CHAR = {
        '0': 'O',
        '1': 'A',
        '2': 'Z',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
    }

    def __init__(self, gpu=False):
        """
        Khởi tạo PaddleOCR reader.
        """
        print("Đang khởi tạo PaddleOCR (lần đầu có thể mất vài giây để tải mô hình)...")
        # Nhận diện biển số xe chỉ cần ký tự tiếng Anh/Số -> Dùng lang='en' để tăng tốc và giảm dung lượng
        self.reader = POCR(use_angle_cls=False, lang='en', enable_mkldnn=False, cpu_threads=6, det_db_thresh=0.3, rec_batch_num=1,)
        print("PaddleOCR đã sẵn sàng!")

    def read_plate(self, image):
        img = self._load_image(image)
        if img is None:
            return ""

        # Nếu img.shape chỉ có 2 phần tử (ảnh xám/nhị phân), ta nhân bản nó lên 3 kênh
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Phòng trường hợp ảnh có dạng (H, W, 1)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # --- Bước 1: Chạy PaddleOCR quét chữ ---
        try:
            results = self.reader.ocr(img)
        except Exception as e:
            print(f"[OCR] Lỗi runtime khi quét chữ: {e}")
            return ""

        # Trường hợp PaddleOCR không đọc được gì hoặc trả về None
        if not results or results[0] is None:
            return ""

        # Sắp xếp các box chữ tìm được và ghép lại thành chuỗi thô
        raw_text = self._sort_and_merge_paddle(results[0], img)
        
        # Làm sạch và sửa lỗi theo cấu trúc biển số Việt Nam
        final_text = self.clean_and_fix(raw_text)
        return final_text

    def _sort_and_merge_paddle(self, result_list, img):
        """
        Sắp xếp các cụm chữ nhận diện được từ trên xuống dưới, từ trái sang phải.
        Phù hợp cho cả biển vuông 2 dòng.
        """
        img_height = img.shape[0]

        if isinstance(result_list, dict) or hasattr(result_list, 'keys'):
            boxes = result_list.get('dt_polys', [])
            texts = result_list.get('rec_texts', [])
            scores = result_list.get('rec_scores', [])
            
            # Đóng gói lại thành dạng List chuẩn: [ [box, (text, score)], ... ]
            parsed_list = []
            for b, t, s in zip(boxes, texts, scores):
                parsed_list.append([b, (t, s)])
            result_list = parsed_list

        # Lọc bỏ các kết quả có độ tin cậy quá thấp (Dưới 40%)
        MIN_CONFIDENCE = 0.4
        valid_results = []
        
        # Xử lý trích xuất an toàn cho nhiều format khác nhau
        for r in result_list:
            try:
                box = r[0]
                if isinstance(r[1], (tuple, list)):
                    text = r[1][0]
                    conf = r[1][1]
                else: 
                    # Đề phòng format trả về dạng phẳng: [box, text, conf]
                    text = r[1]
                    conf = r[2] if len(r) > 2 else 1.0
                
                if conf > MIN_CONFIDENCE:
                    # Lưu lại với cấu trúc phẳng cho dễ dùng
                    valid_results.append([box, text, conf]) 
            except Exception:
                continue

        if not valid_results:
            return ""

        # Hàm lấy tọa độ Y và X trung tâm của hộp chữ
        def center_y(r):
            box = r[0]
            return (box[0][1] + box[2][1]) / 2.0

        def center_x(r):
            box = r[0]
            return (box[0][0] + box[2][0]) / 2.0

        # Sắp xếp từ trên xuống dưới trước theo trục Y
        valid_results.sort(key=center_y)

        # Ngưỡng phân tách dòng (30% chiều cao ảnh crop)
        line_threshold = img_height * 0.30
        lines = []
        current_line = [valid_results[0]]

        for r in valid_results[1:]:
            current_line_avg_y = np.mean([center_y(x) for x in current_line])
            if abs(center_y(r) - current_line_avg_y) <= line_threshold:
                current_line.append(r)
            else:
                lines.append(current_line)
                current_line = [r]
        lines.append(current_line)

        # Trong mỗi dòng, sắp xếp ký tự từ trái sang phải theo trục X
        for line in lines:
            line.sort(key=center_x)

        # Ghép văn bản thô
        raw_text = ""
        for line in lines:
            for r in line:
                text_chunk = r[1] # Lấy chuỗi text ra (vì ở trên đã ép về chuẩn [box, text, conf])
                if isinstance(text_chunk, str):
                    raw_text += text_chunk.upper()

        return raw_text

    def correct_plate_format(self, text):
        text = ''.join(e for e in text if e.isalnum()).upper()
        corrected_text = ""
        
        for i, char in enumerate(text):
            if i in [0, 1]:  # Mã tỉnh: BẮT BUỘC LÀ SỐ
                corrected_text += self.CHAR_TO_NUM.get(char, char)
            elif i == 2:     # Series chính: BẮT BUỘC LÀ CHỮ
                corrected_text += self.NUM_TO_CHAR.get(char, char)
            elif i == 3:     # Ký tự thứ 4: Giữ nguyên (có thể là chữ hoặc số tùy loại biển xe)
                corrected_text += char
            elif i >= 4:     # Phần đuôi số: BẮT BUỘC LÀ SỐ
                corrected_text += self.CHAR_TO_NUM.get(char, char)
        
        return corrected_text

    def clean_and_fix(self, text):
        # Bước 1: Loại bỏ toàn bộ ký tự đặc biệt (dấu chấm, gạch ngang, khoảng trắng)
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        log.debug("Sau làm sạch: '%s'", text)

        # Bước 2: Kiểm tra độ dài — biển số VN hợp lệ có 7-9 ký tự
        if len(text) < 7:
            log.debug("Từ chối: '%s' — Quá ngắn (%d ký tự, cần tối thiểu 7)", text, len(text))
            return ""
        if len(text) > 9:
            log.debug("Từ chối: '%s' — Quá dài (%d ký tự, tối đa 9)", text, len(text))
            return ""

        # Bước 3: Áp dụng ép kiểu ký tự bắt buộc theo vị trí
        fixed_text = self.correct_plate_format(text)
        log.debug("Sau sửa lỗi vị trí: '%s'", fixed_text)

        # Bước 4: Validate bằng RegEx — chỉ chấp nhận nếu khớp cấu trúc biển số VN
        if self._validate_format(fixed_text):
            return fixed_text
        else:
            # Vẫn trả về kết quả đã sửa để nhân viên có thể tham khảo và nhập tay
            # Nếu muốn từ chối hoàn toàn, đổi dòng dưới thành: return ""
            log.debug("Cảnh báo: '%s' — Không khớp hoàn toàn nhưng vẫn trả về để tham khảo", fixed_text)
            return fixed_text

    def _validate_format(self, text):
        # Pattern: 2 số (mã tỉnh) + 1 chữ (series) + chữ/số tùy chọn + 4-5 số (đuôi)
        # Ví dụ hợp lệ: 51G12345, 29A12345, 30HK99999, 59V254411
        pattern = r'^[0-9]{2}[A-Z][A-Z0-9]?[0-9]{4,5}$'
        if re.match(pattern, text):
            log.debug("Biển số HỢP LỆ: %s", text)
            return True
        else:
            if len(text) < 7:
                log.debug("'%s' — Quá ngắn", text)
            elif len(text) > 9:
                log.debug("'%s' — Quá dài", text)
            else:
                log.debug("'%s' — Định dạng chưa khớp hoàn toàn", text)
            return False

    def _load_image(self, image):
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                print(f"Lỗi: Không tìm thấy file ảnh tại '{image}'")
            return img
        else:
            print(f"Lỗi: Kiểu dữ liệu không hỗ trợ: {type(image)}")
            return None

    def process_ocr(self, image):
        return self.read_plate(image)