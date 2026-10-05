Write-Host "Đang dọn dẹp môi trường cũ (nếu có)..." -ForegroundColor Cyan
if (Test-Path ".venv") { Remove-Item -Path ".venv" -Recurse -Force }
if (Test-Path "venv") { Remove-Item -Path "venv" -Recurse -Force }

Write-Host "Đang tạo môi trường ảo .venv mới..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "Đang nâng cấp pip và cài đặt thư viện từ requirements.txt..." -ForegroundColor Cyan
# Gọi trực tiếp python.exe và pip.exe trong môi trường ảo mà không cần Activate trước
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -r requirements.txt

Write-Host "Hoàn tất cài đặt môi trường! Hãy gõ lệnh sau để kích hoạt:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow