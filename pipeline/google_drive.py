"""
google_drive.py – Google Drive sync for pipeline.
Upload/download warehouse.csv (and other data) to/from Google Drive.
Adapted from src/utils/google_drive.py with pipeline-relative paths.
"""
import os
import io
from datetime import datetime, timezone
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --- CONFIG ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(_PROJECT_ROOT, 'credentials.json')
TOKEN_FILE = os.path.join(_PROJECT_ROOT, 'token.json')
DRIVE_ROOT_ID = '1GkjNZ3QeD_tsOZLBHq7fhMw9yVzEc1Os'
SCOPES = ['https://www.googleapis.com/auth/drive']


# =========================================================================== #
#  Service / Auth
# =========================================================================== #

def get_service():
    """Get an authenticated Google Drive API service via OAuth 2.0."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                os.remove(TOKEN_FILE)
                return get_service()
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Thiếu file credentials.json tại: {CREDENTIALS_FILE}\n"
                    "Tải từ Google Cloud Console → OAuth Client ID."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


# =========================================================================== #
#  Helpers
# =========================================================================== #

def parse_drive_time(time_str: str) -> datetime:
    """Chuyển đổi chuỗi thời gian của Drive (ISO 8601) sang đối tượng datetime UTC."""
    # Drive API trả về định dạng "2023-10-25T12:00:00.000Z", thay Z thành +00:00 để dễ parse
    time_str = time_str.replace("Z", "+00:00")
    return datetime.fromisoformat(time_str)

def get_local_time(local_path: str) -> datetime:
    """Lấy thời gian chỉnh sửa cuối cùng của file local (UTC)."""
    mtime = os.path.getmtime(local_path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc)

def _get_file_info(service, name: str, parent_id: str):
    """Trả về dict chứa 'id' và 'modifiedTime' nếu file tồn tại, ngược lại None."""
    query = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    # Lấy thêm trường modifiedTime
    results = service.files().list(q=query, fields="files(id, modifiedTime)").execute()
    files = results.get('files', [])
    return files[0] if files else None



def _check_file_exists(service, name: str, parent_id: str):
    """Return file ID if *name* exists under *parent_id*, else None."""
    query = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None


def _get_or_create_folder(service, folder_name: str, parent_id: str) -> str:
    """Return folder ID, creating if necessary."""
    fid = _check_file_exists(service, folder_name, parent_id)
    if fid:
        return fid
    metadata = {
        'name': folder_name,
        'parents': [parent_id],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=metadata, fields='id').execute()
    return folder.get('id')


# =========================================================================== #
#  Upload single file
# =========================================================================== #

def upload_file(local_path: str, drive_folder_id: str = None,
                force: bool = False, log_callback=None) -> str:
    """Upload file. Nếu file Drive mới hơn file local, sẽ chặn ghi đè trừ khi force=True."""
    drive_folder_id = drive_folder_id or DRIVE_ROOT_ID

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"File không tồn tại: {local_path}")

    service = get_service()
    filename = os.path.basename(local_path)

    # Lấy thông tin file trên Drive
    file_info = _get_file_info(service, filename, drive_folder_id)
    media = MediaFileUpload(local_path, resumable=True)

    if file_info:
        existing_id = file_info['id']
        remote_time_str = file_info.get('modifiedTime')

        # So sánh thời gian
        if remote_time_str and not force:
            remote_time = parse_drive_time(remote_time_str)
            local_time = get_local_time(local_path)

            if remote_time > local_time:
                log(f"[Cảnh báo] File '{filename}' trên Drive MỚI HƠN file local. Bỏ qua upload!")
                log(f"  - Drive: {remote_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
                log(f"  - Local: {local_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
                log("  -> Dùng force=True nếu vẫn muốn ghi đè lên Drive.")
                return existing_id  # Trả về ID cũ, không upload gì cả

        log(f"[Drive] Cập nhật file đã có: {filename}")
        updated = service.files().update(fileId=existing_id, media_body=media, fields='id').execute()
        return updated['id']
    else:
        log(f"[Drive] Upload file mới: {filename}")
        metadata = {'name': filename, 'parents': [drive_folder_id]}
        created = service.files().create(body=metadata, media_body=media, fields='id').execute()
        return created['id']


# =========================================================================== #
#  Download single file
# =========================================================================== #

def download_file(filename: str, local_path: str, drive_folder_id: str = None,
                  force: bool = False, log_callback=None) -> bool:
    """Download file. Nếu file local mới hơn file Drive, sẽ chặn tải về trừ khi force=True."""
    drive_folder_id = drive_folder_id or DRIVE_ROOT_ID

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    service = get_service()
    file_info = _get_file_info(service, filename, drive_folder_id)

    if not file_info:
        log(f"[Drive] Không tìm thấy file '{filename}' trên Drive.")
        return False

    # So sánh thời gian nếu file local đã tồn tại
    if os.path.exists(local_path) and not force:
        remote_time_str = file_info.get('modifiedTime')
        if remote_time_str:
            remote_time = parse_drive_time(remote_time_str)
            local_time = get_local_time(local_path)

            if local_time > remote_time:
                log(f"[Cảnh báo] File local '{filename}' MỚI HƠN file trên Drive. Bỏ qua download!")
                log("  -> Dùng force=True nếu bạn muốn tải bản cũ từ Drive về đè lên bản local.")
                return False

    log(f"[Drive] Đang tải xuống: {filename}...")
    request = service.files().get_media(fileId=file_info['id'])
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            log(f"[Drive] Đang tải... {pct}%")

    fh.close()
    log(f"[Drive] Hoàn tất tải xuống: {filename}")
    return True


# =========================================================================== #
#  Warehouse-specific convenience functions
# =========================================================================== #

def upload_warehouse(warehouse_path: str = None, log_callback=None) -> str:
    """Upload warehouse.csv to Drive root folder."""
    if warehouse_path is None:
        warehouse_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "warehouse.csv"
        )
    return upload_file(warehouse_path, DRIVE_ROOT_ID, log_callback=log_callback)


def download_warehouse(warehouse_path: str = None, log_callback=None) -> bool:
    """Download warehouse.csv from Drive root folder."""
    if warehouse_path is None:
        warehouse_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "warehouse.csv"
        )
    return download_file("warehouse.csv", warehouse_path, DRIVE_ROOT_ID,
                         log_callback=log_callback)
