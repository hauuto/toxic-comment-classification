"""
google_drive.py – Google Drive sync for pipeline.
Upload/download warehouse.csv (and other data) to/from Google Drive.
Adapted from src/utils/google_drive.py with pipeline-relative paths.
"""
import os
import io
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
                log_callback=None) -> str:
    """Upload a single file to Google Drive.

    Parameters
    ----------
    local_path : str
        Full path to the local file.
    drive_folder_id : str, optional
        Target folder on Drive. Defaults to DRIVE_ROOT_ID.
    log_callback : callable, optional
        ``log_callback(msg)`` for status messages.

    Returns
    -------
    str
        The file ID on Drive.
    """
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

    # Check if file already exists → update instead of creating duplicate
    existing_id = _check_file_exists(service, filename, drive_folder_id)

    media = MediaFileUpload(local_path, resumable=True)

    if existing_id:
        log(f"[Drive] Cập nhật file đã có: {filename}")
        updated = service.files().update(
            fileId=existing_id,
            media_body=media,
            fields='id'
        ).execute()
        log(f"[Drive] Hoàn tất cập nhật: {filename} (ID: {updated['id']})")
        return updated['id']
    else:
        log(f"[Drive] Upload file mới: {filename}")
        metadata = {'name': filename, 'parents': [drive_folder_id]}
        created = service.files().create(
            body=metadata,
            media_body=media,
            fields='id'
        ).execute()
        log(f"[Drive] Hoàn tất upload: {filename} (ID: {created['id']})")
        return created['id']


# =========================================================================== #
#  Download single file
# =========================================================================== #

def download_file(filename: str, local_path: str, drive_folder_id: str = None,
                  log_callback=None) -> bool:
    """Download a file from Google Drive to a local path.

    Parameters
    ----------
    filename : str
        Name of the file on Drive.
    local_path : str
        Full local path to save the file to.
    drive_folder_id : str, optional
        Source folder on Drive. Defaults to DRIVE_ROOT_ID.
    log_callback : callable, optional
        For status messages.

    Returns
    -------
    bool
        True if downloaded successfully.
    """
    drive_folder_id = drive_folder_id or DRIVE_ROOT_ID

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    service = get_service()
    file_id = _check_file_exists(service, filename, drive_folder_id)

    if not file_id:
        log(f"[Drive] Không tìm thấy file '{filename}' trên Drive.")
        return False

    log(f"[Drive] Đang tải xuống: {filename}...")
    request = service.files().get_media(fileId=file_id)
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
