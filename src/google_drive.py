import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import io
from googleapiclient.http import MediaIoBaseDownload
# --- CAU HINH ---
# Ban can file credentials.json (tai tu Cloud Console chon OAuth Client ID)
CREDENTIALS_FILE = '../credentials.json'
TOKEN_FILE = '../token.json'
# ID folder 'data' tren Drive
DRIVE_ROOT_ID = '1GkjNZ3QeD_tsOZLBHq7fhMw9yVzEc1Os'
LOCAL_DATA_FOLDER = '../data'
SCOPES = ['https://www.googleapis.com/auth/drive']

# --- CAC HAM HO TRO ---

def get_service():
    """Lay ket noi thong qua OAuth 2.0 (token.json)."""
    creds = None
    # Kiem tra token da luu chua
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Neu chua co hoac het han thi dang nhap lai
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Neu refresh loi thi xoa token cu di de dang nhap lai
                os.remove(TOKEN_FILE)
                return get_service()
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"Loi: Thieu file {CREDENTIALS_FILE}")
                return None

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Luu token lai cho lan sau (hoac gui cho dong nghiep)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def check_file_exists(service, name, parent_id):
    query = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

def get_or_create_folder(service, folder_name, parent_id):
    folder_id = check_file_exists(service, folder_name, parent_id)
    if folder_id:
        return folder_id

    print(f"Tao folder moi: {folder_name}")
    metadata = {
        'name': folder_name,
        'parents': [parent_id],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=metadata, fields='id').execute()
    return folder.get('id')

# --- HAM UPLOAD DE QUY ---

def upload_recursive(service, local_path, drive_parent_id):
    # Kiem tra thu muc local co ton tai khong
    if not os.path.exists(local_path):
        print(f"Loi: Duong dan khong ton tai: {local_path}")
        return

    for item in os.listdir(local_path):
        item_path = os.path.join(local_path, item)

        if item.startswith('.'):
            continue

        if os.path.isfile(item_path):
            if check_file_exists(service, item, drive_parent_id):
                print(f"Bo qua (da co): {item}")
            else:
                print(f"Dang upload: {item}...")
                media = MediaFileUpload(item_path)
                metadata = {'name': item, 'parents': [drive_parent_id]}
                try:
                    service.files().create(
                        body=metadata, media_body=media, fields='id'
                    ).execute()
                    print(f"Xong: {item}")
                except Exception as e:
                    print(f"Loi upload {item}: {e}")

        elif os.path.isdir(item_path):
            sub_folder_id = get_or_create_folder(service, item, drive_parent_id)
            upload_recursive(service, item_path, sub_folder_id)

def upload_all_data():
    print(f"Bat dau upload tu: {LOCAL_DATA_FOLDER}")
    service = get_service()
    if service:
        upload_recursive(service, LOCAL_DATA_FOLDER, DRIVE_ROOT_ID)
        print("Hoan tat qua trinh upload.")



def download_recursive(service, drive_folder_id, local_path):
    # 1. Tao thu muc local neu chua co
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # 2. Lay danh sach file/folder trong folder hien tai
    try:
        query = f"'{drive_folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            pageSize=1000,
            fields="files(id, name, mimeType)"
        ).execute()

        items = results.get('files', [])
    except Exception as e:
        print(f"Loi khi liet ke file: {e}")
        return

    for item in items:
        name = item['name']
        item_id = item['id']
        mime_type = item['mimeType']

        # Duong dan tuong ung tren may tinh
        local_item_path = os.path.join(local_path, name)

        # TRUONG HOP 1: LA FOLDER
        if mime_type == 'application/vnd.google-apps.folder':
            # De quy: Chui vao folder con tai tiep (Truyen service vao)
            download_recursive(service, item_id, local_item_path)

        # TRUONG HOP 2: LA FILE
        else:
            # Check trung: Neu file da co thi bo qua
            if os.path.exists(local_item_path):
                print(f"Bo qua (da co): {name}")
                continue

            # Bat dau tai
            print(f"Dang tai: {name}...")
            try:
                request = service.files().get_media(fileId=item_id)
                fh = io.FileIO(local_item_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while done is False:
                    status, done = downloader.next_chunk()

                print(f"Xong: {name}")
            except Exception as e:
                print(f"Loi khi tai {name}: {e}")
                # Xoa file loi neu tai do dang
                if os.path.exists(local_item_path):
                    os.remove(local_item_path)

# --- HÀM 2: MAIN FUNCTION (Gọi hàm này để chạy) ---
def fetch_data(drive_folder_id=DRIVE_ROOT_ID, local_path=LOCAL_DATA_FOLDER):
    print(f"Bat dau dong bo tu Drive ({drive_folder_id}) ve ({local_path})...")

    # Khoi tao service 1 lan duy nhat o day
    service = get_service()

    if not service:
        print("Loi: Khong the ket noi den Drive (Kiem tra credentials/token).")
        return

    # Bat dau qua trinh de quy
    download_recursive(service, drive_folder_id, local_path)

    print("Hoan tat qua trinh fetch data.")