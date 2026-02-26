"""
google_sheets.py – Google Sheets real-time sync for multi-user collaboration.

Uses raw Google Sheets API v4 (no extra dependency beyond google-api-python-client).
Supports auto-sharding: each sheet tab holds up to SHARD_MAX_ROWS rows.
When a shard is full, a new tab is created automatically.

Two spreadsheets on Google Drive:
  1. warehouse_sheets   – columns: id, text
  2. labeled_data_sheets – columns: id, text, tier1_spam, tier2_toxic, tier3_labels, labeled_by
"""

import os
import csv
import json as _json
import time
from typing import List, Dict, Optional, Callable, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =========================================================================== #
#  CONFIG
# =========================================================================== #

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(_PROJECT_ROOT, "credentials.json")
TOKEN_FILE = os.path.join(_PROJECT_ROOT, "token.json")

# Same Drive folder as google_drive.py
DRIVE_ROOT_ID = "1GkjNZ3QeD_tsOZLBHq7fhMw9yVzEc1Os"

# OAuth scopes – need both Drive (to locate/create spreadsheets) and Sheets
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# Each shard tab holds at most this many data rows (excl. header).
# Google Sheets limit ≈ 10 M cells.  With 2 cols for warehouse that's 5 M rows
# max per spreadsheet, but we cap each tab for performance.
SHARD_MAX_ROWS = 200_000

# Spreadsheet names stored on Google Drive
WAREHOUSE_SPREADSHEET_NAME = "warehouse_sheets"
LABELED_SPREADSHEET_NAME = "labeled_data_sheets"

# Column definitions
WAREHOUSE_COLUMNS = ["id", "text"]
LABELED_COLUMNS = ["id", "text", "tier1_spam", "tier2_toxic", "tier3_labels", "labeled_by"]

# Batch size for Sheets API value-appends (stay within quota)
_BATCH_WRITE_SIZE = 5_000


# =========================================================================== #
#  Custom exceptions
# =========================================================================== #

class SheetsAPINotEnabledError(Exception):
    """Raised when Google Sheets API is not enabled in the Cloud Console."""
    pass


# =========================================================================== #
#  Helpers
# =========================================================================== #

def _get_project_id() -> str:
    """Extract project_id from credentials.json for error messages."""
    try:
        with open(CREDENTIALS_FILE, "r") as f:
            data = _json.load(f)
        return data.get("installed", {}).get("project_id", "unknown")
    except Exception:
        return "unknown"


def _raise_if_api_not_enabled(e: HttpError):
    """If *e* is a 403 PERMISSION_DENIED, raise ``SheetsAPINotEnabledError``
    with a user-friendly Vietnamese message + direct enable link.
    """
    if e.resp.status == 403:
        err = str(e)
        if "PERMISSION_DENIED" in err or "forbidden" in err.lower():
            pid = _get_project_id()
            raise SheetsAPINotEnabledError(
                "Google Sheets API chưa được bật trong project Google Cloud!\n\n"
                "Cách khắc phục:\n"
                "1. Truy cập link:\n"
                f"   https://console.cloud.google.com/apis/library/"
                f"sheets.googleapis.com?project={pid}\n"
                "2. Nhấn nút 'Enable' (Bật)\n"
                "3. Đợi vài giây rồi thử lại.\n\n"
                f"(Project: {pid})"
            ) from e


# =========================================================================== #
#  Auth – reuse same OAuth flow / token.json as google_drive.py
# =========================================================================== #

_cached_creds: Optional[Credentials] = None


def _get_credentials() -> Credentials:
    """Return valid OAuth2 credentials, refreshing or prompting as needed.

    Handles two common issues transparently:
    1. ``token.json`` created with Drive-only scope → deletes and re-auths.
    2. Token expired → refreshes automatically.
    """
    global _cached_creds

    if _cached_creds and _cached_creds.valid:
        return _cached_creds

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

        # Check if token has the required spreadsheets scope
        if creds and creds.scopes:
            has_sheets = any("spreadsheets" in s for s in creds.scopes)
            if not has_sheets:
                # Token was created with Drive-only scope — must re-auth
                os.remove(TOKEN_FILE)
                creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Token may have wrong scopes – remove and re-auth
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                creds = None

        if not creds:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Thiếu file credentials.json tại: {CREDENTIALS_FILE}\n"
                    "Tải từ Google Cloud Console → OAuth Client ID."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    _cached_creds = creds
    return creds


def get_sheets_service():
    """Return an authenticated Google Sheets API v4 service."""
    return build("sheets", "v4", credentials=_get_credentials())


def get_drive_service():
    """Return an authenticated Google Drive API v3 service."""
    return build("drive", "v3", credentials=_get_credentials())


# =========================================================================== #
#  Drive helpers – find / create spreadsheets in the target folder
# =========================================================================== #

def _find_spreadsheet(drive_svc, name: str, parent_id: str) -> Optional[str]:
    """Return spreadsheet file-ID if it exists under *parent_id*, else None."""
    query = (
        f"name='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
    )
    results = drive_svc.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def _create_spreadsheet(drive_svc, sheets_svc, name: str, parent_id: str,
                        columns: List[str]) -> str:
    """Create a new Google Spreadsheet with a first shard tab; return file-ID."""
    body: Dict[str, Any] = {
        "properties": {"title": name},
        "sheets": [
            {
                "properties": {"title": "shard_0", "index": 0},
                "data": [
                    {
                        "startRow": 0,
                        "startColumn": 0,
                        "rowData": [
                            {
                                "values": [
                                    {"userEnteredValue": {"stringValue": c}}
                                    for c in columns
                                ]
                            }
                        ],
                    }
                ],
            }
        ],
    }
    try:
        spreadsheet = (
            sheets_svc.spreadsheets()
            .create(body=body, fields="spreadsheetId")
            .execute()
        )
    except HttpError as e:
        _raise_if_api_not_enabled(e)
        raise

    ss_id = spreadsheet["spreadsheetId"]

    # Move the new spreadsheet into the target Drive folder
    file_info = drive_svc.files().get(fileId=ss_id, fields="parents").execute()
    prev_parents = ",".join(file_info.get("parents", []))
    drive_svc.files().update(
        fileId=ss_id,
        addParents=parent_id,
        removeParents=prev_parents,
        fields="id, parents",
    ).execute()

    return ss_id


def get_or_create_spreadsheet(name: str, columns: List[str],
                              parent_id: str = None) -> str:
    """Return the spreadsheet file-ID, creating it if it doesn't exist."""
    parent_id = parent_id or DRIVE_ROOT_ID
    drive_svc = get_drive_service()
    ss_id = _find_spreadsheet(drive_svc, name, parent_id)
    if ss_id:
        return ss_id
    sheets_svc = get_sheets_service()
    return _create_spreadsheet(drive_svc, sheets_svc, name, parent_id, columns)


# =========================================================================== #
#  Shard management
# =========================================================================== #

def _get_sheet_tabs(sheets_svc, spreadsheet_id: str) -> List[Dict[str, Any]]:
    """Return list of sheet-tab metadata sorted by index."""
    try:
        meta = (
            sheets_svc.spreadsheets()
            .get(spreadsheetId=spreadsheet_id, fields="sheets.properties")
            .execute()
        )
    except HttpError as e:
        _raise_if_api_not_enabled(e)
        raise

    tabs: List[Dict[str, Any]] = []
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        tabs.append(
            {
                "sheetId": props.get("sheetId"),
                "title": props.get("title", ""),
                "index": props.get("index", 0),
            }
        )
    tabs.sort(key=lambda t: t["index"])
    return tabs


def _get_shard_row_count(sheets_svc, spreadsheet_id: str, tab_title: str) -> int:
    """Return number of data rows (excluding header) in a shard tab."""
    rng = f"'{tab_title}'!A1:A"
    try:
        result = (
            sheets_svc.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=rng)
            .execute()
        )
        values = result.get("values", [])
        return max(0, len(values) - 1)  # subtract header row
    except HttpError as e:
        _raise_if_api_not_enabled(e)
        return 0


def _create_new_shard(sheets_svc, spreadsheet_id: str, shard_index: int,
                      columns: List[str]) -> str:
    """Add a new shard tab with header row; return its title."""
    title = f"shard_{shard_index}"
    body: Dict[str, Any] = {
        "requests": [{"addSheet": {"properties": {"title": title}}}]
    }
    try:
        sheets_svc.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body=body
        ).execute()
    except HttpError as e:
        _raise_if_api_not_enabled(e)
        raise

    # Write header into the new tab
    try:
        sheets_svc.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{title}'!A1",
            valueInputOption="RAW",
            body={"values": [columns]},
        ).execute()
    except HttpError as e:
        _raise_if_api_not_enabled(e)
        raise

    return title


def _get_last_shard_info(sheets_svc, spreadsheet_id: str,
                         columns: List[str]):
    """Return ``(tab_title, current_row_count)`` of the last shard.

    Creates ``shard_0`` if the spreadsheet has no shard tabs.
    """
    tabs = _get_sheet_tabs(sheets_svc, spreadsheet_id)
    shard_tabs = [t for t in tabs if t["title"].startswith("shard_")]
    if not shard_tabs:
        title = _create_new_shard(sheets_svc, spreadsheet_id, 0, columns)
        return title, 0

    last = shard_tabs[-1]
    count = _get_shard_row_count(sheets_svc, spreadsheet_id, last["title"])
    return last["title"], count


# =========================================================================== #
#  Core read / write
# =========================================================================== #

def read_all_rows(spreadsheet_id: str, columns: List[str],
                  log: Callable[[str], None] = None) -> List[Dict[str, str]]:
    """Read **all** rows across every shard tab.  Returns list of dicts."""
    log = log or (lambda _m: None)
    sheets_svc = get_sheets_service()
    tabs = _get_sheet_tabs(sheets_svc, spreadsheet_id)
    shard_tabs = sorted(
        [t for t in tabs if t["title"].startswith("shard_")],
        key=lambda t: t["index"],
    )

    all_rows: List[Dict[str, str]] = []
    for tab in shard_tabs:
        title = tab["title"]
        rng = f"'{title}'!A:Z"
        try:
            result = (
                sheets_svc.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=rng)
                .execute()
            )
        except HttpError as e:
            _raise_if_api_not_enabled(e)
            log(f"[Sheets] Lỗi đọc tab {title}: {e}")
            continue

        values = result.get("values", [])
        if len(values) <= 1:
            continue  # header only or empty

        header = values[0]
        for row_vals in values[1:]:
            row_dict: Dict[str, str] = {}
            for i, col in enumerate(columns):
                if i < len(header) and i < len(row_vals):
                    row_dict[col] = row_vals[i]
                else:
                    row_dict[col] = ""
            all_rows.append(row_dict)

        log(f"[Sheets] Đọc {len(values) - 1} dòng từ {title}")

    log(f"[Sheets] Tổng: {len(all_rows)} dòng từ {len(shard_tabs)} shard(s)")
    return all_rows


def append_rows(spreadsheet_id: str, rows: List[Dict[str, str]],
                columns: List[str],
                log: Callable[[str], None] = None) -> int:
    """Append *rows* to the spreadsheet, auto-creating new shard tabs as needed.

    Returns the number of rows actually written.
    """
    if not rows:
        return 0

    log = log or (lambda _m: None)
    sheets_svc = get_sheets_service()

    written = 0
    remaining = list(rows)

    while remaining:
        tab_title, current_count = _get_last_shard_info(
            sheets_svc, spreadsheet_id, columns
        )
        capacity = SHARD_MAX_ROWS - current_count

        if capacity <= 0:
            # Current shard is full → create a new one
            shard_idx = int(tab_title.rsplit("_", 1)[1]) + 1
            tab_title = _create_new_shard(
                sheets_svc, spreadsheet_id, shard_idx, columns
            )
            capacity = SHARD_MAX_ROWS
            log(f"[Sheets] Shard trước đầy ({SHARD_MAX_ROWS} dòng) → tạo shard mới: {tab_title}")

        batch = remaining[:capacity]
        remaining = remaining[capacity:]

        # Convert dicts → list-of-lists for the API
        values = [[str(r.get(c, "")) for c in columns] for r in batch]

        # Write in sub-batches to stay within API quota / payload limits
        for i in range(0, len(values), _BATCH_WRITE_SIZE):
            chunk = values[i : i + _BATCH_WRITE_SIZE]
            try:
                sheets_svc.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{tab_title}'!A1",
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body={"values": chunk},
                ).execute()
            except HttpError as e:
                _raise_if_api_not_enabled(e)
                raise
            written += len(chunk)
            log(f"[Sheets] Đã ghi {written}/{len(rows)} dòng vào {tab_title}...")

        # Tiny delay between shards to stay within rate limits
        if remaining:
            time.sleep(0.3)

    log(f"[Sheets] Hoàn tất ghi {written} dòng")
    return written


def get_existing_ids(spreadsheet_id: str,
                     log: Callable[[str], None] = None) -> set:
    """Read **only** the ``id`` column across all shards.

    Much faster than ``read_all_rows`` when you only need to check for
    duplicates.
    """
    log = log or (lambda _m: None)
    sheets_svc = get_sheets_service()
    tabs = _get_sheet_tabs(sheets_svc, spreadsheet_id)
    shard_tabs = sorted(
        [t for t in tabs if t["title"].startswith("shard_")],
        key=lambda t: t["index"],
    )

    ids: set = set()
    for tab in shard_tabs:
        title = tab["title"]
        rng = f"'{title}'!A:A"
        try:
            result = (
                sheets_svc.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=rng)
                .execute()
            )
        except HttpError as e:
            _raise_if_api_not_enabled(e)
            continue
        values = result.get("values", [])
        for row in values[1:]:  # skip header
            if row:
                ids.add(row[0])

    log(f"[Sheets] Tìm thấy {len(ids)} ID trên cloud")
    return ids


def get_total_row_count(spreadsheet_id: str) -> int:
    """Return total data-row count across all shard tabs."""
    sheets_svc = get_sheets_service()
    tabs = _get_sheet_tabs(sheets_svc, spreadsheet_id)
    shard_tabs = [t for t in tabs if t["title"].startswith("shard_")]
    total = 0
    for tab in shard_tabs:
        total += _get_shard_row_count(sheets_svc, spreadsheet_id, tab["title"])
    return total


# =========================================================================== #
#  High-level sync: WAREHOUSE
# =========================================================================== #

def _get_warehouse_local_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "warehouse.csv")


def _read_local_warehouse(path: str = None) -> List[Dict[str, str]]:
    """Read local ``warehouse.csv`` into a list of dicts."""
    path = path or _get_warehouse_local_path()
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"id": row.get("id", ""), "text": row.get("text", "")})
    return rows


def _write_local_warehouse(rows: List[Dict[str, str]], path: str = None):
    """Overwrite local ``warehouse.csv``."""
    path = path or _get_warehouse_local_path()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WAREHOUSE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({"id": r.get("id", ""), "text": r.get("text", "")})


def sync_warehouse_push(log: Callable[[str], None] = None) -> int:
    """Push local warehouse rows → Google Sheets (append-only, dedup by ``id``).

    Returns the number of **new** rows pushed.
    """
    log = log or print
    ss_id = get_or_create_spreadsheet(WAREHOUSE_SPREADSHEET_NAME, WAREHOUSE_COLUMNS)
    log("[Sync] Đang đọc ID đã có trên cloud...")
    remote_ids = get_existing_ids(ss_id, log=log)

    local_rows = _read_local_warehouse()
    log(f"[Sync] Local: {len(local_rows)} dòng | Cloud: {len(remote_ids)} ID")

    new_rows = [r for r in local_rows if str(r.get("id", "")) not in remote_ids]
    if not new_rows:
        log("[Sync] ✓ Không có dòng mới cần push.")
        return 0

    log(f"[Sync] Push {len(new_rows)} dòng mới lên cloud (auto-shard mỗi {SHARD_MAX_ROWS} dòng)...")
    written = append_rows(ss_id, new_rows, WAREHOUSE_COLUMNS, log=log)
    log(f"[Sync] ✓ Đã push {written} dòng warehouse mới lên Google Sheets.")
    return written


def sync_warehouse_pull(log: Callable[[str], None] = None) -> int:
    """Pull warehouse rows from Google Sheets → merge into local (dedup by ``id``).

    Returns the number of **new** rows pulled into local.
    """
    log = log or print
    ss_id = get_or_create_spreadsheet(WAREHOUSE_SPREADSHEET_NAME, WAREHOUSE_COLUMNS)
    log("[Sync] Đang đọc dữ liệu warehouse từ cloud...")
    remote_rows = read_all_rows(ss_id, WAREHOUSE_COLUMNS, log=log)

    local_rows = _read_local_warehouse()
    local_ids = {str(r.get("id", "")) for r in local_rows}

    new_rows = [r for r in remote_rows if str(r.get("id", "")) not in local_ids]
    if not new_rows:
        log("[Sync] ✓ Local đã có đầy đủ dữ liệu, không có gì mới.")
        return 0

    # Merge: keep all local + append new from cloud, then sort by id
    merged = local_rows + new_rows
    try:
        merged.sort(key=lambda r: int(r.get("id", 0)))
    except (ValueError, TypeError):
        pass

    _write_local_warehouse(merged)
    log(f"[Sync] ✓ Pull {len(new_rows)} dòng mới → local warehouse ({len(merged)} tổng).")
    return len(new_rows)


# =========================================================================== #
#  High-level sync: LABELED DATA
# =========================================================================== #

def _get_labeled_local_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_data.csv")


def _read_local_labeled(path: str = None) -> List[Dict[str, str]]:
    """Read local ``labeled_data.csv`` into a list of dicts."""
    path = path or _get_labeled_local_path()
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "tier1_spam": row.get("tier1_spam", "Not Spam"),
                    "tier2_toxic": row.get("tier2_toxic", "Clean"),
                    "tier3_labels": row.get("tier3_labels", ""),
                    "labeled_by": row.get("labeled_by", "unknown"),
                }
            )
    return rows


def _write_local_labeled(rows: List[Dict[str, str]], path: str = None):
    """Overwrite local ``labeled_data.csv``."""
    path = path or _get_labeled_local_path()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LABELED_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in LABELED_COLUMNS})


def sync_labeled_push(labeled_by: str = "unknown",
                      log: Callable[[str], None] = None) -> int:
    """Push local labeled rows → Google Sheets (append-only, dedup by ``id``).

    *labeled_by* is stamped onto rows that don't already have a value.
    Returns the number of **new** rows pushed.
    """
    log = log or print
    ss_id = get_or_create_spreadsheet(LABELED_SPREADSHEET_NAME, LABELED_COLUMNS)
    log("[Sync] Đang đọc ID labeled đã có trên cloud...")
    remote_ids = get_existing_ids(ss_id, log=log)

    local_rows = _read_local_labeled()
    # Tag rows with labeled_by if not set
    for r in local_rows:
        if not r.get("labeled_by") or r["labeled_by"] == "unknown":
            r["labeled_by"] = labeled_by

    log(f"[Sync] Local labeled: {len(local_rows)} dòng | Cloud: {len(remote_ids)} ID")

    new_rows = [r for r in local_rows if str(r.get("id", "")) not in remote_ids]
    if not new_rows:
        log("[Sync] ✓ Không có dòng labeled mới cần push.")
        return 0

    log(f"[Sync] Push {len(new_rows)} dòng labeled mới lên cloud (auto-shard mỗi {SHARD_MAX_ROWS} dòng)...")
    written = append_rows(ss_id, new_rows, LABELED_COLUMNS, log=log)
    log(f"[Sync] ✓ Đã push {written} dòng labeled mới lên Google Sheets.")
    return written


def sync_labeled_pull(log: Callable[[str], None] = None) -> int:
    """Pull labeled rows from Google Sheets → merge into local (dedup by ``id``).

    Returns the number of **new** rows pulled into local.
    """
    log = log or print
    ss_id = get_or_create_spreadsheet(LABELED_SPREADSHEET_NAME, LABELED_COLUMNS)
    log("[Sync] Đang đọc labeled data từ cloud...")
    remote_rows = read_all_rows(ss_id, LABELED_COLUMNS, log=log)

    local_rows = _read_local_labeled()
    local_ids = {str(r.get("id", "")) for r in local_rows}

    new_rows = [r for r in remote_rows if str(r.get("id", "")) not in local_ids]
    if not new_rows:
        log("[Sync] ✓ Local đã có đầy đủ labeled data, không có gì mới.")
        return 0

    merged = local_rows + new_rows
    try:
        merged.sort(key=lambda r: int(r.get("id", 0)))
    except (ValueError, TypeError):
        pass

    _write_local_labeled(merged)
    log(f"[Sync] ✓ Pull {len(new_rows)} dòng labeled mới → local ({len(merged)} tổng).")
    return len(new_rows)


def get_cloud_stats(log: Callable[[str], None] = None) -> Dict[str, int]:
    """Return row counts on cloud for both spreadsheets.

    Returns ``{"warehouse": N, "labeled": M}`` (``-1`` on error).
    """
    log = log or (lambda _m: None)
    stats: Dict[str, int] = {}
    try:
        wh_id = get_or_create_spreadsheet(
            WAREHOUSE_SPREADSHEET_NAME, WAREHOUSE_COLUMNS
        )
        stats["warehouse"] = get_total_row_count(wh_id)
    except Exception:
        stats["warehouse"] = -1
    try:
        lb_id = get_or_create_spreadsheet(
            LABELED_SPREADSHEET_NAME, LABELED_COLUMNS
        )
        stats["labeled"] = get_total_row_count(lb_id)
    except Exception:
        stats["labeled"] = -1
    return stats
