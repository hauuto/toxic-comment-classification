"""
warehouse.py – Append-only CSV warehouse for all crawled & preprocessed comments.
File format: id,text (auto-increment id, permanent accumulation).
Now lives inside nlp_pipeline/ package.
"""
import os
import csv
import threading

_lock = threading.Lock()

def _default_path() -> str:
    """Default warehouse path: pipeline/warehouse.csv (one level up from nlp_pipeline/)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "warehouse.csv")


def _get_max_id(warehouse_path: str) -> int:
    """Return the current max id in the warehouse (0 if empty/missing)."""
    if not os.path.isfile(warehouse_path):
        return 0
    max_id = 0
    try:
        with open(warehouse_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rid = int(row["id"])
                    if rid > max_id:
                        max_id = rid
                except (ValueError, KeyError):
                    pass
    except Exception:
        pass
    return max_id


def append_to_warehouse(rows: list, warehouse_path: str = None) -> int:
    """Append rows to warehouse.csv.

    Parameters
    ----------
    rows : list[dict]
        Each dict must have key ``"text"``.  The ``"id"`` key is ignored –
        ids are auto-assigned based on the current max id in the file.
    warehouse_path : str, optional
        Path to the warehouse CSV.  Defaults to ``pipeline/warehouse.csv``.

    Returns
    -------
    int
        Number of rows actually written.
    """
    if not rows:
        return 0

    warehouse_path = warehouse_path or _default_path()

    with _lock:
        file_exists = os.path.isfile(warehouse_path)
        start_id = _get_max_id(warehouse_path) + 1

        with open(warehouse_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text"])
            if not file_exists:
                writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow({"id": start_id + i, "text": row.get("text", "")})

    return len(rows)


def get_warehouse_count(warehouse_path: str = None) -> int:
    """Return total number of data rows in the warehouse."""
    warehouse_path = warehouse_path or _default_path()
    if not os.path.isfile(warehouse_path):
        return 0
    count = 0
    try:
        with open(warehouse_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for _ in reader:
                count += 1
    except Exception:
        pass
    return count


def read_warehouse(warehouse_path: str = None) -> list:
    """Read all rows from warehouse.csv.

    Returns
    -------
    list[dict]
        Each dict has keys ``"id"`` (int) and ``"text"`` (str).
    """
    warehouse_path = warehouse_path or _default_path()
    if not os.path.isfile(warehouse_path):
        return []
    rows = []
    try:
        with open(warehouse_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({"id": int(row.get("id", 0)), "text": row.get("text", "")})
    except Exception:
        pass
    return rows


def overwrite_warehouse(rows: list, warehouse_path: str = None) -> int:
    """Completely rewrite the warehouse with the given rows.

    Parameters
    ----------
    rows : list[dict]
        Each dict must have ``"id"`` and ``"text"`` keys.

    Returns
    -------
    int
        Number of rows written.
    """
    warehouse_path = warehouse_path or _default_path()
    with _lock:
        with open(warehouse_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"id": row.get("id", 0), "text": row.get("text", "")})
    return len(rows)
