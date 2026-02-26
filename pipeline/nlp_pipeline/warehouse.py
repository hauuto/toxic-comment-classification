"""
warehouse.py – Append-only CSV warehouse for all crawled & preprocessed comments.
File format: id,text (content-hash id for cross-machine dedup).
Now lives inside nlp_pipeline/ package.
"""
import os
import csv
import hashlib
import threading
import math
from typing import Optional, List, Dict

_lock = threading.Lock()


def text_hash_id(text: str) -> str:
    """Return a deterministic 16-char hex ID derived from *text*.

    Using SHA-256 truncated to 16 hex chars (64 bits).  Two machines that
    crawl the same comment will produce the **same** ID, which lets the
    dedup logic in ``append_to_warehouse`` and Google Sheets sync work
    correctly even when multiple users crawl in parallel.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


CLUSTER_SIZE_DEFAULT = 25_000

def _default_path() -> str:
    """Default warehouse path: pipeline/warehouse.csv (one level up from nlp_pipeline/)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "warehouse.csv")


def _get_existing_ids(warehouse_path: str) -> set:
    """Return the set of all IDs currently in the warehouse file."""
    ids: set = set()
    if not os.path.isfile(warehouse_path):
        return ids
    try:
        with open(warehouse_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get("id", "")
                if rid:
                    ids.add(str(rid))
    except Exception:
        pass
    return ids


def append_to_warehouse(rows: list, warehouse_path: str = None) -> int:
    """Append rows to warehouse.csv (dedup by content-hash ID).

    Parameters
    ----------
    rows : list[dict]
        Each dict must have key ``"text"``.  If ``"id"`` is present it is
        used as-is; otherwise a content-hash ID is generated via
        ``text_hash_id(text)``.
    warehouse_path : str, optional
        Path to the warehouse CSV.  Defaults to ``pipeline/warehouse.csv``.

    Returns
    -------
    int
        Number of rows actually written (after dedup).
    """
    if not rows:
        return 0

    warehouse_path = warehouse_path or _default_path()

    with _lock:
        file_exists = os.path.isfile(warehouse_path)
        existing_ids = _get_existing_ids(warehouse_path) if file_exists else set()

        new_rows = []
        for row in rows:
            text = row.get("text", "")
            rid = str(row.get("id", "")) or text_hash_id(text)
            if rid not in existing_ids:
                existing_ids.add(rid)
                new_rows.append({"id": rid, "text": text})

        if not new_rows:
            return 0

        with open(warehouse_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text"])
            if not file_exists:
                writer.writeheader()
            for r in new_rows:
                writer.writerow(r)

    return len(new_rows)


def get_warehouse_count(warehouse_path: str = None) -> int:
    """Return total number of data rows in the warehouse."""
    warehouse_path = warehouse_path or _default_path()
    if not os.path.isfile(warehouse_path):
        return 0
    count = 0
    try:
        with open(warehouse_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for _ in reader:
                count += 1
    except Exception:
        pass
    return count


def get_warehouse_cluster_count(cluster_size: int = CLUSTER_SIZE_DEFAULT, warehouse_path: str = None) -> int:
    """Return number of clusters for current warehouse.

    A cluster is a contiguous chunk of `cluster_size` rows, counted top-down.
    """
    if cluster_size <= 0:
        cluster_size = CLUSTER_SIZE_DEFAULT
    total = get_warehouse_count(warehouse_path=warehouse_path)
    return int(math.ceil(total / cluster_size)) if total > 0 else 0


def get_warehouse_clusters(cluster_size: int = CLUSTER_SIZE_DEFAULT, warehouse_path: str = None) -> List[Dict[str, int]]:
    """Return cluster metadata list.

    Each item: {"cluster_index": i, "start_row": a, "end_row": b, "size": n}
    where start_row/end_row are 0-based row offsets (excluding header), inclusive.
    """
    if cluster_size <= 0:
        cluster_size = CLUSTER_SIZE_DEFAULT
    total = get_warehouse_count(warehouse_path=warehouse_path)
    if total <= 0:
        return []

    clusters: List[Dict[str, int]] = []
    count = int(math.ceil(total / cluster_size))
    for i in range(count):
        start = i * cluster_size
        end = min(total - 1, start + cluster_size - 1)
        clusters.append(
            {
                "cluster_index": i,
                "start_row": start,
                "end_row": end,
                "size": end - start + 1,
            }
        )
    return clusters


def read_warehouse_slice(offset: int, limit: int, warehouse_path: str = None) -> list:
    """Read a slice of rows from warehouse.csv without loading the entire file.

    Parameters
    ----------
    offset : int
        0-based row offset (excluding header).
    limit : int
        Maximum number of rows to return.
    """
    warehouse_path = warehouse_path or _default_path()
    if not os.path.isfile(warehouse_path):
        return []
    if offset < 0:
        offset = 0
    if limit <= 0:
        return []

    rows = []
    try:
        with open(warehouse_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Skip `offset` data rows
            skipped = 0
            for row in reader:
                if skipped < offset:
                    skipped += 1
                    continue
                rows.append({"id": str(row.get("id", "")), "text": row.get("text", "")})
                if len(rows) >= limit:
                    break
    except Exception:
        return []

    return rows


def read_warehouse_cluster(cluster_index: int, cluster_size: int = CLUSTER_SIZE_DEFAULT, warehouse_path: str = None) -> list:
    """Read one cluster by index (top-down)."""
    if cluster_size <= 0:
        cluster_size = CLUSTER_SIZE_DEFAULT
    if cluster_index < 0:
        cluster_index = 0

    offset = cluster_index * cluster_size
    return read_warehouse_slice(offset=offset, limit=cluster_size, warehouse_path=warehouse_path)


def read_warehouse(warehouse_path: str = None) -> list:
    """Read all rows from warehouse.csv.

    Returns
    -------
    list[dict]
        Each dict has keys ``"id"`` (str) and ``"text"`` (str).
    """
    warehouse_path = warehouse_path or _default_path()
    if not os.path.isfile(warehouse_path):
        return []
    rows = []
    try:
        with open(warehouse_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({"id": str(row.get("id", "")), "text": row.get("text", "")})
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
                writer.writerow({"id": row.get("id", ""), "text": row.get("text", "")})
    return len(rows)
