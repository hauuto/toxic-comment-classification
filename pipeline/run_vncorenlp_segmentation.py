"""
Standalone script – Chạy VnCoreNLP word segmentation trên warehouse.csv.

Kiến trúc: load VnCoreNLP trực tiếp trong process, xử lý theo batch (BATCH_SIZE=16).
Dùng word_segment() thay vì annotate_text() — chỉ tách từ, bỏ POS/NER/dep parsing.
Checkpoint mỗi batch → nếu crash, chạy lại để tiếp tục.

Chạy:
    cd pipeline
    python run_vncorenlp_segmentation.py
"""

import csv
import os
import re
import sys
import time

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
WAREHOUSE_PATH = os.path.join(SCRIPT_DIR, "warehouse.csv")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "warehouse_segmentation_checkpoint.txt")
CURRENT_PATH = os.path.join(SCRIPT_DIR, "warehouse_segmentation_current.txt")
POISON_LOG_PATH = os.path.join(SCRIPT_DIR, "warehouse_segmentation_poison.txt")

# ── Config ──
BATCH_SIZE = 32       # number of rows per VnCoreNLP call
SAVE_INTERVAL = 2000  # save warehouse to disk every N rows

sys.path.insert(0, SCRIPT_DIR)

# ── Emoji canonicalization + placeholder protection ──
_EMOJI_CANON_RE = re.compile(
    r":\s*([a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+"
    r"(?:\s*_\s*[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+)*)\s*:",
    re.UNICODE | re.IGNORECASE,
)

_PLACEHOLDER_RE = re.compile(
    r"<(?:url|mention|hashtag|email|date|time|num|ip)>|"
    r":[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+:|"
    r"(?<!\w)[a-zA-Z]+[''][a-zA-Z]+(?!\w)",
    re.UNICODE | re.IGNORECASE,
)

_EMOJI_TOKEN_SPACED_RE = re.compile(
    r":\s*(?P<name>[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+"
    r"(?:\s*_\s*[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+)*)\s*:",
    re.UNICODE | re.IGNORECASE,
)

_CANON_PLACEHOLDER_RE = re.compile(
    r"<\s*(url|mention|hashtag|email|date|time|num|ip)\s*>",
    re.IGNORECASE,
)


def _canonicalize_emoji(text):
    def _canon(m):
        name = m.group(1)
        name = re.sub(r"\s*_\s*", "_", name.strip())
        return f":{name}:"
    result = text
    for _ in range(5):
        new = _EMOJI_CANON_RE.sub(_canon, result)
        if new == result:
            break
        result = new
    return result


def _canonicalize_emoji_tokens(text):
    if not text:
        return text
    def _fix(m):
        name = m.group("name")
        name = re.sub(r"\s*_\s*", "_", name.strip())
        return f":{name}:"
    return _EMOJI_TOKEN_SPACED_RE.sub(_fix, text)


def _canonicalize_placeholders(text):
    if not text:
        return text
    return _CANON_PLACEHOLDER_RE.sub(lambda m: f"<{m.group(1).upper()}>", text)


def _protect(text):
    store = {}
    counter = [0]
    def _replace(m):
        marker = f"XPHX{counter[0]}XPHX"
        store[marker] = m.group()
        counter[0] += 1
        return marker
    canonicalized = _canonicalize_emoji(text)
    protected = _PLACEHOLDER_RE.sub(_replace, canonicalized)
    return protected, store


def _restore(text, store):
    for marker, original in store.items():
        text = text.replace(marker, original)
    return text


# ── Checkpoint helpers ──
def _read_checkpoint():
    if os.path.isfile(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return -1


def _write_checkpoint(idx):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(idx))


def _remove_checkpoint():
    if os.path.isfile(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


# ── Current-row tracker (detect JVM crash) ──
def _write_current(idx):
    """Write the index of the row we're about to process.
    If the process crashes, this file tells us which row caused it."""
    with open(CURRENT_PATH, "w") as f:
        f.write(str(idx))


def _read_current():
    """Read the last 'current' row index. Returns -1 if not found."""
    if os.path.isfile(CURRENT_PATH):
        try:
            with open(CURRENT_PATH, "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return -1


def _clear_current():
    """Mark current row as done (write -1)."""
    with open(CURRENT_PATH, "w") as f:
        f.write("-1")


def _remove_current():
    if os.path.isfile(CURRENT_PATH):
        os.remove(CURRENT_PATH)


# ── Poison row tracking ──
def _load_poison_rows():
    """Load set of row indices known to crash VnCoreNLP."""
    if not os.path.isfile(POISON_LOG_PATH):
        return set()
    result = set()
    try:
        with open(POISON_LOG_PATH, "r") as f:
            for line in f:
                part = line.split("#")[0].strip()
                if part.isdigit():
                    result.add(int(part))
    except Exception:
        pass
    return result


def _add_poison_row(idx, row_id, text_len):
    """Append a poison row entry to the log file."""
    with open(POISON_LOG_PATH, "a") as f:
        f.write(f"{idx}  # id={row_id} len={text_len}\n")
    print(f"  ☠ Dòng {row_id} (index={idx}, len={text_len}) → POISON (gây crash JVM)")


def _remove_poison_log():
    if os.path.isfile(POISON_LOG_PATH):
        os.remove(POISON_LOG_PATH)


def _save_warehouse(rows):
    with open(WAREHOUSE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(rows)


def _sanitize_text(text):
    """Remove surrogates and null bytes that crash JNI/JVM."""
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    text = text.replace("\x00", "")
    return text


# =====================================================================
#  MAIN — direct in-process VnCoreNLP
# =====================================================================
def main():
    if not os.path.isfile(WAREHOUSE_PATH):
        print(f"[ERROR] Không tìm thấy {WAREHOUSE_PATH}")
        sys.exit(1)

    # 1. Read CSV
    print(f"[1/3] Đọc {WAREHOUSE_PATH}...")
    rows = []
    with open(WAREHOUSE_PATH, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    total = len(rows)
    print(f"       → {total} dòng")

    if total == 0:
        print("[DONE] Warehouse trống.")
        return

    # 2. Checkpoint + poison detection
    resume_from = _read_checkpoint()
    poison_rows = _load_poison_rows()

    # Detect crash: if "current" file has a valid index > checkpoint,
    # that row crashed the JVM last run → mark as poison
    crash_row = _read_current()
    if crash_row >= 0 and crash_row > resume_from:
        row_data = rows[crash_row] if crash_row < total else {}
        text_len = len(row_data.get("text", ""))
        print(f"  ☠ Phát hiện dòng {row_data.get('id', '?')} (index={crash_row}, len={text_len}) đã crash JVM!")
        if crash_row not in poison_rows:
            _add_poison_row(crash_row, row_data.get("id", "?"), text_len)
            poison_rows.add(crash_row)
    _remove_current()

    if resume_from >= 0:
        print(f"  ★ Checkpoint: tiếp tục từ dòng {resume_from + 2}/{total}")
    else:
        resume_from = -1

    if poison_rows:
        print(f"  ☠ {len(poison_rows)} dòng poison sẽ skip")

    # 3. Load VnCoreNLP directly in-process
    print(f"[2/3] Khởi tạo VnCoreNLP từ {MODELS_DIR}...")
    import py_vncorenlp
    model = py_vncorenlp.VnCoreNLP(save_dir=MODELS_DIR)
    print(f"       → VnCoreNLP sẵn sàng")

    # 4. Pre-process: collect rows that need segmentation
    print(f"[3/3] Tách từ ({total} dòng, batch_size={BATCH_SIZE})...")
    errors = 0
    skipped = 0
    poisoned = len(poison_rows)
    last_i = resume_from
    t0 = time.perf_counter()

    # Sentinel marker to separate rows within a batch.
    # VnCoreNLP may insert spaces, so we match with regex later.
    SENTINEL = "XSEPX"
    _SENTINEL_RE = re.compile(r"\s*X\s*S\s*E\s*P\s*X\s*")

    # Build list of (index, row) that need processing
    work_items = []
    for i, row in enumerate(rows):
        if i <= resume_from:
            continue
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        if i in poison_rows:
            skipped += 1
            continue
        sanitized = _sanitize_text(text)
        if not sanitized.strip():
            skipped += 1
            continue
        work_items.append((i, row, sanitized))

    # ── Helper: process a single row (fallback) ──
    def _process_single(idx, row, text):
        nonlocal errors
        protected, store = _protect(text)
        try:
            sentences = model.word_segment(protected)
            segmented = " ".join(sentences)
            segmented = _restore(segmented, store)
            segmented = re.sub(r"XPHX\s*\d+\s*XPHX", "<NUM>", segmented)
            segmented = _canonicalize_emoji_tokens(segmented)
            segmented = _canonicalize_placeholders(segmented)
            row["text"] = segmented
        except Exception as e:
            errors += 1
            if errors <= 20:
                print(f"  [ERR] Dòng {row.get('id', '?')}: {str(e)[:120]}")

    # ── Helper: process a batch via single word_segment call ──
    def _process_batch(batch):
        """batch = list of (index, row, sanitized_text)"""
        nonlocal errors

        # Pre-process: protect placeholders for each row
        prepared = []  # (idx, row, protected_text, store)
        for idx, row, text in batch:
            protected, store = _protect(text)
            prepared.append((idx, row, protected, store))

        # Join all protected texts with sentinel separator
        combined = f"\n{SENTINEL}\n".join(p[2] for p in prepared)

        try:
            sentences = model.word_segment(combined)
            combined_result = " ".join(sentences)

            # Split result back by sentinel
            parts = _SENTINEL_RE.split(combined_result)

            if len(parts) != len(prepared):
                # Sentinel got mangled — fallback to single-row processing
                for idx, row, text in batch:
                    _process_single(idx, row, text)
                return

            # Map each part back to its row
            for (idx, row, protected, store), segmented in zip(prepared, parts):
                segmented = segmented.strip()
                segmented = _restore(segmented, store)
                segmented = re.sub(r"XPHX\s*\d+\s*XPHX", "<NUM>", segmented)
                segmented = _canonicalize_emoji_tokens(segmented)
                segmented = _canonicalize_placeholders(segmented)
                row["text"] = segmented

        except Exception:
            # Batch failed — retry each row individually
            for idx, row, text in batch:
                _process_single(idx, row, text)

    # ── Main loop: process in batches ──
    total_work = len(work_items)
    try:
        for b_start in range(0, total_work, BATCH_SIZE):
            batch = work_items[b_start:b_start + BATCH_SIZE]
            batch_first_idx = batch[0][0]
            batch_last_idx = batch[-1][0]
            last_i = batch_last_idx

            # Write checkpoint BEFORE calling VnCoreNLP
            _write_current(batch_first_idx)
            _write_checkpoint(batch_first_idx - 1)

            _process_batch(batch)

            # Batch done — update checkpoint
            _clear_current()
            _write_checkpoint(batch_last_idx)

            # Progress
            done_count = b_start + len(batch)
            done_row = batch_last_idx + 1
            if done_count % 500 < BATCH_SIZE or done_count >= total_work:
                elapsed = time.perf_counter() - t0
                speed = done_count / elapsed if elapsed > 0 else 0
                remaining = total_work - done_count
                eta = remaining / speed if speed > 0 else 0
                print(f"  [{done_row}/{total}] {speed:.0f} rows/s | ETA {eta:.0f}s"
                      f" | err={errors} skip={skipped} poison={poisoned}")

            # Save warehouse to disk periodically
            if done_count % SAVE_INTERVAL < BATCH_SIZE:
                _save_warehouse(rows)
                elapsed = time.perf_counter() - t0
                print(f"  ★ Saved tại dòng {done_row}/{total} ({elapsed:.0f}s)")

    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] Ctrl+C!")
        _save_warehouse(rows)
        _write_checkpoint(last_i)
        _remove_current()
        print(f"  ★ Checkpoint saved tại dòng {last_i + 1}. Chạy lại để tiếp tục.")
        sys.exit(0)

    # Done — clean up tracking files
    _remove_current()
    elapsed = time.perf_counter() - t0
    print(f"       → Xong trong {elapsed:.1f}s (err={errors}, skip={skipped}, poison={poisoned})")

    print(f"Ghi đè {WAREHOUSE_PATH}...")
    _save_warehouse(rows)
    _remove_checkpoint()
    _remove_poison_log()
    print(f"[DONE] Hoàn tất! {total} dòng đã được xử lý.")


if __name__ == "__main__":
    main()
