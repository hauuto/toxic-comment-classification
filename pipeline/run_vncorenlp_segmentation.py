"""
Standalone script – Chạy VnCoreNLP word segmentation trên warehouse.csv.

Kiến trúc đơn giản: load VnCoreNLP trực tiếp trong process, xử lý tuần tự.
Checkpoint mỗi 5000 dòng → nếu crash, chạy lại để tiếp tục.

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
SAVE_INTERVAL = 500   # save warehouse to disk every N rows

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

    # 4. Process rows
    print(f"[3/3] Tách từ ({total} dòng)...")
    errors = 0
    skipped = 0
    poisoned = len(poison_rows)
    last_i = resume_from
    t0 = time.perf_counter()

    try:
        for i, row in enumerate(rows):
            if i <= resume_from:
                continue

            last_i = i
            text = row.get("text", "")
            if not text or not text.strip():
                continue

            # Skip poison rows
            if i in poison_rows:
                skipped += 1
                continue

            # Sanitize text before processing
            text = _sanitize_text(text)
            if not text.strip():
                skipped += 1
                continue

            # Protect placeholders
            protected, store = _protect(text)

            # Write "current" + checkpoint BEFORE calling VnCoreNLP — if JVM
            # crashes, we know exactly which row caused it AND don't lose progress
            _write_current(i)
            _write_checkpoint(i - 1)  # mark previous row as last successfully done

            try:
                result = model.annotate_text(protected)
                tokens = []
                for sentence in result.values():
                    for word_info in sentence:
                        tokens.append(word_info["wordForm"])
                segmented = " ".join(tokens)

                # Restore placeholders and canonicalize
                segmented = _restore(segmented, store)
                segmented = re.sub(r"XPHX\s*\d+\s*XPHX", "<NUM>", segmented)
                segmented = _canonicalize_emoji_tokens(segmented)
                segmented = _canonicalize_placeholders(segmented)
                row["text"] = segmented

            except Exception as e:
                errors += 1
                if errors <= 20:
                    err = str(e)
                    print(f"  [ERR] Dòng {row.get('id', '?')}: {err[:120]}")
                # keep original text on any error

            # Clear current marker after successful processing
            _clear_current()
            _write_checkpoint(i)

            # Progress
            done = i + 1
            if done % 500 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                processed = done - (resume_from + 1)
                speed = processed / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta = remaining / speed if speed > 0 else 0
                print(f"  [{done}/{total}] {speed:.0f} rows/s | ETA {eta:.0f}s"
                      f" | err={errors} skip={skipped} poison={poisoned}")

            # Save warehouse to disk periodically
            if (i + 1) % SAVE_INTERVAL == 0:
                _save_warehouse(rows)
                elapsed = time.perf_counter() - t0
                print(f"  ★ Saved tại dòng {i + 1}/{total} ({elapsed:.0f}s)")

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
