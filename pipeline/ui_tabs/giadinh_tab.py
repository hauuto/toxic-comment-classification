import csv
import os
import re
import threading
from collections import Counter

import customtkinter as ctk
from tkinter import messagebox, ttk


def _pipeline_dir() -> str:
    # ui_tabs/ is a subfolder of pipeline/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warehouse_path() -> str:
    return os.path.join(_pipeline_dir(), "warehouse.csv")


_TOKEN_CLEAN_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


def _clean_token(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    token = _TOKEN_CLEAN_RE.sub("", token)
    return token


def _setup_giadinh_tab(app):
    tab = app.tab_giadinh
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(2, weight=1)

    header = ctk.CTkFrame(tab)
    header.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    header.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(
        header,
        text="Tổng hợp 'gia đình / gia_đình' trong warehouse",
        font=("Arial", 15, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 2))

    ctk.CTkLabel(
        header,
        text=(
            "Quét streaming trên warehouse.csv để xem 'gia đình/gia_đình' thường đi với từ nào (trước/sau). "
            "Có Find & Replace toàn bộ (có backup)."
        ),
        text_color="gray",
    ).grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 10))

    app.gd_file_label = ctk.CTkLabel(header, text=f"File: {os.path.basename(_warehouse_path())}", text_color="gray")
    app.gd_file_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))

    app.gd_scan_btn = ctk.CTkButton(header, text="🔎 Quét thống kê", width=130, command=app.gd_scan)
    app.gd_scan_btn.grid(row=2, column=2, sticky="e", padx=10, pady=(0, 10))

    stats = ctk.CTkFrame(tab)
    stats.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
    stats.grid_columnconfigure(0, weight=1)

    app.gd_stats_label = ctk.CTkLabel(stats, text="Chưa quét", text_color="gray")
    app.gd_stats_label.grid(row=0, column=0, sticky="w", padx=10, pady=8)

    # Table
    tf = ctk.CTkFrame(tab)
    tf.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")
    tf.grid_columnconfigure(0, weight=1)
    tf.grid_rowconfigure(0, weight=1)

    app.gd_tree = ttk.Treeview(tf, columns=("left", "right", "count"), show="headings", selectmode="browse")
    app.gd_tree.heading("left", text="Từ trước")
    app.gd_tree.heading("right", text="Từ sau")
    app.gd_tree.heading("count", text="Số lần")

    app.gd_tree.column("left", width=250, anchor="w")
    app.gd_tree.column("right", width=250, anchor="w")
    app.gd_tree.column("count", width=90, anchor="center")

    gd_scroll = ttk.Scrollbar(tf, orient="vertical", command=app.gd_tree.yview)
    app.gd_tree.configure(yscrollcommand=gd_scroll.set)
    app.gd_tree.grid(row=0, column=0, sticky="nsew")
    gd_scroll.grid(row=0, column=1, sticky="ns")

    # Find & Replace
    fr = ctk.CTkFrame(tab)
    fr.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
    fr.grid_columnconfigure(1, weight=1)
    fr.grid_columnconfigure(3, weight=1)

    ctk.CTkLabel(fr, text="Find:").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
    app.gd_find_var = ctk.StringVar(value="gia đình")
    app.gd_find_entry = ctk.CTkEntry(fr, textvariable=app.gd_find_var, placeholder_text="VD: gia đình", width=260)
    app.gd_find_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 5))

    ctk.CTkLabel(fr, text="Replace:").grid(row=0, column=2, sticky="w", padx=10, pady=(10, 5))
    app.gd_replace_var = ctk.StringVar(value="")
    app.gd_replace_entry = ctk.CTkEntry(fr, textvariable=app.gd_replace_var, placeholder_text="Nhập text thay thế", width=260)
    app.gd_replace_entry.grid(row=0, column=3, sticky="ew", padx=(0, 10), pady=(10, 5))

    app.gd_apply_btn = ctk.CTkButton(fr, text="✏️ Replace toàn bộ", width=150, fg_color="#EA580C", hover_color="#C2410C", command=app.gd_apply_replace)
    app.gd_apply_btn.grid(row=0, column=4, sticky="e", padx=10, pady=(10, 5))

    app.gd_status_label = ctk.CTkLabel(tab, text="Trạng thái: Sẵn sàng", text_color="gray")
    app.gd_status_label.grid(row=4, column=0, sticky="w", padx=10, pady=(0, 10))

    app._gd_is_busy = False


def _gd_set_busy(app, busy: bool, msg: str | None = None, color: str = "gray"):
    app._gd_is_busy = busy
    try:
        app.gd_scan_btn.configure(state=("disabled" if busy else "normal"))
        app.gd_apply_btn.configure(state=("disabled" if busy else "normal"))
    except Exception:
        pass
    if msg is not None:
        app.gd_status_label.configure(text=f"Trạng thái: {msg}", text_color=color)


def _gd_display_contexts(app, contexts: list[tuple[str, str, int]]):
    app.gd_tree.delete(*app.gd_tree.get_children())
    for left, right, count in contexts:
        app.gd_tree.insert("", "end", values=(left, right, count))


def gd_scan(app):
    if getattr(app, "_gd_is_busy", False):
        return

    path = _warehouse_path()
    if not os.path.exists(path):
        messagebox.showerror("Lỗi", f"Không tìm thấy file: {path}")
        return

    def _worker():
        _gd_set_busy(app, True, "Đang quét...", "blue")

        total_rows = 0
        occ_total = 0
        occ_phrase = 0
        occ_underscore = 0
        contexts = Counter()  # (left, right) -> count

        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_rows += 1
                    text = (row.get("text") or "")
                    if "gia" not in text.lower():
                        continue

                    raw_tokens = text.split()
                    toks = [_clean_token(t).lower() for t in raw_tokens]

                    i = 0
                    while i < len(toks):
                        t = toks[i]
                        if not t:
                            i += 1
                            continue

                        if t == "gia_đình":
                            occ_total += 1
                            occ_underscore += 1
                            left = toks[i - 1] if i - 1 >= 0 else ""
                            right = toks[i + 1] if i + 1 < len(toks) else ""
                            contexts[(left, right)] += 1
                            i += 1
                            continue

                        if t == "gia" and i + 1 < len(toks) and toks[i + 1] == "đình":
                            occ_total += 1
                            occ_phrase += 1
                            left = toks[i - 1] if i - 1 >= 0 else ""
                            right = toks[i + 2] if i + 2 < len(toks) else ""
                            contexts[(left, right)] += 1
                            i += 2
                            continue

                        i += 1

        except Exception as e:
            app.after(0, lambda: (
                _gd_set_busy(app, False, "Lỗi khi quét", "red"),
                messagebox.showerror("Lỗi", f"Không quét được: {e}"),
            ))
            return

        top = contexts.most_common(200)
        top_rows = [(l or "(đầu câu)", r or "(cuối câu)", c) for (l, r), c in top]

        def _done():
            _gd_display_contexts(app, top_rows)
            app.gd_stats_label.configure(
                text=(
                    f"Đã quét {total_rows:,} dòng | Tổng occurrence: {occ_total:,} "
                    f"(gia đình: {occ_phrase:,} | gia_đình: {occ_underscore:,}) | Hiển thị top {len(top_rows)}"
                ),
                text_color="green",
            )
            _gd_set_busy(app, False, "Quét xong", "green")

        app.after(0, _done)

    threading.Thread(target=_worker, daemon=True).start()


def gd_apply_replace(app):
    if getattr(app, "_gd_is_busy", False):
        return

    find_text = (app.gd_find_var.get() or "").strip()
    replace_text = app.gd_replace_var.get() or ""

    if not find_text:
        messagebox.showwarning("Nhắc nhở", "Hãy nhập Find.")
        return

    path = _warehouse_path()
    if not os.path.exists(path):
        messagebox.showerror("Lỗi", f"Không tìm thấy file: {path}")
        return

    if not messagebox.askyesno(
        "Xác nhận",
        f"Replace toàn bộ trong warehouse.csv?\n\nFind: {find_text}\nReplace: {replace_text}\n\nHệ thống sẽ tạo backup trước khi ghi đè.",
    ):
        return

    def _worker():
        _gd_set_busy(app, True, "Đang replace (streaming)...", "blue")

        tmp_path = f"{path}.tmp"
        rows = 0
        changed_rows = 0
        replaced_count = 0

        try:
            bak_path = app._backup_file(path)

            with open(path, "r", encoding="utf-8-sig", newline="") as fin, open(
                tmp_path, "w", encoding="utf-8", newline=""
            ) as fout:
                reader = csv.DictReader(fin)
                fieldnames = list(reader.fieldnames or [])
                if not fieldnames:
                    raise RuntimeError("CSV không có header")

                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    rows += 1
                    text = row.get("text")
                    if isinstance(text, str) and find_text in text:
                        new_text = text.replace(find_text, replace_text)
                        if new_text != text:
                            changed_rows += 1
                            replaced_count += text.count(find_text)
                            row["text"] = new_text
                    writer.writerow(row)

            os.replace(tmp_path, path)

        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            app.after(0, lambda: (
                _gd_set_busy(app, False, "Lỗi khi replace", "red"),
                messagebox.showerror("Lỗi", f"Replace thất bại: {e}"),
            ))
            return

        def _done():
            _gd_set_busy(app, False, f"Replace xong | rows={rows:,} changed_rows={changed_rows:,} repl={replaced_count:,}", "green")
            messagebox.showinfo(
                "Hoàn tất",
                f"Đã replace xong.\n\nrows={rows:,}\nchanged_rows={changed_rows:,}\nreplacements={replaced_count:,}",
            )
            try:
                app.wh_load_data()
            except Exception:
                pass

        app.after(0, _done)

    threading.Thread(target=_worker, daemon=True).start()
