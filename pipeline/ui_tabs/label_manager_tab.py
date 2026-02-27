import csv
import os
import threading
import time

import customtkinter as ctk
from tkinter import messagebox, ttk

from google_drive import download_labeled_data, upload_labeled_data
from lmstudio_classifier import (
    TIER1_LABELS,
    TIER2_LABELS,
    TIER3_ALL_LABELS,
    TIER3_CLEAN_LABELS,
    TIER3_TOXIC_LABELS,
)


def _setup_label_manager_tab(app):
    tab = app.tab_label_mgr
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(1, weight=1)

    header = ctk.CTkFrame(tab)
    header.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

    row1 = ctk.CTkFrame(header, fg_color="transparent")
    row1.pack(fill="x", padx=5, pady=(5, 2))
    ctk.CTkButton(row1, text="⟳ Tải dữ liệu", width=110, command=app._lm_load_data).pack(side="left", padx=5)
    ctk.CTkLabel(row1, text="Tìm kiếm:").pack(side="left", padx=(15, 5))
    app.lm_search_var = ctk.StringVar()
    ctk.CTkEntry(
        row1,
        textvariable=app.lm_search_var,
        placeholder_text="Nhập text để lọc...",
        width=180,
    ).pack(side="left", padx=5)
    ctk.CTkButton(row1, text="Lọc", width=60, command=app._lm_filter_data).pack(side="left", padx=5)

    ctk.CTkLabel(row1, text="Tier1:").pack(side="left", padx=(10, 3))
    app.lm_t1_filter_var = ctk.StringVar(value="Tất cả")
    ctk.CTkOptionMenu(
        row1,
        values=["Tất cả"] + TIER1_LABELS,
        variable=app.lm_t1_filter_var,
        command=lambda _: app._lm_filter_data(),
        width=100,
    ).pack(side="left", padx=2)

    ctk.CTkLabel(row1, text="Tier2:").pack(side="left", padx=(8, 3))
    app.lm_t2_filter_var = ctk.StringVar(value="Tất cả")
    ctk.CTkOptionMenu(
        row1,
        values=["Tất cả"] + TIER2_LABELS,
        variable=app.lm_t2_filter_var,
        command=lambda _: app._lm_filter_data(),
        width=100,
    ).pack(side="left", padx=2)

    ctk.CTkLabel(row1, text="Tier3:").pack(side="left", padx=(8, 3))
    app.lm_t3_filter_var = ctk.StringVar(value="Tất cả")
    ctk.CTkOptionMenu(
        row1,
        values=["Tất cả"] + TIER3_ALL_LABELS,
        variable=app.lm_t3_filter_var,
        command=lambda _: app._lm_filter_data(),
        width=120,
    ).pack(side="left", padx=2)

    ctk.CTkButton(row1, text="Xóa lọc", width=70, command=app._lm_clear_filter).pack(side="left", padx=5)
    app.lm_stats_label = ctk.CTkLabel(row1, text="Labeled: 0 dòng", text_color="gray")
    app.lm_stats_label.pack(side="right", padx=10)

    row2 = ctk.CTkFrame(header, fg_color="transparent")
    row2.pack(fill="x", padx=5, pady=(2, 5))
    ctk.CTkButton(
        row2,
        text="Xuất CSV",
        width=90,
        fg_color="#059669",
        hover_color="#047857",
        command=app._lm_export_csv,
    ).pack(side="left", padx=5)

    # --- Edit labels section ---
    ctk.CTkLabel(row2, text="Sửa →").pack(side="left", padx=(20, 3))
    ctk.CTkLabel(row2, text="T1:").pack(side="left", padx=(0, 2))
    app.lm_edit_t1_var = ctk.StringVar(value="—")
    ctk.CTkOptionMenu(row2, values=["—"] + TIER1_LABELS, variable=app.lm_edit_t1_var, width=95).pack(side="left", padx=2)
    ctk.CTkLabel(row2, text="T2:").pack(side="left", padx=(5, 2))
    app.lm_edit_t2_var = ctk.StringVar(value="—")
    ctk.CTkOptionMenu(row2, values=["—"] + TIER2_LABELS, variable=app.lm_edit_t2_var, width=85).pack(side="left", padx=2)
    ctk.CTkLabel(row2, text="T3:").pack(side="left", padx=(5, 2))
    app.lm_edit_t3_var = ctk.StringVar(value="—")
    ctk.CTkOptionMenu(row2, values=["—"] + TIER3_ALL_LABELS, variable=app.lm_edit_t3_var, width=120).pack(side="left", padx=2)
    ctk.CTkButton(
        row2,
        text="Áp dụng",
        width=80,
        fg_color="#7C3AED",
        hover_color="#6D28D9",
        command=app._lm_edit_label,
    ).pack(side="left", padx=5)

    # --- Drive sync + delete ---
    ctk.CTkButton(
        row2,
        text="Xóa dòng chọn",
        width=110,
        fg_color="red",
        hover_color="darkred",
        command=app._lm_delete_selected,
    ).pack(side="right", padx=5)
    ctk.CTkButton(
        row2,
        text="⬇ Drive",
        width=90,
        fg_color="#0284C7",
        hover_color="#0369A1",
        command=app._lm_download_drive,
    ).pack(side="right", padx=3)
    ctk.CTkButton(
        row2,
        text="⬆ Drive",
        width=90,
        fg_color="#0284C7",
        hover_color="#0369A1",
        command=app._lm_upload_drive,
    ).pack(side="right", padx=3)

    main_frame = ctk.CTkFrame(tab)
    main_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
    main_frame.grid_columnconfigure(0, weight=3)
    main_frame.grid_columnconfigure(1, weight=2)
    main_frame.grid_rowconfigure(0, weight=1)

    tf = ctk.CTkFrame(main_frame)
    tf.grid(row=0, column=0, padx=(5, 3), pady=5, sticky="nsew")
    tf.grid_columnconfigure(0, weight=1)
    tf.grid_rowconfigure(0, weight=1)
    app.lm_tree = ttk.Treeview(
        tf,
        columns=("id", "text", "tier1", "tier2", "tier3"),
        show="headings",
        selectmode="extended",
    )
    app.lm_tree.heading("id", text="ID")
    app.lm_tree.heading("text", text="Nội dung bình luận")
    app.lm_tree.heading("tier1", text="Tier1 Spam")
    app.lm_tree.heading("tier2", text="Tier2 Toxic")
    app.lm_tree.heading("tier3", text="Tier3 Labels")
    app.lm_tree.column("id", width=40, anchor="center")
    app.lm_tree.column("text", width=300, anchor="w")
    app.lm_tree.column("tier1", width=80, anchor="center")
    app.lm_tree.column("tier2", width=80, anchor="center")
    app.lm_tree.column("tier3", width=130, anchor="center")
    lm_scroll = ttk.Scrollbar(tf, orient="vertical", command=app.lm_tree.yview)
    app.lm_tree.configure(yscrollcommand=lm_scroll.set)
    app.lm_tree.grid(row=0, column=0, sticky="nsew")
    lm_scroll.grid(row=0, column=1, sticky="ns")

    chart_frame = ctk.CTkFrame(main_frame)
    chart_frame.grid(row=0, column=1, padx=(3, 5), pady=5, sticky="nsew")
    chart_frame.grid_columnconfigure(0, weight=1)
    chart_frame.grid_rowconfigure(0, weight=1)
    app.lm_chart_frame = chart_frame
    app._lm_canvas = None

    app.lm_status_label = ctk.CTkLabel(tab, text="Trạng thái: Sẵn sàng", text_color="gray")
    app.lm_status_label.grid(row=2, column=0, pady=5, sticky="w", padx=10)
    app._lm_all_rows = []
    app._lm_is_processing = False
    app._lm_load_data()


def _migrate_old_label(old_label: str):
    """Migrate old single-label format to 3-tier. Returns (tier1, tier2, tier3_str)."""
    old_label = old_label.strip()
    if old_label == "Spam":
        return "Spam", "Clean", "Neutral"
    elif old_label in ("Hate Speech", "Harassment", "Obscene"):
        return "Not Spam", "Toxic", old_label
    elif old_label == "Clean":
        return "Not Spam", "Clean", "Neutral"
    else:
        return "Not Spam", "Clean", "Neutral"


def _lm_load_data(app):
    labeled_path = app._get_labeled_data_path()
    app._lm_all_rows = []
    if os.path.isfile(labeled_path):
        try:
            raw_rows, fieldnames, enc_used = app._read_csv_dicts_with_fallback(labeled_path)
            # Auto-convert only when we are reasonably confident (cp1258/cp1252).
            # Avoid destructive conversions from latin-1 or utf-8(replace).
            if enc_used in ("cp1258", "cp1252"):
                bak = app._backup_file(labeled_path)
                app._rewrite_csv_utf8sig(labeled_path, raw_rows, fieldnames)
                app.lm_status_label.configure(text=f"Đã sửa encoding (backup: {os.path.basename(bak)})", text_color="orange")
                raw_rows, fieldnames, enc_used = app._read_csv_dicts_with_fallback(labeled_path)

            # If file content looks like mojibake (double-encoded), repair once.
            need_repair = False
            for r in raw_rows[:200]:
                if not isinstance(r, dict):
                    continue
                for v in r.values():
                    if isinstance(v, str) and app._looks_mojibake(v):
                        need_repair = True
                        break
                if need_repair:
                    break

            if need_repair:
                bak = app._backup_file(labeled_path)
                app._rewrite_csv_utf8sig(labeled_path, raw_rows, fieldnames)
                app.lm_status_label.configure(text=f"Đã repair chữ Việt (backup: {os.path.basename(bak)})", text_color="orange")
                raw_rows, fieldnames, enc_used = app._read_csv_dicts_with_fallback(labeled_path)

            for row in raw_rows:
                if isinstance(row, dict):
                    row = app._sanitize_csv_row(row)
                # New 3-tier format
                if "tier1_spam" in row:
                    app._lm_all_rows.append(
                        {
                            "id": row.get("id", ""),
                            "text": row.get("text", ""),
                            "tier1_spam": row.get("tier1_spam", "Not Spam"),
                            "tier2_toxic": row.get("tier2_toxic", "Clean"),
                            "tier3_labels": row.get("tier3_labels", ""),
                        }
                    )
                else:
                    # Backward compatibility: migrate old format (id, text, label)
                    old_label = row.get("label", "Clean")
                    t1, t2, t3 = app._migrate_old_label(str(old_label))
                    app._lm_all_rows.append(
                        {
                            "id": row.get("id", ""),
                            "text": row.get("text", ""),
                            "tier1_spam": t1,
                            "tier2_toxic": t2,
                            "tier3_labels": t3,
                        }
                    )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc labeled_data.csv: {e}")

    _lm_display_rows(app, app._lm_all_rows)
    _lm_update_chart(app)
    app.lm_stats_label.configure(text=f"Labeled: {len(app._lm_all_rows)} dòng")
    if app._lm_all_rows:
        app.lm_status_label.configure(text=f"Đã tải {len(app._lm_all_rows)} dòng từ labeled_data.csv", text_color="green")
    else:
        app.lm_status_label.configure(text="Chưa có dữ liệu labeled_data.csv", text_color="gray")


def _lm_display_rows(app, rows):
    app.lm_tree.delete(*app.lm_tree.get_children())
    for row in rows:
        app.lm_tree.insert(
            "",
            "end",
            values=(
                row.get("id", ""),
                str(row.get("text", "")).replace("\n", "  "),
                row.get("tier1_spam", ""),
                row.get("tier2_toxic", ""),
                row.get("tier3_labels", ""),
            ),
        )


def _lm_filter_data(app):
    query = app.lm_search_var.get().strip().lower()
    t1_filter = app.lm_t1_filter_var.get()
    t2_filter = app.lm_t2_filter_var.get()
    t3_filter = app.lm_t3_filter_var.get()
    filtered = app._lm_all_rows
    if query:
        filtered = [r for r in filtered if query in r.get("text", "").lower()]
    if t1_filter != "Tất cả":
        filtered = [r for r in filtered if r.get("tier1_spam", "") == t1_filter]
    if t2_filter != "Tất cả":
        filtered = [r for r in filtered if r.get("tier2_toxic", "") == t2_filter]
    if t3_filter != "Tất cả":
        filtered = [r for r in filtered if t3_filter in r.get("tier3_labels", "").split("|")]
    _lm_display_rows(app, filtered)
    app.lm_status_label.configure(text=f"Hiển thị {len(filtered)}/{len(app._lm_all_rows)} dòng", text_color="blue")


def _lm_clear_filter(app):
    app.lm_search_var.set("")
    app.lm_t1_filter_var.set("Tất cả")
    app.lm_t2_filter_var.set("Tất cả")
    app.lm_t3_filter_var.set("Tất cả")
    _lm_display_rows(app, app._lm_all_rows)
    app.lm_status_label.configure(text=f"Hiển thị tất cả {len(app._lm_all_rows)} dòng", text_color="green")


def _lm_save_data(app):
    labeled_path = app._get_labeled_data_path()
    try:
        with open(labeled_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "tier1_spam", "tier2_toxic", "tier3_labels"])
            writer.writeheader()
            writer.writerows(app._lm_all_rows)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể lưu file: {e}")


def _lm_delete_selected(app):
    selected = app.lm_tree.selection()
    if not selected:
        messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 dòng để xóa.")
        return
    ids_to_delete = set()
    for item in selected:
        vals = app.lm_tree.item(item)["values"]
        if vals:
            ids_to_delete.add(str(vals[0]))
    if not messagebox.askyesno("Xác nhận", f"Xóa {len(ids_to_delete)} dòng?"):
        return
    app._lm_all_rows = [r for r in app._lm_all_rows if str(r["id"]) not in ids_to_delete]
    _lm_save_data(app)
    _lm_display_rows(app, app._lm_all_rows)
    _lm_update_chart(app)
    app.lm_stats_label.configure(text=f"Labeled: {len(app._lm_all_rows)} dòng")
    app.lm_status_label.configure(text=f"Đã xóa {len(ids_to_delete)} dòng.", text_color="orange")


def _lm_edit_label(app):
    selected = app.lm_tree.selection()
    if not selected:
        messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 dòng để sửa nhãn.")
        return
    new_t1 = app.lm_edit_t1_var.get()
    new_t2 = app.lm_edit_t2_var.get()
    new_t3 = app.lm_edit_t3_var.get()
    if new_t1 == "—" and new_t2 == "—" and new_t3 == "—":
        messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 tier để sửa (T1, T2, hoặc T3).")
        return
    ids_to_edit = set()
    for item in selected:
        vals = app.lm_tree.item(item)["values"]
        if vals:
            ids_to_edit.add(str(vals[0]))
    changed = 0
    for row in app._lm_all_rows:
        if str(row["id"]) in ids_to_edit:
            if new_t1 != "—":
                row["tier1_spam"] = new_t1
            if new_t2 != "—":
                row["tier2_toxic"] = new_t2
                # If changing tier2, clear tier3 if incompatible
                current_t3 = row.get("tier3_labels", "").split("|") if row.get("tier3_labels") else []
                valid_pool = TIER3_TOXIC_LABELS if new_t2 == "Toxic" else TIER3_CLEAN_LABELS
                compatible = [lbl for lbl in current_t3 if lbl in valid_pool]
                row["tier3_labels"] = "|".join(compatible)
            if new_t3 != "—":
                # Append or replace tier3
                current_t3 = row.get("tier3_labels", "").split("|") if row.get("tier3_labels") else []
                current_t3 = [lbl for lbl in current_t3 if lbl]
                if new_t3 not in current_t3:
                    current_t3.append(new_t3)
                row["tier3_labels"] = "|".join(current_t3)
            changed += 1
    _lm_save_data(app)
    _lm_display_rows(app, app._lm_all_rows)
    _lm_update_chart(app)
    parts = []
    if new_t1 != "—":
        parts.append(f"T1→{new_t1}")
    if new_t2 != "—":
        parts.append(f"T2→{new_t2}")
    if new_t3 != "—":
        parts.append(f"T3+={new_t3}")
    app.lm_status_label.configure(text=f"Đã sửa {changed} dòng: {', '.join(parts)}", text_color="green")


def _lm_export_csv(app):
    if not app._lm_all_rows:
        messagebox.showwarning("Nhắc nhở", "Không có dữ liệu để xuất.")
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(os.getcwd(), f"labeled_export_{timestamp}.csv")
    try:
        with open(export_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "tier1_spam", "tier2_toxic", "tier3_labels"])
            writer.writeheader()
            writer.writerows(app._lm_all_rows)
        messagebox.showinfo("Thành công", f"Đã xuất {len(app._lm_all_rows)} dòng ra:\n{os.path.basename(export_path)}")
        app.refresh_file_list()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xuất CSV: {e}")


def _lm_update_chart(app):
    # Import matplotlib lazily to avoid import-order/backends issues.
    import sys
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    total = len(app._lm_all_rows)

    # Count tiers
    t1_counts = {"Spam": 0, "Not Spam": 0}
    t2_counts = {"Toxic": 0, "Clean": 0}
    t3_counts = {}
    for lbl in TIER3_ALL_LABELS:
        t3_counts[lbl] = 0

    for row in app._lm_all_rows:
        t1 = row.get("tier1_spam", "Not Spam")
        t2 = row.get("tier2_toxic", "Clean")
        t1_counts[t1] = t1_counts.get(t1, 0) + 1
        t2_counts[t2] = t2_counts.get(t2, 0) + 1
        t3_str = row.get("tier3_labels", "")
        if t3_str:
            for lbl in t3_str.split("|"):
                lbl = lbl.strip()
                if lbl:
                    t3_counts[lbl] = t3_counts.get(lbl, 0) + 1

    if app._lm_canvas:
        app._lm_canvas.get_tk_widget().destroy()
        app._lm_canvas = None

    bg_color = "#2B2B2B"
    fig = Figure(figsize=(5, 6), dpi=90, facecolor=bg_color)

    # --- Tier 1: Pie ---
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_facecolor(bg_color)
    t1_labels = list(t1_counts.keys())
    t1_values = list(t1_counts.values())
    t1_colors = ["#F59E0B", "#22C55E"]
    if sum(t1_values) > 0:
        wedges, texts, autotexts = ax1.pie(
            t1_values,
            labels=t1_labels,
            colors=t1_colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"color": "white", "fontsize": 8},
        )
        for at in autotexts:
            at.set_fontsize(7)
    ax1.set_title(
        f"Tier 1 – Spam Check (n={total})",
        fontsize=10,
        color="white",
        fontweight="bold",
        pad=5,
    )

    # --- Tier 2: Pie ---
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_facecolor(bg_color)
    t2_labels = list(t2_counts.keys())
    t2_values = list(t2_counts.values())
    t2_colors = ["#EF4444", "#22C55E"]
    if sum(t2_values) > 0:
        wedges2, texts2, autotexts2 = ax2.pie(
            t2_values,
            labels=t2_labels,
            colors=t2_colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"color": "white", "fontsize": 8},
        )
        for at in autotexts2:
            at.set_fontsize(7)
    ax2.set_title("Tier 2 – Toxic Check", fontsize=10, color="white", fontweight="bold", pad=5)

    # --- Tier 3: Horizontal bar ---
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_facecolor(bg_color)
    t3_color_map = {
        "Hate Speech": "#EF4444",
        "Harassment": "#8B5CF6",
        "Obscene": "#EC4899",
        "Positive": "#22C55E",
        "Negative": "#F59E0B",
        "Neutral": "#6B7280",
    }
    t3_labels = list(t3_counts.keys())
    t3_values = list(t3_counts.values())
    t3_colors = [t3_color_map.get(lbl, "#6B7280") for lbl in t3_labels]
    bars = ax3.barh(t3_labels, t3_values, color=t3_colors, edgecolor="#444", height=0.55)
    max_val = max(t3_values) if t3_values and max(t3_values) > 0 else 1
    for bar, val in zip(bars, t3_values):
        if val > 0:
            ax3.text(
                bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                ha="left",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
    ax3.set_title("Tier 3 – Multi-label", fontsize=10, color="white", fontweight="bold", pad=5)
    ax3.tick_params(colors="white", labelsize=7)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_color("#555")
    ax3.spines["left"].set_color("#555")
    if max_val > 0:
        ax3.set_xlim(0, max_val * 1.25)

    fig.tight_layout(pad=1.5)
    app._lm_canvas = FigureCanvasTkAgg(fig, master=app.lm_chart_frame)
    app._lm_canvas.draw()
    app._lm_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    plt.close(fig)


def _lm_upload_drive(app):
    """Upload labeled_data.csv to Google Drive in background thread."""
    if app._lm_is_processing:
        messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
        return
    if not app._lm_all_rows:
        messagebox.showwarning("Nhắc nhở", "Chưa có dữ liệu labeled để upload.")
        return

    app._lm_is_processing = True
    app.lm_status_label.configure(text="⬆ Đang upload labeled_data.csv lên Google Drive...", text_color="orange")

    def _upload():
        try:
            def _log(msg):
                app.after(0, lambda: app.lm_status_label.configure(text=msg, text_color="orange"))

            upload_labeled_data(log_callback=_log)
            app.after(
                0,
                lambda: app.lm_status_label.configure(
                    text=f"✓ Đã upload labeled_data.csv lên Google Drive ({len(app._lm_all_rows)} dòng).",
                    text_color="green",
                ),
            )
        except FileNotFoundError as e:
            app.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            app.after(0, lambda: app.lm_status_label.configure(text="✗ Upload thất bại: thiếu credentials.", text_color="red"))
        except Exception as e:
            app.after(0, lambda: messagebox.showerror("Lỗi Upload", f"Không thể upload lên Drive:\n{str(e)}"))
            app.after(0, lambda: app.lm_status_label.configure(text=f"✗ Upload thất bại: {str(e)[:80]}", text_color="red"))
        finally:
            app._lm_is_processing = False

    threading.Thread(target=_upload, daemon=True).start()


def _lm_download_drive(app):
    """Download labeled_data.csv from Google Drive in background thread."""
    if app._lm_is_processing:
        messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
        return

    if app._lm_all_rows:
        if not messagebox.askyesno(
            "Xác nhận",
            f"Labeled data hiện có {len(app._lm_all_rows)} dòng.\n"
            "Tải từ Drive sẽ GHI ĐÈ toàn bộ dữ liệu local.\n\n"
            "Bạn có muốn tiếp tục?",
        ):
            return

    app._lm_is_processing = True
    app.lm_status_label.configure(text="⬇ Đang tải labeled_data.csv từ Google Drive...", text_color="orange")

    def _download():
        try:
            def _log(msg):
                app.after(0, lambda: app.lm_status_label.configure(text=msg, text_color="orange"))

            success = download_labeled_data(log_callback=_log)
            if success:
                app.after(0, app._lm_load_data)
                app.after(
                    0,
                    lambda: app.lm_status_label.configure(
                        text="✓ Đã tải labeled_data.csv từ Google Drive và cập nhật.",
                        text_color="green",
                    ),
                )
            else:
                app.after(
                    0,
                    lambda: app.lm_status_label.configure(
                        text="✗ Không tìm thấy labeled_data.csv trên Google Drive.",
                        text_color="red",
                    ),
                )
        except FileNotFoundError as e:
            app.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            app.after(0, lambda: app.lm_status_label.configure(text="✗ Download thất bại: thiếu credentials.", text_color="red"))
        except Exception as e:
            app.after(0, lambda: messagebox.showerror("Lỗi Download", f"Không thể tải từ Drive:\n{str(e)}"))
            app.after(0, lambda: app.lm_status_label.configure(text=f"✗ Download thất bại: {str(e)[:80]}", text_color="red"))
        finally:
            app._lm_is_processing = False

    threading.Thread(target=_download, daemon=True).start()
