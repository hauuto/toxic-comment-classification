import csv
import functools
import os
import threading
import time

import customtkinter as ctk
import pandas as pd
from pandarallel import pandarallel
from tkinter import messagebox, ttk

from google_drive import download_warehouse, upload_warehouse
from nlp_pipeline import _canonicalize_emoji_tokens, _canonicalize_placeholders
from nlp_pipeline._parallel_worker import preprocess_no_segment
from nlp_pipeline.warehouse import overwrite_warehouse, read_warehouse


def _setup_warehouse_tab(app):
    tab = app.tab_warehouse
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(1, weight=1)

    # --- Header / toolbar ---
    header = ctk.CTkFrame(tab)
    header.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

    # Row 1: search + stats
    row1 = ctk.CTkFrame(header, fg_color="transparent")
    row1.pack(fill="x", padx=5, pady=(5, 2))

    ctk.CTkButton(row1, text="⟳ Tải dữ liệu", width=110, command=app.wh_load_data).pack(side="left", padx=5)

    ctk.CTkLabel(row1, text="Tìm kiếm:").pack(side="left", padx=(15, 5))
    app.wh_search_var = ctk.StringVar()
    app.wh_search_entry = ctk.CTkEntry(
        row1,
        textvariable=app.wh_search_var,
        placeholder_text="Nhập text để lọc...",
        width=250,
    )
    app.wh_search_entry.pack(side="left", padx=5)
    ctk.CTkButton(row1, text="Lọc", width=60, command=app.wh_filter_data).pack(side="left", padx=5)
    ctk.CTkButton(row1, text="Xóa lọc", width=70, command=app.wh_clear_filter).pack(side="left", padx=5)

    app.wh_stats_label = ctk.CTkLabel(row1, text="Warehouse: 0 dòng", text_color="gray")
    app.wh_stats_label.pack(side="right", padx=10)

    # Row 2: preprocessing options + actions
    row2 = ctk.CTkFrame(header, fg_color="transparent")
    row2.pack(fill="x", padx=5, pady=(2, 5))

    ctk.CTkLabel(row2, text="Tiền xử lý:").pack(side="left", padx=(5, 5))

    app.wh_dec_var = ctk.BooleanVar(value=True)
    app.wh_fil_var = ctk.BooleanVar(value=True)
    app.wh_nor_var = ctk.BooleanVar(value=True)
    app.wh_seg_backend_var = ctk.StringVar(value="Tắt")

    ctk.CTkCheckBox(row2, text="Decoder", variable=app.wh_dec_var).pack(side="left", padx=4)
    ctk.CTkCheckBox(row2, text="Filter", variable=app.wh_fil_var).pack(side="left", padx=4)
    ctk.CTkCheckBox(row2, text="Normalizer", variable=app.wh_nor_var).pack(side="left", padx=4)
    ctk.CTkLabel(row2, text="Tách từ:").pack(side="left", padx=(10, 5))
    app.wh_seg_backend_menu = ctk.CTkOptionMenu(
        row2,
        values=["Tắt", "VnCoreNLP", "Underthesea"],
        variable=app.wh_seg_backend_var,
        width=130,
    )
    app.wh_seg_backend_menu.pack(side="left", padx=4)

    ctk.CTkButton(
        row2,
        text="▶ Chạy Preprocessing",
        width=150,
        fg_color="#2563EB",
        hover_color="#1D4ED8",
        command=app.wh_run_preprocessing,
    ).pack(side="left", padx=(15, 5))
    ctk.CTkButton(
        row2,
        text="Xóa trùng lặp",
        width=110,
        fg_color="#7C3AED",
        hover_color="#6D28D9",
        command=app.wh_remove_duplicates,
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        row2,
        text="Xuất CSV",
        width=90,
        fg_color="#059669",
        hover_color="#047857",
        command=app.wh_export_csv,
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        row2,
        text="Xóa dòng chọn",
        width=100,
        fg_color="red",
        hover_color="darkred",
        command=app.wh_delete_selected,
    ).pack(side="right", padx=5)

    # Row 3: Google Drive sync
    row3 = ctk.CTkFrame(header, fg_color="transparent")
    row3.pack(fill="x", padx=5, pady=(2, 5))

    ctk.CTkLabel(row3, text="☁ Google Drive:", font=("Arial", 13, "bold")).pack(side="left", padx=(5, 10))
    ctk.CTkButton(
        row3,
        text="⬆ Upload lên Drive",
        width=150,
        fg_color="#EA580C",
        hover_color="#C2410C",
        command=app.wh_upload_drive,
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        row3,
        text="⬇ Tải từ Drive",
        width=150,
        fg_color="#0284C7",
        hover_color="#0369A1",
        command=app.wh_download_drive,
    ).pack(side="left", padx=5)

    # --- Data table ---
    tf = ctk.CTkFrame(tab)
    tf.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
    tf.grid_columnconfigure(0, weight=1)
    tf.grid_rowconfigure(0, weight=1)

    app.wh_tree = ttk.Treeview(tf, columns=("id", "text"), show="headings", selectmode="extended")
    app.wh_tree.heading("id", text="ID")
    app.wh_tree.heading("text", text="Nội dung bình luận")
    app.wh_tree.column("id", width=60, anchor="center")
    app.wh_tree.column("text", width=800, anchor="w")

    wh_scroll = ttk.Scrollbar(tf, orient="vertical", command=app.wh_tree.yview)
    app.wh_tree.configure(yscrollcommand=wh_scroll.set)
    app.wh_tree.grid(row=0, column=0, sticky="nsew")
    wh_scroll.grid(row=0, column=1, sticky="ns")

    # Status bar
    app.wh_status_label = ctk.CTkLabel(tab, text="Trạng thái: Sẵn sàng", text_color="gray")
    app.wh_status_label.grid(row=2, column=0, pady=5, sticky="w", padx=10)

    # Internal data cache
    app._wh_all_rows = []  # list of {"id": int, "text": str}
    app._wh_is_processing = False

    # Initial load
    app.wh_load_data()


def wh_load_data(app):
    """Load warehouse.csv into the table."""
    app._wh_all_rows = read_warehouse()
    _wh_display_rows(app, app._wh_all_rows)
    app.wh_stats_label.configure(text=f"Warehouse: {len(app._wh_all_rows)} dòng")
    app.wh_status_label.configure(
        text=f"Đã tải {len(app._wh_all_rows)} dòng từ warehouse.csv",
        text_color="green",
    )
    try:
        app._lbl_refresh_clusters()
    except Exception:
        pass


def _wh_display_rows(app, rows):
    """Populate the treeview with a list of row dicts."""
    app.wh_tree.delete(*app.wh_tree.get_children())
    for row in rows:
        display_text = str(row.get("text", "")).replace("\n", "  ")
        app.wh_tree.insert("", "end", values=(row.get("id", ""), display_text))


def wh_filter_data(app):
    """Filter displayed rows by search term."""
    query = app.wh_search_var.get().strip().lower()
    if not query:
        _wh_display_rows(app, app._wh_all_rows)
        return
    filtered = [r for r in app._wh_all_rows if query in r.get("text", "").lower()]
    _wh_display_rows(app, filtered)
    app.wh_status_label.configure(
        text=f"Hiển thị {len(filtered)}/{len(app._wh_all_rows)} dòng (lọc: '{query}')",
        text_color="blue",
    )


def wh_clear_filter(app):
    """Clear search filter and show all data."""
    app.wh_search_var.set("")
    _wh_display_rows(app, app._wh_all_rows)
    app.wh_status_label.configure(
        text=f"Hiển thị tất cả {len(app._wh_all_rows)} dòng",
        text_color="green",
    )


def wh_delete_selected(app):
    """Delete selected rows from warehouse."""
    selected = app.wh_tree.selection()
    if not selected:
        messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 dòng để xóa.")
        return
    ids_to_delete = set()
    for item in selected:
        vals = app.wh_tree.item(item)["values"]
        if vals:
            ids_to_delete.add(int(vals[0]))
    count_before = len(app._wh_all_rows)
    if not messagebox.askyesno("Xác nhận", f"Xóa {len(ids_to_delete)} dòng khỏi warehouse.csv?"):
        return
    app._wh_all_rows = [r for r in app._wh_all_rows if r["id"] not in ids_to_delete]
    # Re-assign IDs
    for i, row in enumerate(app._wh_all_rows):
        row["id"] = i + 1
    overwrite_warehouse(app._wh_all_rows)
    _wh_display_rows(app, app._wh_all_rows)
    removed = count_before - len(app._wh_all_rows)
    app.wh_stats_label.configure(text=f"Warehouse: {len(app._wh_all_rows)} dòng")
    app.wh_status_label.configure(text=f"Đã xóa {removed} dòng.", text_color="orange")
    try:
        app._lbl_refresh_clusters()
    except Exception:
        pass


def wh_remove_duplicates(app):
    """Remove duplicate texts from warehouse."""
    seen = set()
    unique = []
    for row in app._wh_all_rows:
        text = row.get("text", "").strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(row)
    removed = len(app._wh_all_rows) - len(unique)
    if removed == 0:
        messagebox.showinfo("Thông báo", "Không có dòng trùng lặp nào.")
        return
    # Re-assign IDs
    for i, row in enumerate(unique):
        row["id"] = i + 1
    app._wh_all_rows = unique
    overwrite_warehouse(app._wh_all_rows)
    _wh_display_rows(app, app._wh_all_rows)
    app.wh_stats_label.configure(text=f"Warehouse: {len(app._wh_all_rows)} dòng")
    app.wh_status_label.configure(text=f"Đã xóa {removed} dòng trùng lặp.", text_color="green")
    try:
        app._lbl_refresh_clusters()
    except Exception:
        pass


def wh_export_csv(app):
    """Export current warehouse data to a timestamped CSV."""
    if not app._wh_all_rows:
        messagebox.showwarning("Nhắc nhở", "Warehouse trống, không có gì để xuất.")
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(os.getcwd(), f"warehouse_export_{timestamp}.csv")
    try:
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text"])
            writer.writeheader()
            writer.writerows(app._wh_all_rows)
        messagebox.showinfo(
            "Thành công",
            f"Đã xuất {len(app._wh_all_rows)} dòng ra:\n{os.path.basename(export_path)}",
        )
        app.refresh_file_list()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xuất CSV: {e}")


def wh_run_preprocessing(app):
    """Run NLP preprocessing on all warehouse rows in a background thread."""
    if app._wh_is_processing:
        messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
        return
    if not app._wh_all_rows:
        messagebox.showwarning("Nhắc nhở", "Warehouse trống.")
        return

    u_dec = app.wh_dec_var.get()
    u_fil = app.wh_fil_var.get()
    u_nor = app.wh_nor_var.get()
    seg_choice = (app.wh_seg_backend_var.get() or "").strip()
    if seg_choice == "Underthesea":
        segmentor_backend = "underthesea"
        use_segmentor = True
    elif seg_choice == "VnCoreNLP":
        segmentor_backend = "vncorenlp"
        use_segmentor = True
    else:
        segmentor_backend = "whitespace"
        use_segmentor = False

    if not (u_dec or u_fil or u_nor or use_segmentor):
        messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 bước tiền xử lý.")
        return

    app._wh_is_processing = True
    app.wh_status_label.configure(text="Đang khởi tạo NLP pipeline... Vui lòng đợi.", text_color="orange")

    def _process():
        try:
            preprocessor = app._get_preprocessor(segmentor_backend)
            total = len(app._wh_all_rows)

            # ── Phase 1: Decode → Filter → Normalize (parallel via pandarallel) ──
            app.after(
                0,
                lambda: app.wh_status_label.configure(
                    text=f"Phase 1/2: Tiền xử lý song song ({total} dòng)...",
                    text_color="orange",
                ),
            )

            nb_workers = min(max((os.cpu_count() or 4) - 2, 1), 4)
            pandarallel.initialize(nb_workers=nb_workers, progress_bar=False, verbose=0)

            df = pd.DataFrame(app._wh_all_rows)

            _row_fn = functools.partial(
                preprocess_no_segment,
                use_decoder=u_dec,
                use_filter=u_fil,
                use_normalizer=u_nor,
            )
            df["cleaned"] = df["text"].parallel_apply(_row_fn)

            # Drop invalid rows
            df = df.dropna(subset=["cleaned"]).reset_index(drop=True)
            phase1_kept = len(df)
            removed_count = total - phase1_kept

            app.after(
                0,
                lambda: app.wh_status_label.configure(
                    text=f"Phase 1 xong: giữ {phase1_kept}/{total} dòng. "
                         + ("Đang tách từ..." if use_segmentor else "Hoàn tất."),
                    text_color="orange" if use_segmentor else "green",
                ),
            )

            # ── Phase 2: Segmentation (serial – VnCoreNLP JVM is single-threaded) ──
            if use_segmentor and phase1_kept > 0:
                cleaned_list = df["cleaned"].tolist()
                for idx in range(len(cleaned_list)):
                    try:
                        segmented = preprocessor.segmentor.segment(cleaned_list[idx])
                        if segmented:
                            segmented = _canonicalize_emoji_tokens(segmented)
                            segmented = _canonicalize_placeholders(segmented)
                            cleaned_list[idx] = segmented
                    except Exception:
                        pass  # keep unsegmented text on failure

                    if (idx + 1) % 500 == 0 or idx + 1 == len(cleaned_list):
                        app.after(
                            0,
                            lambda i=idx + 1: app.wh_status_label.configure(
                                text=f"Phase 2/2: Tách từ... {i}/{phase1_kept}",
                                text_color="orange",
                            ),
                        )
                df["cleaned"] = cleaned_list

            # ── Build final result ──
            kept = [
                {"id": i + 1, "text": row["cleaned"]}
                for i, row in df.iterrows()
            ]

            app._wh_all_rows = kept
            overwrite_warehouse(kept)
            app.after(0, _wh_display_rows, app, kept)
            app.after(0, lambda: app.wh_stats_label.configure(text=f"Warehouse: {len(kept)} dòng"))
            app.after(
                0,
                lambda: app.wh_status_label.configure(
                    text=f"Hoàn tất! Giữ {len(kept)}/{total} dòng (loại {removed_count} dòng).",
                    text_color="green",
                ),
            )
            app.after(0, lambda: app._lbl_refresh_clusters())
        except Exception as e:
            err_msg = str(e)
            app.after(0, lambda: app.wh_status_label.configure(text=f"Lỗi: {err_msg}", text_color="red"))
        finally:
            app._wh_is_processing = False

    threading.Thread(target=_process, daemon=True).start()


def wh_upload_drive(app):
    """Upload warehouse.csv to Google Drive in background thread."""
    if app._wh_is_processing:
        messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
        return
    if not app._wh_all_rows:
        messagebox.showwarning("Nhắc nhở", "Warehouse trống, không có gì để upload.")
        return

    app._wh_is_processing = True
    app.wh_status_label.configure(text="⬆ Đang upload warehouse.csv lên Google Drive...", text_color="orange")

    def _upload():
        try:
            def _log(msg):
                app.after(0, lambda: app.wh_status_label.configure(text=msg, text_color="orange"))

            upload_warehouse(log_callback=_log)
            app.after(
                0,
                lambda: app.wh_status_label.configure(
                    text=f"✓ Đã upload warehouse.csv lên Google Drive ({len(app._wh_all_rows)} dòng).",
                    text_color="green",
                ),
            )
        except FileNotFoundError as e:
            app.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            app.after(0, lambda: app.wh_status_label.configure(text="✗ Upload thất bại: thiếu credentials.", text_color="red"))
        except Exception as e:
            app.after(0, lambda: messagebox.showerror("Lỗi Upload", f"Không thể upload lên Drive:\n{str(e)}"))
            app.after(0, lambda: app.wh_status_label.configure(text=f"✗ Upload thất bại: {str(e)[:80]}", text_color="red"))
        finally:
            app._wh_is_processing = False

    threading.Thread(target=_upload, daemon=True).start()


def wh_download_drive(app):
    """Download warehouse.csv from Google Drive in background thread."""
    if app._wh_is_processing:
        messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
        return

    if app._wh_all_rows:
        if not messagebox.askyesno(
            "Xác nhận",
            f"Warehouse hiện có {len(app._wh_all_rows)} dòng.\n"
            "Tải từ Drive sẽ GHI ĐÈ toàn bộ dữ liệu local.\n\n"
            "Bạn có muốn tiếp tục?",
        ):
            return

    app._wh_is_processing = True
    app.wh_status_label.configure(text="⬇ Đang tải warehouse.csv từ Google Drive...", text_color="orange")

    def _download():
        try:
            def _log(msg):
                app.after(0, lambda: app.wh_status_label.configure(text=msg, text_color="orange"))

            success = download_warehouse(log_callback=_log)
            if success:
                app.after(0, app.wh_load_data)
                app.after(
                    0,
                    lambda: app.wh_status_label.configure(
                        text="✓ Đã tải warehouse.csv từ Google Drive và cập nhật.",
                        text_color="green",
                    ),
                )
            else:
                app.after(
                    0,
                    lambda: app.wh_status_label.configure(
                        text="✗ Không tìm thấy warehouse.csv trên Google Drive.",
                        text_color="red",
                    ),
                )
        except FileNotFoundError as e:
            app.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            app.after(
                0,
                lambda: app.wh_status_label.configure(text="✗ Download thất bại: thiếu credentials.", text_color="red"),
            )
        except Exception as e:
            app.after(0, lambda: messagebox.showerror("Lỗi Download", f"Không thể tải từ Drive:\n{str(e)}"))
            app.after(
                0,
                lambda: app.wh_status_label.configure(text=f"✗ Download thất bại: {str(e)[:80]}", text_color="red"),
            )
        finally:
            app._wh_is_processing = False

    threading.Thread(target=_download, daemon=True).start()
