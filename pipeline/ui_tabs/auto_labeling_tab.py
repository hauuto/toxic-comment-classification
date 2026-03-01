import concurrent.futures
import os
import threading
from datetime import datetime, timezone

import customtkinter as ctk
from tkinter import messagebox, ttk

from gemini_hierarchical_classifier import GeminiHierarchicalClassifier, GeminiSafetyBlockError
from nlp_pipeline.warehouse import (
    CLUSTER_SIZE_DEFAULT,
    get_warehouse_clusters,
    get_warehouse_count,
    read_warehouse,
    read_warehouse_cluster,
)


def _pipeline_dir() -> str:
    # ui_tabs/ is a subfolder of pipeline/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_labeled_data_path(app) -> str:
    return os.path.join(_pipeline_dir(), "labeled_data.csv")


def _setup_labeling_tab(app):
    tab = app.tab_labeling
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(2, weight=1)

    conn_frame = ctk.CTkFrame(tab)
    conn_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
    conn_frame.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(conn_frame, text="Gemini Model:").grid(row=0, column=0, padx=10, pady=8, sticky="w")
    app.lbl_model_var = ctk.StringVar(value="")
    app.lbl_model_entry = ctk.CTkEntry(
        conn_frame,
        textvariable=app.lbl_model_var,
        width=300,
        placeholder_text="(mặc định: gemini-2.0-flash)",
    )
    app.lbl_model_entry.grid(row=0, column=1, padx=5, pady=8, sticky="w")

    app.lbl_test_btn = ctk.CTkButton(
        conn_frame,
        text="🔌 Test Connection",
        width=140,
        command=app._lbl_test_connection,
    )
    app.lbl_test_btn.grid(row=0, column=2, padx=10, pady=8)

    ctrl_frame = ctk.CTkFrame(tab)
    ctrl_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    ctk.CTkLabel(ctrl_frame, text="Batch size:").pack(side="left", padx=(10, 5))
    app.lbl_batch_var = ctk.StringVar(value="5")
    ctk.CTkEntry(ctrl_frame, textvariable=app.lbl_batch_var, width=60).pack(side="left", padx=(0, 15))

    ctk.CTkLabel(ctrl_frame, text="Workers:").pack(side="left", padx=(5, 5))
    app.lbl_workers_var = ctk.StringVar(value="4")
    ctk.CTkEntry(ctrl_frame, textvariable=app.lbl_workers_var, width=60).pack(side="left", padx=(0, 15))

    # Cluster selector (25k rows per cluster)
    ctk.CTkLabel(ctrl_frame, text="Cluster:").pack(side="left", padx=(5, 5))
    app.lbl_cluster_var = ctk.StringVar(value="")
    app.lbl_cluster_menu = ctk.CTkOptionMenu(
        ctrl_frame,
        values=["(đang tải...)"] ,
        variable=app.lbl_cluster_var,
        command=app._lbl_on_cluster_change,
        width=210,
    )
    app.lbl_cluster_menu.pack(side="left", padx=(0, 15))

    app.lbl_start_btn = ctk.CTkButton(
        ctrl_frame,
        text="▶ Bắt Đầu Gán Nhãn",
        width=160,
        fg_color="#2563EB",
        hover_color="#1D4ED8",
        command=app._lbl_start_labeling,
    )
    app.lbl_start_btn.pack(side="left", padx=5)

    app.lbl_stop_btn = ctk.CTkButton(
        ctrl_frame,
        text="⏹ Dừng",
        width=80,
        fg_color="red",
        hover_color="darkred",
        state="disabled",
        command=app._lbl_stop_labeling,
    )
    app.lbl_stop_btn.pack(side="left", padx=5)

    app.lbl_reset_btn = ctk.CTkButton(
        ctrl_frame,
        text="🔄 Reset",
        width=80,
        fg_color="#6B7280",
        hover_color="#4B5563",
        command=app._lbl_reset_labeled_data,
    )
    app.lbl_reset_btn.pack(side="left", padx=5)

    app.lbl_progress_var = ctk.DoubleVar(value=0.0)
    app.lbl_progress = ctk.CTkProgressBar(ctrl_frame, variable=app.lbl_progress_var, width=250)
    app.lbl_progress.pack(side="left", padx=(15, 5))
    app.lbl_progress.set(0)

    app.lbl_progress_text = ctk.CTkLabel(ctrl_frame, text="0 / 0", text_color="gray")
    app.lbl_progress_text.pack(side="left", padx=5)

    work_frame = ctk.CTkFrame(tab)
    work_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
    work_frame.grid_columnconfigure(1, weight=3)
    work_frame.grid_columnconfigure(0, weight=1)
    work_frame.grid_rowconfigure(0, weight=1)

    app.lbl_log_textbox = ctk.CTkTextbox(work_frame, corner_radius=5)
    app.lbl_log_textbox.grid(row=0, column=0, padx=(5, 5), pady=5, sticky="nsew")
    app.lbl_log_textbox.insert("0.0", "Sẵn sàng. (Gemini: set GEMINI_API_KEY trong .env)\n")
    app.lbl_log_textbox.configure(state="disabled")

    table_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    table_frame.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    app.lbl_tree = ttk.Treeview(table_frame, columns=("id", "text", "tier1", "tier2"), show="headings")
    app.lbl_tree.heading("id", text="ID")
    app.lbl_tree.heading("text", text="Nội dung bình luận")
    app.lbl_tree.heading("tier1", text="Tier1 Toxic")
    app.lbl_tree.heading("tier2", text="Tier2 Labels")
    app.lbl_tree.column("id", width=40, anchor="center")
    app.lbl_tree.column("text", width=400, anchor="w")
    app.lbl_tree.column("tier1", width=80, anchor="center")
    app.lbl_tree.column("tier2", width=180, anchor="center")

    lbl_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=app.lbl_tree.yview)
    app.lbl_tree.configure(yscrollcommand=lbl_scroll.set)
    app.lbl_tree.grid(row=0, column=0, sticky="nsew")
    lbl_scroll.grid(row=0, column=1, sticky="ns")

    app.lbl_status_label = ctk.CTkLabel(tab, text="Trạng thái: Đang chờ lệnh", text_color="gray")
    app.lbl_status_label.grid(row=3, column=0, pady=5, sticky="w", padx=10)

    app._lbl_is_running = False
    app._lbl_stop_event = threading.Event()

    # Init cluster options
    app._lbl_cluster_size = CLUSTER_SIZE_DEFAULT
    app._lbl_cluster_options = {}
    app._lbl_refresh_clusters()


def _lbl_log(app, msg):
    app.lbl_log_textbox.configure(state="normal")
    app.lbl_log_textbox.insert("end", f"{msg}\n")
    app.lbl_log_textbox.see("end")
    app.lbl_log_textbox.configure(state="disabled")


def _lbl_cluster_history_path(app) -> str:
    return os.path.join(_pipeline_dir(), "cluster_history.json")


def _lbl_load_cluster_history(app) -> dict:
    path = app._lbl_cluster_history_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = __import__("json").load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _lbl_save_cluster_history(app, data: dict) -> None:
    path = app._lbl_cluster_history_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            __import__("json").dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _lbl_refresh_clusters(app):
    """Recompute cluster list from current warehouse.csv and refresh the dropdown."""
    try:
        clusters = get_warehouse_clusters(cluster_size=app._lbl_cluster_size)
    except Exception:
        clusters = []

    try:
        total_rows = int(get_warehouse_count())
    except Exception:
        total_rows = 0

    options: dict[str, dict] = {}
    values: list[str] = []

    if total_rows > 0:
        all_label = f"Toàn bộ (1–{total_rows}) — {total_rows}"
        options[all_label] = {
            "cluster_index": -1,
            "start_id": 1,
            "end_id": total_rows,
            "size": total_rows,
            "is_all": True,
        }
        values.append(all_label)

    for c in clusters:
        idx = int(c.get("cluster_index", 0))
        start_row = int(c.get("start_row", 0))
        end_row = int(c.get("end_row", 0))
        size = int(c.get("size", 0))
        start_id = start_row + 1
        end_id = end_row + 1
        label = f"Cluster {idx + 1} ({start_id}–{end_id}) — {size}"
        options[label] = {"cluster_index": idx, "start_id": start_id, "end_id": end_id, "size": size}
        values.append(label)

    if not values:
        values = ["(warehouse trống)"]
        options = {values[0]: {"cluster_index": 0, "start_id": 0, "end_id": 0, "size": 0}}

    app._lbl_cluster_options = options
    try:
        app.lbl_cluster_menu.configure(values=values)
    except Exception:
        return

    history = app._lbl_load_cluster_history()
    last_idx = history.get("last_selected_cluster_index", 0)
    if not isinstance(last_idx, int):
        last_idx = 0

    selected = None
    for text, meta in options.items():
        if int(meta.get("cluster_index", 0)) == last_idx:
            selected = text
            break
    if selected is None:
        selected = values[0]

    try:
        app.lbl_cluster_var.set(selected)
    except Exception:
        pass


def _lbl_on_cluster_change(app, selected_value: str):
    meta = app._lbl_cluster_options.get(selected_value, {})
    idx = int(meta.get("cluster_index", 0)) if meta else 0

    history = app._lbl_load_cluster_history()
    history["cluster_size"] = int(app._lbl_cluster_size)
    history["last_selected_cluster_index"] = idx

    recent = history.get("recent", [])
    if not isinstance(recent, list):
        recent = []
    recent.append(
        {
            "cluster_index": idx,
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "warehouse_rows": get_warehouse_count(),
        }
    )
    history["recent"] = recent[-30:]
    app._lbl_save_cluster_history(history)

    if meta and meta.get("size", 0):
        if meta.get("is_all") or idx < 0:
            app.lbl_status_label.configure(
                text=f"Trạng thái: Đã chọn Toàn bộ (1–{meta.get('end_id')})",
                text_color="gray",
            )
        else:
            app.lbl_status_label.configure(
                text=f"Trạng thái: Đã chọn Cluster {idx + 1} ({meta.get('start_id')}–{meta.get('end_id')})",
                text_color="gray",
            )


def _lbl_test_connection(app):
    model_name = app.lbl_model_var.get().strip()
    effective_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    app._lbl_log(f"🔌 Đang kiểm tra Gemini API (model={effective_model}) ...")

    app.lbl_test_btn.configure(state="disabled", text="Đang kiểm tra...")

    def _test():
        result = GeminiHierarchicalClassifier.test_connection(model=model_name)
        if result["ok"]:
            models_str = ", ".join(result["models"]) if result["models"] else "(unknown model)"
            app.after(0, app._lbl_log, f"✅ Gemini OK! Model: {models_str}")
            app.after(0, lambda: app.lbl_status_label.configure(text="✅ Gemini API sẵn sàng", text_color="green"))
            if result.get("models") and not model_name:
                app.after(0, lambda: app.lbl_model_var.set(result["models"][0]))
        else:
            app.after(0, app._lbl_log, f"❌ Lỗi Gemini: {result['error']}")
            app.after(0, lambda: app.lbl_status_label.configure(text="❌ Không thể kết nối Gemini", text_color="red"))
        app.after(0, lambda: app.lbl_test_btn.configure(state="normal", text="🔌 Test Connection"))

    threading.Thread(target=_test, daemon=True).start()


def _lbl_start_labeling(app):
    if app._lbl_is_running:
        return

    if not os.getenv("GEMINI_API_KEY", "").strip():
        messagebox.showwarning("Lỗi", "Thiếu GEMINI_API_KEY! Hãy set trong .env hoặc environment.")
        return

    selected_cluster_text = (app.lbl_cluster_var.get() or "").strip()
    selected_meta = app._lbl_cluster_options.get(selected_cluster_text, {})
    cluster_index = int(selected_meta.get("cluster_index", 0)) if selected_meta else 0
    is_all = bool(selected_meta.get("is_all")) or cluster_index < 0

    if is_all:
        rows = read_warehouse()
    else:
        rows = read_warehouse_cluster(cluster_index=cluster_index, cluster_size=app._lbl_cluster_size)

    if not rows:
        messagebox.showwarning("Lỗi", "Warehouse trống! Hãy crawl dữ liệu trước.")
        return

    try:
        batch_size = max(1, min(int(app.lbl_batch_var.get()), 20))
    except ValueError:
        batch_size = 5

    try:
        workers = int((app.lbl_workers_var.get() or "").strip() or "1")
    except Exception:
        workers = 1
    workers = max(1, min(workers, 32))

    request_retries = 3
    model_name = app.lbl_model_var.get().strip()
    effective_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    app.lbl_tree.delete(*app.lbl_tree.get_children())
    app.lbl_log_textbox.configure(state="normal")
    app.lbl_log_textbox.delete("0.0", "end")
    app.lbl_log_textbox.configure(state="disabled")
    app.lbl_progress.set(0)
    app.lbl_progress_text.configure(text=f"0 / {len(rows)}")

    app._lbl_stop_event.clear()
    app._lbl_is_running = True
    app.lbl_start_btn.configure(state="disabled", text="Đang chạy...")
    app.lbl_stop_btn.configure(state="normal")
    app.lbl_reset_btn.configure(state="disabled")
    app.lbl_model_entry.configure(state="disabled")

    app._lbl_log(f"🚀 Bắt đầu gán nhãn {len(rows)} bình luận")
    if selected_meta and selected_meta.get("size", 0):
        if is_all:
            app._lbl_log(f"   Phạm vi: Toàn bộ (1–{selected_meta.get('end_id')})")
        else:
            app._lbl_log(
                f"   Cluster: {cluster_index + 1} ({selected_meta.get('start_id')}–{selected_meta.get('end_id')})"
            )
    app._lbl_log("   Provider: Gemini (Google AI Studio)")
    app._lbl_log(f"   Model: {effective_model}")
    app._lbl_log(f"   Batch size: {batch_size}")
    app._lbl_log(f"   Workers: {workers}")
    app._lbl_log("=" * 50)

    def _labeling_thread():
        import pandas as pd

        def _make_classifier():
            return GeminiHierarchicalClassifier(model=effective_model, timeout=120)

        classifier_single = None
        if workers <= 1:
            classifier_single = _make_classifier()

        labeled_path = app._get_labeled_data_path()

        labeled_ids: set[int] = set()
        existing_rows: list[dict] = []
        label_counts: dict[str, int] = {}
        file_exists = False

        if os.path.exists(labeled_path):
            try:
                raw_rows, fieldnames, enc_used = app._read_csv_dicts_with_fallback(labeled_path)
                if enc_used not in ("utf-8-sig", "utf-8", "utf-8(replace)"):
                    app._backup_file(labeled_path)
                    app._rewrite_csv_utf8sig(labeled_path, raw_rows, fieldnames)
                    raw_rows, fieldnames, enc_used = app._read_csv_dicts_with_fallback(labeled_path)

                for erow in raw_rows:
                    try:
                        eid = int(erow.get("id", 0))
                    except (ValueError, TypeError):
                        eid = 0
                    if eid:
                        labeled_ids.add(eid)
                    existing_rows.append(erow)
                    t1 = erow.get("tier1_label", "")
                    t2_str = erow.get("tier2_labels", "")
                    if t1:
                        label_counts[t1] = label_counts.get(t1, 0) + 1
                    if t2_str:
                        for lbl in str(t2_str).split("|"):
                            lbl = lbl.strip()
                            if lbl:
                                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                file_exists = len(labeled_ids) > 0
            except Exception:
                pass

        total = len(rows)

        def _safe_int(v) -> int:
            try:
                return int(v)
            except Exception:
                return 0

        # Resume behavior:
        # - For "Toàn bộ": start from (max labeled id + 1) and go ascending.
        # - For a cluster: skip any already-labeled ids inside that cluster.
        max_labeled_id = max(labeled_ids) if labeled_ids else 0
        resume_start_id = max_labeled_id + 1 if (is_all and max_labeled_id > 0) else None

        # Cluster-scoped resume: only count/pre-fill rows belonging to current selection
        cluster_ids = set(_safe_int(r.get("id", 0)) for r in rows)
        existing_rows_cluster = []
        if existing_rows:
            for erow in existing_rows:
                eid = _safe_int(erow.get("id", 0))
                if eid and eid in cluster_ids:
                    existing_rows_cluster.append(erow)

        # "skipped" is used as progress baseline.
        if is_all and max_labeled_id > 0:
            skipped = min(max_labeled_id, total)
        else:
            skipped = len(existing_rows_cluster)

        if is_all and max_labeled_id > 0:
            app.after(0, app._lbl_log, f"♻ Resume Toàn bộ: đã có đến ID={max_labeled_id}. Sẽ chạy từ ID={resume_start_id} → {total}.")
        elif existing_rows_cluster:
            app.after(0, app._lbl_log, f"♻ Tiếp tục từ {skipped}/{total} dòng đã gán nhãn trước đó")

        # Filter pending rows
        if is_all and resume_start_id is not None:
            pending_rows = [
                r
                for r in rows
                if (_safe_int(r.get("id", 0)) >= resume_start_id) and (_safe_int(r.get("id", 0)) not in labeled_ids)
            ]
        else:
            pending_rows = [r for r in rows if _safe_int(r.get("id", 0)) not in labeled_ids]

        if not pending_rows:
            app.after(0, app._lbl_log, f"✅ Tất cả {total} dòng đã được gán nhãn. Không cần làm gì thêm.")
            stats_str = " | ".join(f"{k}: {v}" for k, v in label_counts.items())
            if stats_str:
                app.after(0, app._lbl_log, f"   Thống kê: [{stats_str}]")
            app.after(0, lambda: app.lbl_progress.set(1.0))
            app.after(0, lambda t=total: app.lbl_progress_text.configure(text=f"{t} / {t}"))
            final_msg = f"✅ Hoàn tất — {total} dòng đã gán nhãn → labeled_data.csv"
            app.after(0, lambda: app.lbl_status_label.configure(text=final_msg, text_color="green"))
            app._lbl_is_running = False
            app.after(0, lambda: app.lbl_start_btn.configure(state="normal", text="▶ Bắt Đầu Gán Nhãn"))
            app.after(0, lambda: app.lbl_stop_btn.configure(state="disabled"))
            app.after(0, lambda: app.lbl_reset_btn.configure(state="normal"))
            app.after(0, lambda: app.lbl_model_entry.configure(state="normal"))
            app.after(0, app.refresh_file_list)
            return

        processed = skipped
        if skipped > 0:
            p = (skipped / total) if total else 0
            app.after(0, lambda v=p: app.lbl_progress.set(v))
            app.after(0, lambda v=skipped, t=total: app.lbl_progress_text.configure(text=f"{v} / {t}"))

        app.after(0, app._lbl_log, f"📋 Còn {len(pending_rows)} dòng cần gán nhãn")

        def _write_batch_results(batch_rows, predictions):
            nonlocal processed, file_exists
            csv_rows = []
            for row_i, pred in zip(batch_rows, predictions):
                t1 = pred.get("tier1_label", "Clean")
                t2_list = pred.get("tier2_labels", []) or []
                t2 = "|".join(t2_list) if t2_list else ""

                label_counts[t1] = label_counts.get(t1, 0) + 1
                for lbl in t2_list:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                processed += 1
                csv_rows.append(
                    {
                        "id": row_i.get("id", processed),
                        "text": row_i.get("text", ""),
                        "tier1_label": t1,
                        "tier2_labels": t2,
                    }
                )

                rid = row_i.get("id", processed)
                dt = str(row_i.get("text", "")).replace("\n", "  ")[:80]
                app.after(0, lambda r=rid, d=dt, s1=t1, s2=t2: app.lbl_tree.insert("", "end", values=(r, d, s1, s2)))

            if csv_rows:
                pd.DataFrame(csv_rows).to_csv(
                    labeled_path,
                    mode="a",
                    header=not file_exists,
                    index=False,
                    encoding="utf-8-sig",
                )
                file_exists = True

            p = (processed / total) if total else 0
            pc = processed
            app.after(0, lambda v=p: app.lbl_progress.set(v))
            app.after(0, lambda v=pc, t=total: app.lbl_progress_text.configure(text=f"{v} / {t}"))
            stats_str = " | ".join(f"{k}: {v}" for k, v in label_counts.items())
            app.after(0, app._lbl_log, f"   ✓ Batch xong. [{stats_str}]")

        def _split_batches():
            batches = []
            buf_rows = []
            buf_tasks = []
            for row in pending_rows:
                if app._lbl_stop_event.is_set():
                    break
                buf_rows.append(row)
                buf_tasks.append({"data": {"text": str(row.get("text", ""))}})
                if len(buf_tasks) >= batch_size:
                    batches.append((buf_rows, buf_tasks))
                    buf_rows = []
                    buf_tasks = []
            if buf_tasks and not app._lbl_stop_event.is_set():
                batches.append((buf_rows, buf_tasks))
            return batches

        try:
            batches = _split_batches()
            if app._lbl_stop_event.is_set():
                app.after(0, app._lbl_log, "🛑 Đã nhận lệnh DỪNG. Dữ liệu đã gán được lưu.")

            if workers <= 1:
                for i, (batch_rows, batch_tasks) in enumerate(batches):
                    if app._lbl_stop_event.is_set():
                        app.after(0, app._lbl_log, "🛑 Đã nhận lệnh DỪNG. Dữ liệu đã gán được lưu.")
                        break
                    app.after(0, app._lbl_log, f"📤 Gửi batch {i + 1}/{len(batches)} (n={len(batch_tasks)}) ...")
                    try:
                        predictions = classifier_single.predict(batch_tasks, retries=request_retries, strict=True)
                    except GeminiSafetyBlockError as e:
                        # Safety block is deterministic — log and use defaults, don't stop the job.
                        app.after(0, app._lbl_log, f"⚠️ Batch {i + 1}/{len(batches)} bị safety-block: {e}. Gán nhãn mặc định (Clean/Neutral).")
                        predictions = [{"tier1_label": "Clean", "tier2_labels": ["Neutral"]} for _ in batch_tasks]
                    except Exception as e:
                        app._lbl_stop_event.set()
                        raise RuntimeError(f"Request thất bại ở batch {i + 1}/{len(batches)}: {e}") from e

                    if not isinstance(predictions, list) or len(predictions) != len(batch_tasks):
                        app._lbl_stop_event.set()
                        raise RuntimeError(
                            f"Response không hợp lệ ở batch {i + 1}/{len(batches)}: got {type(predictions)} len={getattr(predictions, '__len__', lambda: '?')()}"
                        )
                    _write_batch_results(batch_rows, predictions)
            else:
                thread_local = threading.local()

                def _predict_batch(batch_tasks):
                    clf = getattr(thread_local, "classifier", None)
                    if clf is None:
                        clf = _make_classifier()
                        thread_local.classifier = clf
                    return clf.predict(batch_tasks, retries=request_retries, strict=True)

                max_inflight = max(1, min(len(batches), workers * 2))
                inflight: dict[int, concurrent.futures.Future] = {}
                next_to_write = 0
                next_to_submit = 0

                ex = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
                try:
                    while next_to_write < len(batches):
                        if app._lbl_stop_event.is_set():
                            app.after(0, app._lbl_log, "🛑 Đã nhận lệnh DỪNG. Đang hủy các batch còn lại...")
                            break

                        while (
                            (not app._lbl_stop_event.is_set())
                            and next_to_submit < len(batches)
                            and len(inflight) < max_inflight
                        ):
                            batch_rows, batch_tasks = batches[next_to_submit]
                            app.after(
                                0,
                                app._lbl_log,
                                f"📤 (Song song) Submit batch {next_to_submit + 1}/{len(batches)} (n={len(batch_tasks)}) ...",
                            )
                            inflight[next_to_submit] = ex.submit(_predict_batch, batch_tasks)
                            next_to_submit += 1

                        fut = inflight.get(next_to_write)
                        if fut is None:
                            break

                        batch_rows, batch_tasks = batches[next_to_write]
                        try:
                            predictions = fut.result()
                        except GeminiSafetyBlockError as e:
                            app.after(0, app._lbl_log, f"⚠️ Batch {next_to_write + 1}/{len(batches)} bị safety-block: {e}. Gán nhãn mặc định (Clean/Neutral).")
                            predictions = [{"tier1_label": "Clean", "tier2_labels": ["Neutral"]} for _ in batch_tasks]
                        except Exception as e:
                            app._lbl_stop_event.set()
                            raise RuntimeError(f"Request thất bại ở batch {next_to_write + 1}/{len(batches)}: {e}") from e

                        inflight.pop(next_to_write, None)
                        if not isinstance(predictions, list) or len(predictions) != len(batch_tasks):
                            app._lbl_stop_event.set()
                            raise RuntimeError(
                                f"Response không hợp lệ ở batch {next_to_write + 1}/{len(batches)}: got {type(predictions)} len={getattr(predictions, '__len__', lambda: '?')()}"
                            )

                        _write_batch_results(batch_rows, predictions)
                        next_to_write += 1
                finally:
                    ex.shutdown(wait=not app._lbl_stop_event.is_set(), cancel_futures=True)

            newly_labeled = processed - skipped
            if app._lbl_stop_event.is_set():
                app.after(0, app._lbl_log, "\n" + "=" * 50)
                app.after(0, app._lbl_log, f"🛑 ĐÃ DỪNG: {processed}/{total} bình luận đã được lưu ({newly_labeled} mới gán)")
                stats_str = " | ".join(f"{k}: {v}" for k, v in label_counts.items())
                if stats_str:
                    app.after(0, app._lbl_log, f"   Thống kê: [{stats_str}]")
                app.after(0, app._lbl_log, f"📂 Đã lưu: {labeled_path}")
                stop_msg = f"🛑 Đã dừng — {processed}/{total} dòng đã lưu"
                app.after(0, lambda: app.lbl_status_label.configure(text=stop_msg, text_color="orange"))
            else:
                app.after(0, app._lbl_log, "\n" + "=" * 50)
                app.after(0, app._lbl_log, f"✅ HOÀN TẤT: {processed}/{total} bình luận đã gán nhãn ({newly_labeled} mới gán)")
                for lbl, cnt in label_counts.items():
                    app.after(0, app._lbl_log, f"   {lbl}: {cnt}")
                app.after(0, app._lbl_log, f"📂 Đã lưu: {labeled_path}")
                final_msg = f"✅ Hoàn tất — {processed}/{total} dòng đã gán nhãn ({newly_labeled} mới) → labeled_data.csv"
                app.after(0, lambda: app.lbl_status_label.configure(text=final_msg, text_color="green"))
        except Exception as e:
            err_msg = str(e)
            app.after(0, app._lbl_log, f"❌ Lỗi: {err_msg}")
            app.after(0, lambda: app.lbl_status_label.configure(text=f"❌ Lỗi: {err_msg[:80]}", text_color="red"))
        finally:
            app._lbl_is_running = False
            app.after(0, lambda: app.lbl_start_btn.configure(state="normal", text="▶ Bắt Đầu Gán Nhãn"))
            app.after(0, lambda: app.lbl_stop_btn.configure(state="disabled"))
            app.after(0, lambda: app.lbl_reset_btn.configure(state="normal"))
            app.after(0, lambda: app.lbl_model_entry.configure(state="normal"))
            app.after(0, app.refresh_file_list)

    threading.Thread(target=_labeling_thread, daemon=True).start()


def _lbl_stop_labeling(app):
    if app._lbl_is_running:
        app._lbl_log("⏹ Đang gửi lệnh dừng...")
        app._lbl_stop_event.set()
        app.lbl_stop_btn.configure(state="disabled", text="Đang dừng...")


def _lbl_reset_labeled_data(app):
    if app._lbl_is_running:
        messagebox.showwarning("Lỗi", "Không thể reset khi đang gán nhãn! Hãy dừng trước.")
        return

    labeled_path = app._get_labeled_data_path()
    if not os.path.exists(labeled_path):
        messagebox.showinfo("Thông báo", "Chưa có file labeled_data.csv để xóa.")
        return

    confirm = messagebox.askyesno(
        "Xác nhận Reset",
        "Bạn có chắc chắn muốn xóa toàn bộ dữ liệu đã gán nhãn?\n\n"
        "File labeled_data.csv sẽ bị xóa và bạn phải gán nhãn lại từ đầu.\n"
        "Hành động này KHÔNG THỂ hoàn tác!",
        icon="warning",
    )

    if not confirm:
        return

    try:
        os.remove(labeled_path)
        app.lbl_tree.delete(*app.lbl_tree.get_children())
        app.lbl_progress.set(0)
        app.lbl_progress_text.configure(text="0 / 0")
        app.lbl_log_textbox.configure(state="normal")
        app.lbl_log_textbox.delete("0.0", "end")
        app.lbl_log_textbox.configure(state="disabled")
        app._lbl_log("🔄 Đã reset — labeled_data.csv đã bị xóa.")
        app._lbl_log("Sẵn sàng gán nhãn lại từ đầu.")
        app.lbl_status_label.configure(text="🔄 Đã reset dữ liệu gán nhãn", text_color="orange")
        app.refresh_file_list()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xóa file: {e}")
