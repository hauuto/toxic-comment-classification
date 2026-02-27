import threading

import customtkinter as ctk
from tkinter import messagebox, ttk

from crawler import extract_comments_stream
from nlp_pipeline.warehouse import append_to_warehouse


def _setup_crawler_tab(app):
    tab = app.tab_crawler
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(2, weight=1)

    input_frame = ctk.CTkFrame(tab)
    input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
    input_frame.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(input_frame, text="Nhập URL bài viết (cách nhau bởi dấu ;):").grid(
        row=0, column=0, padx=10, pady=10
    )
    app.url_entry = ctk.CTkEntry(
        input_frame,
        placeholder_text="https://www.facebook.com/...; https://youtube.com/...",
    )
    app.url_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")

    opt_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    opt_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    app.headless_var = ctk.BooleanVar(value=False)
    app.headless_checkbox = ctk.CTkCheckBox(opt_frame, text="Chạy ẩn (Headless)", variable=app.headless_var)
    app.headless_checkbox.pack(side="left", padx=(0, 20))

    app.use_decoder_var = ctk.BooleanVar(value=True)
    app.use_filter_var = ctk.BooleanVar(value=True)
    app.use_normalizer_var = ctk.BooleanVar(value=True)
    app.seg_backend_var = ctk.StringVar(value="VnCoreNLP")

    nlp_frame = ctk.CTkFrame(opt_frame, fg_color="transparent")
    nlp_frame.pack(side="left", fill="x", expand=True)

    ctk.CTkLabel(nlp_frame, text="Pipeline Các Bước Tiền Xử Lý:").pack(side="left", padx=(0, 10))
    app.chk_dec = ctk.CTkCheckBox(nlp_frame, text="Decoder", variable=app.use_decoder_var)
    app.chk_dec.pack(side="left", padx=5)
    app.chk_fil = ctk.CTkCheckBox(nlp_frame, text="Filter", variable=app.use_filter_var)
    app.chk_fil.pack(side="left", padx=5)
    app.chk_nor = ctk.CTkCheckBox(nlp_frame, text="Normalizer", variable=app.use_normalizer_var)
    app.chk_nor.pack(side="left", padx=5)
    ctk.CTkLabel(nlp_frame, text="Tách từ:").pack(side="left", padx=(10, 5))
    app.seg_backend_menu = ctk.CTkOptionMenu(
        nlp_frame,
        values=["Tắt", "VnCoreNLP", "Underthesea"],
        variable=app.seg_backend_var,
        width=130,
    )
    app.seg_backend_menu.pack(side="left", padx=5)

    btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

    app.run_button = ctk.CTkButton(btn_frame, text="Bắt Đầu", command=app.start_crawling)
    app.run_button.pack(side="left", padx=10)

    app.stop_button = ctk.CTkButton(
        btn_frame,
        text="Dừng",
        command=app.stop_crawling,
        state="disabled",
        fg_color="red",
        hover_color="darkred",
    )
    app.stop_button.pack(side="left", padx=10)

    work_frame = ctk.CTkFrame(tab)
    work_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
    work_frame.grid_columnconfigure(1, weight=3)
    work_frame.grid_columnconfigure(0, weight=1)
    work_frame.grid_rowconfigure(0, weight=1)

    app.log_textbox = ctk.CTkTextbox(work_frame, corner_radius=5)
    app.log_textbox.grid(row=0, column=0, padx=(5, 5), pady=5, sticky="nsew")
    app.log_textbox.insert("0.0", "Hệ thống sẵn sàng.\n")
    app.log_textbox.configure(state="disabled")

    table_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    table_frame.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    columns = ("id", "text")
    app.tree_data = ttk.Treeview(table_frame, columns=columns, show="headings")
    app.tree_data.heading("id", text="ID")
    app.tree_data.heading("text", text="Nội dung bình luận đã quét")
    app.tree_data.column("id", width=50, anchor="center")
    app.tree_data.column("text", width=400, anchor="w")

    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=app.tree_data.yview)
    app.tree_data.configure(yscrollcommand=scrollbar.set)

    app.tree_data.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    app.status_label = ctk.CTkLabel(tab, text="Trạng thái: Đang chờ lệnh", text_color="gray")
    app.status_label.grid(row=3, column=0, pady=5, sticky="w", padx=10)


def log_message(app, message):
    app.log_textbox.configure(state="normal")
    app.log_textbox.insert("end", f"{message}\n")
    app.log_textbox.see("end")
    app.log_textbox.configure(state="disabled")


def handle_new_data(app, batch):
    app.after(0, app._append_to_table, batch)
    try:
        append_to_warehouse(batch)
        app.after(0, app._lbl_refresh_clusters)
    except Exception as e:
        app.after(0, app.log_message, f"Lỗi ghi warehouse: {e}")
    app.extracted_data.extend(batch)


def _append_to_table(app, batch):
    for item in batch:
        display_text = item["text"].replace("\n", "  ")
        app.tree_data.insert("", "end", values=(item["id"], display_text))

    if len(app.tree_data.get_children()) > 0:
        app.tree_data.yview_moveto(1)

    app.status_label.configure(
        text=f"Đã thu thập & lọc: {len(app.extracted_data)} bình luận",
        text_color="green",
    )


def set_gui_state(app, running):
    if running:
        app.run_button.configure(state="disabled", text="Đang chạy...")
        app.stop_button.configure(state="normal")
        app.url_entry.configure(state="disabled")
        app.chk_dec.configure(state="disabled")
        app.chk_fil.configure(state="disabled")
        app.chk_nor.configure(state="disabled")
        app.seg_backend_menu.configure(state="disabled")
        app.is_running = True
    else:
        app.run_button.configure(state="normal", text="Bắt Đầu")
        app.stop_button.configure(state="disabled")
        app.url_entry.configure(state="normal")
        app.chk_dec.configure(state="normal")
        app.chk_fil.configure(state="normal")
        app.chk_nor.configure(state="normal")
        app.seg_backend_menu.configure(state="normal")
        app.is_running = False
        app.refresh_file_list()


def _crawl_thread(app, url, headless, u_dec, u_fil, u_nor, use_segmentor, segmentor_backend, preprocessor):
    try:
        extract_comments_stream(
            url_input=url,
            headless=headless,
            use_decoder=u_dec,
            use_filter=u_fil,
            use_normalizer=u_nor,
            use_segmentor=use_segmentor,
            segmentor_backend=segmentor_backend,
            preprocessor=preprocessor,
            log_callback=lambda msg: app.after(0, app.log_message, msg),
            data_callback=app.handle_new_data,
            stop_event=app.stop_event,
        )
        app.after(
            0,
            app.log_message,
            f"--- HOÀN TẤT. Đã lưu tổng {len(app.extracted_data)} mục vào warehouse. ---",
        )
    except Exception as e:
        app.after(0, app.log_message, f"Lỗi không xác định: {str(e)}")
    finally:
        app.after(0, app.set_gui_state, False)


def start_crawling(app):
    if app.is_running:
        return

    url_input = app.url_entry.get().strip()
    if not url_input:
        messagebox.showwarning("Lỗi", "Vui lòng nhập ít nhất một URL!")
        return

    headless = app.headless_var.get()
    u_dec = app.use_decoder_var.get()
    u_fil = app.use_filter_var.get()
    u_nor = app.use_normalizer_var.get()

    seg_choice = (app.seg_backend_var.get() or "").strip()
    if seg_choice == "Underthesea":
        segmentor_backend = "underthesea"
        use_segmentor = True
    elif seg_choice == "VnCoreNLP":
        segmentor_backend = "vncorenlp"
        use_segmentor = True
    else:
        segmentor_backend = "whitespace"
        use_segmentor = False

    preprocessor = None
    if u_dec or u_fil or u_nor or use_segmentor:
        preprocessor = app._get_preprocessor(segmentor_backend)

    app.extracted_data = []
    app.tree_data.delete(*app.tree_data.get_children())
    app.log_textbox.configure(state="normal")
    app.log_textbox.delete("0.0", "end")
    app.log_textbox.configure(state="disabled")

    app.stop_event.clear()
    app.set_gui_state(True)
    app.log_message("Dữ liệu sẽ được lưu vào warehouse.csv (đã qua pipeline) và warehouse_raw.csv (data gốc chỉ qua Filter)")

    thread = threading.Thread(
        target=app._crawl_thread,
        args=(
            url_input,
            headless,
            u_dec,
            u_fil,
            u_nor,
            use_segmentor,
            segmentor_backend,
            preprocessor,
        ),
        daemon=True,
    )
    thread.start()


def stop_crawling(app):
    if app.is_running:
        app.log_message("Đang gửi lệnh yêu cầu dừng (vui lòng chờ vài giây)...")
        app.stop_event.set()
        app.stop_button.configure(state="disabled", text="Đang dừng...")
