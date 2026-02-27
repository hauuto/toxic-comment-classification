import threading

import customtkinter as ctk
from tkinter import messagebox, ttk

from crawler import VOZCrawler, ThreadsCrawler, load_keyword_history
from nlp_pipeline.warehouse import get_warehouse_count


def _setup_keyword_crawler_tab(app):
    tab = app.tab_keyword
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(2, weight=1)

    input_frame = ctk.CTkFrame(tab)
    input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
    input_frame.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(input_frame, text="Keyword:").grid(row=0, column=0, padx=10, pady=10)
    app.kw_entry = ctk.CTkEntry(input_frame, placeholder_text="Nhập từ khóa (nhiều keyword cách nhau bằng dấu ;)")
    app.kw_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")

    ctk.CTkLabel(input_frame, text="Platform:").grid(row=0, column=2, padx=(10, 5), pady=10)
    app.kw_platform_var = ctk.StringVar(value="VOZ")
    app.kw_platform_menu = ctk.CTkOptionMenu(
        input_frame,
        values=["VOZ", "Threads"],
        variable=app.kw_platform_var,
        command=app._on_kw_platform_change,
        width=120,
    )
    app.kw_platform_menu.grid(row=0, column=3, padx=(0, 10), pady=10)

    param_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    param_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=5, sticky="w")

    app.kw_voz_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
    ctk.CTkLabel(app.kw_voz_frame, text="Max Threads:").pack(side="left", padx=(0, 5))
    app.kw_max_threads_var = ctk.StringVar(value="10")
    ctk.CTkEntry(app.kw_voz_frame, textvariable=app.kw_max_threads_var, width=60).pack(side="left", padx=(0, 15))
    ctk.CTkLabel(app.kw_voz_frame, text="Max Pages:").pack(side="left", padx=(0, 5))
    app.kw_max_pages_var = ctk.StringVar(value="50")
    ctk.CTkEntry(app.kw_voz_frame, textvariable=app.kw_max_pages_var, width=60).pack(side="left", padx=(0, 15))
    ctk.CTkLabel(app.kw_voz_frame, text="Workers:").pack(side="left", padx=(0, 5))
    app.kw_num_workers_var = ctk.StringVar(value="3")
    ctk.CTkEntry(app.kw_voz_frame, textvariable=app.kw_num_workers_var, width=40).pack(side="left", padx=(0, 15))

    app.kw_threads_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
    ctk.CTkLabel(app.kw_threads_frame, text="Max Posts:").pack(side="left", padx=(0, 5))
    app.kw_max_posts_var = ctk.StringVar(value="10")
    ctk.CTkEntry(app.kw_threads_frame, textvariable=app.kw_max_posts_var, width=60).pack(side="left", padx=(0, 15))
    ctk.CTkLabel(app.kw_threads_frame, text="Max Scroll:").pack(side="left", padx=(0, 5))
    app.kw_max_scroll_var = ctk.StringVar(value="30")
    ctk.CTkEntry(app.kw_threads_frame, textvariable=app.kw_max_scroll_var, width=60).pack(side="left", padx=(0, 15))

    app.kw_voz_frame.pack(side="left")

    nlp_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    nlp_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=5, sticky="w")
    ctk.CTkLabel(nlp_frame, text="Tiền Xử Lý:").pack(side="left", padx=(0, 10))

    app.kw_dec_var = ctk.BooleanVar(value=True)
    app.kw_fil_var = ctk.BooleanVar(value=True)
    app.kw_nor_var = ctk.BooleanVar(value=True)
    app.kw_seg_backend_var = ctk.StringVar(value="VnCoreNLP")

    app.kw_chk_dec = ctk.CTkCheckBox(nlp_frame, text="Decoder", variable=app.kw_dec_var)
    app.kw_chk_dec.pack(side="left", padx=5)
    app.kw_chk_fil = ctk.CTkCheckBox(nlp_frame, text="Filter", variable=app.kw_fil_var)
    app.kw_chk_fil.pack(side="left", padx=5)
    app.kw_chk_nor = ctk.CTkCheckBox(nlp_frame, text="Normalizer", variable=app.kw_nor_var)
    app.kw_chk_nor.pack(side="left", padx=5)
    ctk.CTkLabel(nlp_frame, text="Tách từ:").pack(side="left", padx=(10, 5))
    app.kw_seg_backend_menu = ctk.CTkOptionMenu(
        nlp_frame,
        values=["Tắt", "VnCoreNLP", "Underthesea"],
        variable=app.kw_seg_backend_var,
        width=130,
    )
    app.kw_seg_backend_menu.pack(side="left", padx=5)

    btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    btn_frame.grid(row=3, column=0, columnspan=4, pady=10)

    app.kw_run_button = ctk.CTkButton(btn_frame, text="Bắt Đầu", command=app.start_keyword_crawling)
    app.kw_run_button.pack(side="left", padx=10)
    app.kw_stop_button = ctk.CTkButton(
        btn_frame,
        text="Dừng",
        command=app.stop_keyword_crawling,
        state="disabled",
        fg_color="red",
        hover_color="darkred",
    )
    app.kw_stop_button.pack(side="left", padx=10)

    work_frame = ctk.CTkFrame(tab)
    work_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
    work_frame.grid_columnconfigure(0, weight=1)
    work_frame.grid_columnconfigure(1, weight=3)
    work_frame.grid_columnconfigure(2, weight=1)
    work_frame.grid_rowconfigure(0, weight=1)

    app.kw_log_textbox = ctk.CTkTextbox(work_frame, corner_radius=5)
    app.kw_log_textbox.grid(row=0, column=0, padx=(5, 5), pady=5, sticky="nsew")
    app.kw_log_textbox.insert("0.0", "Hệ thống Keyword Crawler sẵn sàng.\n")
    app.kw_log_textbox.configure(state="disabled")

    table_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    table_frame.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    kw_columns = ("id", "text")
    app.kw_tree_data = ttk.Treeview(table_frame, columns=kw_columns, show="headings")
    app.kw_tree_data.heading("id", text="ID")
    app.kw_tree_data.heading("text", text="Nội dung bình luận")
    app.kw_tree_data.column("id", width=50, anchor="center")
    app.kw_tree_data.column("text", width=400, anchor="w")
    kw_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=app.kw_tree_data.yview)
    app.kw_tree_data.configure(yscrollcommand=kw_scroll.set)
    app.kw_tree_data.grid(row=0, column=0, sticky="nsew")
    kw_scroll.grid(row=0, column=1, sticky="ns")

    history_frame = ctk.CTkFrame(work_frame)
    history_frame.grid(row=0, column=2, padx=(5, 5), pady=5, sticky="nsew")
    history_frame.grid_rowconfigure(1, weight=1)
    history_frame.grid_columnconfigure(0, weight=1)

    hist_header = ctk.CTkFrame(history_frame, fg_color="transparent")
    hist_header.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    ctk.CTkLabel(hist_header, text="Lịch sử Keyword", font=("Arial", 13, "bold")).pack(side="left")
    ctk.CTkButton(hist_header, text="⟳", width=30, command=app.refresh_keyword_history).pack(side="right", padx=2)
    ctk.CTkButton(hist_header, text="Cào lại", width=60, command=app._reuse_history_keyword).pack(side="right", padx=2)

    hist_tree_frame = ctk.CTkFrame(history_frame, fg_color="transparent")
    hist_tree_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
    hist_tree_frame.grid_rowconfigure(0, weight=1)
    hist_tree_frame.grid_columnconfigure(0, weight=1)

    app.kw_history_tree = ttk.Treeview(hist_tree_frame, columns=("platform", "keyword"), show="headings", height=10)
    app.kw_history_tree.heading("platform", text="Platform")
    app.kw_history_tree.heading("keyword", text="Keyword")
    app.kw_history_tree.column("platform", width=70, anchor="center")
    app.kw_history_tree.column("keyword", width=130, anchor="w")
    hist_scroll = ttk.Scrollbar(hist_tree_frame, orient="vertical", command=app.kw_history_tree.yview)
    app.kw_history_tree.configure(yscrollcommand=hist_scroll.set)
    app.kw_history_tree.grid(row=0, column=0, sticky="nsew")
    hist_scroll.grid(row=0, column=1, sticky="ns")

    app.kw_status_label = ctk.CTkLabel(tab, text="Trạng thái: Đang chờ lệnh", text_color="gray")
    app.kw_status_label.grid(row=3, column=0, pady=5, sticky="w", padx=10)

    app.kw_is_running = False
    app.kw_stop_event = threading.Event()
    app.kw_extracted_data = []
    app.kw_active_crawler = None

    app.refresh_keyword_history()


def _on_kw_platform_change(app, platform):
    if platform == "VOZ":
        app.kw_threads_frame.pack_forget()
        app.kw_voz_frame.pack(side="left")
    else:
        app.kw_voz_frame.pack_forget()
        app.kw_threads_frame.pack(side="left")


def refresh_keyword_history(app):
    for item in app.kw_history_tree.get_children():
        app.kw_history_tree.delete(item)
    try:
        history = load_keyword_history()
        for platform in ["voz", "threads"]:
            for kw in history.get(platform, []):
                app.kw_history_tree.insert("", "end", values=(platform.upper(), kw))
    except Exception:
        pass


def _reuse_history_keyword(app):
    selected = app.kw_history_tree.selection()
    if not selected:
        messagebox.showwarning("Nhắc nhở", "Hãy chọn 1 keyword từ lịch sử.")
        return
    item = app.kw_history_tree.item(selected[0])
    platform = item["values"][0]
    keyword = item["values"][1]
    app.kw_entry.delete(0, "end")
    app.kw_entry.insert(0, keyword)
    app.kw_platform_var.set(platform)
    app._on_kw_platform_change(platform)


def kw_log_message(app, message):
    app.kw_log_textbox.configure(state="normal")
    app.kw_log_textbox.insert("end", f"{message}\n")
    app.kw_log_textbox.see("end")
    app.kw_log_textbox.configure(state="disabled")


def kw_handle_new_data(app, batch):
    app.after(0, app._kw_append_to_table, batch)
    app.kw_extracted_data.extend(batch)


def _kw_append_to_table(app, batch):
    for item in batch:
        display_text = str(item.get("text", "")).replace("\n", "  ")
        app.kw_tree_data.insert("", "end", values=(item.get("id", ""), display_text))
    if len(app.kw_tree_data.get_children()) > 0:
        app.kw_tree_data.yview_moveto(1)
    wh_count = get_warehouse_count()
    app.kw_status_label.configure(
        text=f"Thu thập: {len(app.kw_extracted_data)} | Warehouse: {wh_count} dòng",
        text_color="green",
    )


def _set_kw_gui_state(app, running):
    if running:
        app.kw_run_button.configure(state="disabled", text="Đang chạy...")
        app.kw_stop_button.configure(state="normal")
        app.kw_entry.configure(state="disabled")
        app.kw_platform_menu.configure(state="disabled")
        app.kw_chk_dec.configure(state="disabled")
        app.kw_chk_fil.configure(state="disabled")
        app.kw_chk_nor.configure(state="disabled")
        app.kw_seg_backend_menu.configure(state="disabled")
        app.kw_is_running = True
    else:
        app.kw_run_button.configure(state="normal", text="Bắt Đầu")
        app.kw_stop_button.configure(state="disabled", text="Dừng")
        app.kw_entry.configure(state="normal")
        app.kw_platform_menu.configure(state="normal")
        app.kw_chk_dec.configure(state="normal")
        app.kw_chk_fil.configure(state="normal")
        app.kw_chk_nor.configure(state="normal")
        app.kw_seg_backend_menu.configure(state="normal")
        app.kw_is_running = False
        app.refresh_file_list()
        app.refresh_keyword_history()


def _keyword_crawl_thread(
    app,
    keywords,
    platform,
    u_dec,
    u_fil,
    u_nor,
    use_segmentor,
    preprocessor,
    max_threads,
    max_pages,
    max_posts,
    max_scroll,
    num_workers=3,
):
    crawler = None
    try:
        if preprocessor is not None:
            app.after(0, app.kw_log_message, "Đang khởi tạo bộ tiền xử lý NLP...")

        log_cb = lambda msg: app.after(0, app.kw_log_message, msg)

        for kw_idx, keyword in enumerate(keywords, start=1):
            if app.kw_stop_event.is_set():
                app.after(0, app.kw_log_message, "Đã nhận lệnh DỪNG.")
                break

            app.after(
                0,
                app.kw_log_message,
                f"\n{'='*50}\n[{kw_idx}/{len(keywords)}] Keyword: {keyword}\n{'='*50}",
            )

            if crawler:
                try:
                    crawler.close()
                except Exception:
                    pass

            if platform == "VOZ":
                crawler = VOZCrawler(
                    keyword=keyword,
                    max_threads=max_threads,
                    max_pages=max_pages,
                    log_callback=log_cb,
                    stop_event=app.kw_stop_event,
                    data_callback=app.kw_handle_new_data,
                    preprocessor=preprocessor,
                    use_decoder=u_dec,
                    use_filter=u_fil,
                    use_normalizer=u_nor,
                    use_segmentor=use_segmentor,
                    num_workers=num_workers,
                )
            else:
                crawler = ThreadsCrawler(
                    keyword=keyword,
                    max_posts=max_posts,
                    max_scroll=max_scroll,
                    log_callback=log_cb,
                    stop_event=app.kw_stop_event,
                    data_callback=app.kw_handle_new_data,
                    preprocessor=preprocessor,
                    use_decoder=u_dec,
                    use_filter=u_fil,
                    use_normalizer=u_nor,
                    use_segmentor=use_segmentor,
                )

            app.kw_active_crawler = crawler
            crawler.crawl_keyword(keyword)

        app.after(
            0,
            app.kw_log_message,
            f"\n--- HOÀN TẤT TẤT CẢ {len(keywords)} KEYWORD. Tổng {len(app.kw_extracted_data)} bình luận. ---",
        )
    except Exception as e:
        app.after(0, app.kw_log_message, f"Lỗi không xác định: {str(e)}")
    finally:
        if crawler:
            try:
                crawler.close()
            except Exception:
                pass
        app.kw_active_crawler = None
        app.after(0, app._set_kw_gui_state, False)


def start_keyword_crawling(app):
    if app.kw_is_running:
        return

    raw_input = app.kw_entry.get().strip()
    if not raw_input:
        messagebox.showwarning("Lỗi", "Vui lòng nhập keyword!")
        return

    keywords = [kw.strip() for kw in raw_input.split(";") if kw.strip()]
    if not keywords:
        messagebox.showwarning("Lỗi", "Vui lòng nhập ít nhất một keyword hợp lệ!")
        return

    platform = app.kw_platform_var.get()

    history = load_keyword_history()
    platform_key = platform.lower()
    already_crawled = [kw for kw in keywords if kw in history.get(platform_key, [])]

    u_dec = app.kw_dec_var.get()
    u_fil = app.kw_fil_var.get()
    u_nor = app.kw_nor_var.get()
    seg_choice = (app.kw_seg_backend_var.get() or "").strip()
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

    max_threads = int(app.kw_max_threads_var.get() or 10)
    max_pages = int(app.kw_max_pages_var.get() or 50)
    max_posts = int(app.kw_max_posts_var.get() or 10)
    max_scroll = int(app.kw_max_scroll_var.get() or 30)
    num_workers = int(app.kw_num_workers_var.get() or 3)

    app.kw_extracted_data = []
    app.kw_tree_data.delete(*app.kw_tree_data.get_children())
    app.kw_log_textbox.configure(state="normal")
    app.kw_log_textbox.delete("0.0", "end")
    app.kw_log_textbox.configure(state="disabled")

    app.kw_stop_event.clear()
    app._set_kw_gui_state(True)

    if already_crawled:
        app.kw_log_message(
            f"⚠ Các keyword đã có trong lịch sử {platform}: {', '.join(already_crawled)}. Vẫn tiếp tục cào..."
        )
    app.kw_log_message(f"Bắt đầu cào {platform} với {len(keywords)} keyword: {'; '.join(keywords)}")

    thread = threading.Thread(
        target=app._keyword_crawl_thread,
        args=(
            keywords,
            platform,
            u_dec,
            u_fil,
            u_nor,
            use_segmentor,
            preprocessor,
            max_threads,
            max_pages,
            max_posts,
            max_scroll,
            num_workers,
        ),
        daemon=True,
    )
    thread.start()


def stop_keyword_crawling(app):
    if app.kw_is_running:
        app.kw_log_message("Đang gửi lệnh yêu cầu dừng...")
        app.kw_stop_event.set()
        app.kw_stop_button.configure(state="disabled", text="Đang dừng...")
        if app.kw_active_crawler:
            try:
                app.kw_active_crawler.close()
            except Exception:
                pass
