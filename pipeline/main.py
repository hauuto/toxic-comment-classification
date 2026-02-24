import os
import csv
import glob
import time
import json
import re
import threading
import customtkinter as ctk
from tkinter import messagebox, ttk

from crawler import extract_comments_stream, VOZCrawler, ThreadsCrawler, load_keyword_history
from nlp_pipeline.warehouse import append_to_warehouse, get_warehouse_count, read_warehouse, overwrite_warehouse
from nlp_pipeline import VietnameseCommentPreprocessor

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FB Comment Management System by 17 Production")
        self.geometry("1100x800")
        self.minsize(950, 650)

        # Background threading variables
        self.extracted_data = []
        self.is_running = False
        self.stop_event = threading.Event()
        self.current_output_file = ""

        # Create Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_crawler = self.tabview.add("Facebook Crawler")
        self.tab_keyword = self.tabview.add("Keyword Crawler")
        self.tab_warehouse = self.tabview.add("Warehouse Manager")
        self.tab_files = self.tabview.add("File Manager")
        self.tab_config = self.tabview.add("Config Manager")

        self._setup_crawler_tab()
        self._setup_keyword_crawler_tab()
        self._setup_warehouse_tab()
        self._setup_file_manager_tab()
        self._setup_config_tab()

    # ---------------------------------------------------------
    # TAB 1: CRAWLER
    # ---------------------------------------------------------
    def _setup_crawler_tab(self):
        tab = self.tab_crawler
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        # Top Control Frame
        input_frame = ctk.CTkFrame(tab)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text="Nhập URL bài viết (cách nhau bởi dấu ;):").grid(row=0, column=0, padx=10, pady=10)
        self.url_entry = ctk.CTkEntry(input_frame, placeholder_text="https://www.facebook.com/...; https://youtube.com/...")
        self.url_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")

        # Options
        opt_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        opt_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        self.headless_var = ctk.BooleanVar(value=False)
        self.headless_checkbox = ctk.CTkCheckBox(opt_frame, text="Chạy ẩn (Headless)", variable=self.headless_var)
        self.headless_checkbox.pack(side="left", padx=(0, 20))

        # NLP Toggles
        self.use_decoder_var = ctk.BooleanVar(value=True)
        self.use_filter_var = ctk.BooleanVar(value=True)
        self.use_normalizer_var = ctk.BooleanVar(value=True)
        self.use_segmentor_var = ctk.BooleanVar(value=True)
        
        nlp_frame = ctk.CTkFrame(opt_frame, fg_color="transparent")
        nlp_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(nlp_frame, text="Pipeline Các Bước Tiền Xử Lý:").pack(side="left", padx=(0, 10))
        self.chk_dec = ctk.CTkCheckBox(nlp_frame, text="Decoder", variable=self.use_decoder_var)
        self.chk_dec.pack(side="left", padx=5)
        self.chk_fil = ctk.CTkCheckBox(nlp_frame, text="Filter", variable=self.use_filter_var)
        self.chk_fil.pack(side="left", padx=5)
        self.chk_nor = ctk.CTkCheckBox(nlp_frame, text="Normalizer", variable=self.use_normalizer_var)
        self.chk_nor.pack(side="left", padx=5)
        self.chk_seg = ctk.CTkCheckBox(nlp_frame, text="VnCoreNLP Segmentor", variable=self.use_segmentor_var)
        self.chk_seg.pack(side="left", padx=5)

        # Buttons
        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.run_button = ctk.CTkButton(btn_frame, text="Bắt Đầu", command=self.start_crawling)
        self.run_button.pack(side="left", padx=10)
        
        self.stop_button = ctk.CTkButton(btn_frame, text="Dừng", command=self.stop_crawling, state="disabled", fg_color="red", hover_color="darkred")
        self.stop_button.pack(side="left", padx=10)

        # Main Workspace: Log (left) + Table (right)
        work_frame = ctk.CTkFrame(tab)
        work_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        work_frame.grid_columnconfigure(1, weight=3) 
        work_frame.grid_columnconfigure(0, weight=1) 
        work_frame.grid_rowconfigure(0, weight=1)

        self.log_textbox = ctk.CTkTextbox(work_frame, corner_radius=5)
        self.log_textbox.grid(row=0, column=0, padx=(5, 5), pady=5, sticky="nsew")
        self.log_textbox.insert("0.0", "Hệ thống sẵn sàng.\n")
        self.log_textbox.configure(state="disabled")

        table_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
        table_frame.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        columns = ("id", "text")
        self.tree_data = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.tree_data.heading("id", text="ID")
        self.tree_data.heading("text", text="Nội dung bình luận đã quét")
        self.tree_data.column("id", width=50, anchor="center")
        self.tree_data.column("text", width=400, anchor="w")
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree_data.yview)
        self.tree_data.configure(yscrollcommand=scrollbar.set)
        
        self.tree_data.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.status_label = ctk.CTkLabel(tab, text="Trạng thái: Đang chờ lệnh", text_color="gray")
        self.status_label.grid(row=3, column=0, pady=5, sticky="w", padx=10)

    # ---------------------------------------------------------
    # TAB 2: KEYWORD CRAWLER (VOZ / Threads)
    # ---------------------------------------------------------
    def _setup_keyword_crawler_tab(self):
        tab = self.tab_keyword
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        # --- Top input frame ---
        input_frame = ctk.CTkFrame(tab)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text="Keyword:").grid(row=0, column=0, padx=10, pady=10)
        self.kw_entry = ctk.CTkEntry(input_frame, placeholder_text="Nhập từ khóa cần cào...")
        self.kw_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")

        ctk.CTkLabel(input_frame, text="Platform:").grid(row=0, column=2, padx=(10, 5), pady=10)
        self.kw_platform_var = ctk.StringVar(value="VOZ")
        self.kw_platform_menu = ctk.CTkOptionMenu(
            input_frame, values=["VOZ", "Threads"],
            variable=self.kw_platform_var, command=self._on_kw_platform_change, width=120
        )
        self.kw_platform_menu.grid(row=0, column=3, padx=(0, 10), pady=10)

        # --- Parameters row ---
        param_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        param_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=5, sticky="w")

        # VOZ params
        self.kw_voz_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        ctk.CTkLabel(self.kw_voz_frame, text="Max Threads:").pack(side="left", padx=(0, 5))
        self.kw_max_threads_var = ctk.StringVar(value="10")
        ctk.CTkEntry(self.kw_voz_frame, textvariable=self.kw_max_threads_var, width=60).pack(side="left", padx=(0, 15))
        ctk.CTkLabel(self.kw_voz_frame, text="Max Pages:").pack(side="left", padx=(0, 5))
        self.kw_max_pages_var = ctk.StringVar(value="50")
        ctk.CTkEntry(self.kw_voz_frame, textvariable=self.kw_max_pages_var, width=60).pack(side="left", padx=(0, 15))

        # Threads params
        self.kw_threads_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        ctk.CTkLabel(self.kw_threads_frame, text="Max Posts:").pack(side="left", padx=(0, 5))
        self.kw_max_posts_var = ctk.StringVar(value="10")
        ctk.CTkEntry(self.kw_threads_frame, textvariable=self.kw_max_posts_var, width=60).pack(side="left", padx=(0, 15))
        ctk.CTkLabel(self.kw_threads_frame, text="Max Scroll:").pack(side="left", padx=(0, 5))
        self.kw_max_scroll_var = ctk.StringVar(value="30")
        ctk.CTkEntry(self.kw_threads_frame, textvariable=self.kw_max_scroll_var, width=60).pack(side="left", padx=(0, 15))

        # Show VOZ params by default
        self.kw_voz_frame.pack(side="left")

        # NLP toggles
        nlp_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        nlp_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(nlp_frame, text="Tiền Xử Lý:").pack(side="left", padx=(0, 10))

        self.kw_dec_var = ctk.BooleanVar(value=True)
        self.kw_fil_var = ctk.BooleanVar(value=True)
        self.kw_nor_var = ctk.BooleanVar(value=True)
        self.kw_seg_var = ctk.BooleanVar(value=True)

        self.kw_chk_dec = ctk.CTkCheckBox(nlp_frame, text="Decoder", variable=self.kw_dec_var)
        self.kw_chk_dec.pack(side="left", padx=5)
        self.kw_chk_fil = ctk.CTkCheckBox(nlp_frame, text="Filter", variable=self.kw_fil_var)
        self.kw_chk_fil.pack(side="left", padx=5)
        self.kw_chk_nor = ctk.CTkCheckBox(nlp_frame, text="Normalizer", variable=self.kw_nor_var)
        self.kw_chk_nor.pack(side="left", padx=5)
        self.kw_chk_seg = ctk.CTkCheckBox(nlp_frame, text="VnCoreNLP Segmentor", variable=self.kw_seg_var)
        self.kw_chk_seg.pack(side="left", padx=5)

        # Buttons
        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.grid(row=3, column=0, columnspan=4, pady=10)

        self.kw_run_button = ctk.CTkButton(btn_frame, text="Bắt Đầu", command=self.start_keyword_crawling)
        self.kw_run_button.pack(side="left", padx=10)
        self.kw_stop_button = ctk.CTkButton(btn_frame, text="Dừng", command=self.stop_keyword_crawling,
                                             state="disabled", fg_color="red", hover_color="darkred")
        self.kw_stop_button.pack(side="left", padx=10)

        # --- Main workspace: Log + Table + History ---
        work_frame = ctk.CTkFrame(tab)
        work_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        work_frame.grid_columnconfigure(0, weight=1)
        work_frame.grid_columnconfigure(1, weight=3)
        work_frame.grid_columnconfigure(2, weight=1)
        work_frame.grid_rowconfigure(0, weight=1)

        # Log textbox (left)
        self.kw_log_textbox = ctk.CTkTextbox(work_frame, corner_radius=5)
        self.kw_log_textbox.grid(row=0, column=0, padx=(5, 5), pady=5, sticky="nsew")
        self.kw_log_textbox.insert("0.0", "Hệ thống Keyword Crawler sẵn sàng.\n")
        self.kw_log_textbox.configure(state="disabled")

        # Data table (center)
        table_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
        table_frame.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        kw_columns = ("id", "text")
        self.kw_tree_data = ttk.Treeview(table_frame, columns=kw_columns, show="headings")
        self.kw_tree_data.heading("id", text="ID")
        self.kw_tree_data.heading("text", text="Nội dung bình luận")
        self.kw_tree_data.column("id", width=50, anchor="center")
        self.kw_tree_data.column("text", width=400, anchor="w")
        kw_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.kw_tree_data.yview)
        self.kw_tree_data.configure(yscrollcommand=kw_scroll.set)
        self.kw_tree_data.grid(row=0, column=0, sticky="nsew")
        kw_scroll.grid(row=0, column=1, sticky="ns")

        # History panel (right)
        history_frame = ctk.CTkFrame(work_frame)
        history_frame.grid(row=0, column=2, padx=(5, 5), pady=5, sticky="nsew")
        history_frame.grid_rowconfigure(1, weight=1)
        history_frame.grid_columnconfigure(0, weight=1)

        hist_header = ctk.CTkFrame(history_frame, fg_color="transparent")
        hist_header.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(hist_header, text="Lịch sử Keyword", font=("Arial", 13, "bold")).pack(side="left")
        ctk.CTkButton(hist_header, text="⟳", width=30, command=self.refresh_keyword_history).pack(side="right", padx=2)
        ctk.CTkButton(hist_header, text="Cào lại", width=60, command=self._reuse_history_keyword).pack(side="right", padx=2)

        hist_tree_frame = ctk.CTkFrame(history_frame, fg_color="transparent")
        hist_tree_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        hist_tree_frame.grid_rowconfigure(0, weight=1)
        hist_tree_frame.grid_columnconfigure(0, weight=1)

        self.kw_history_tree = ttk.Treeview(hist_tree_frame, columns=("platform", "keyword"), show="headings", height=10)
        self.kw_history_tree.heading("platform", text="Platform")
        self.kw_history_tree.heading("keyword", text="Keyword")
        self.kw_history_tree.column("platform", width=70, anchor="center")
        self.kw_history_tree.column("keyword", width=130, anchor="w")
        hist_scroll = ttk.Scrollbar(hist_tree_frame, orient="vertical", command=self.kw_history_tree.yview)
        self.kw_history_tree.configure(yscrollcommand=hist_scroll.set)
        self.kw_history_tree.grid(row=0, column=0, sticky="nsew")
        hist_scroll.grid(row=0, column=1, sticky="ns")

        # Status
        self.kw_status_label = ctk.CTkLabel(tab, text="Trạng thái: Đang chờ lệnh", text_color="gray")
        self.kw_status_label.grid(row=3, column=0, pady=5, sticky="w", padx=10)

        # State variables
        self.kw_is_running = False
        self.kw_stop_event = threading.Event()
        self.kw_extracted_data = []
        self.kw_active_crawler = None  # reference to close on stop

        # Initial history load
        self.refresh_keyword_history()

    def _on_kw_platform_change(self, platform):
        if platform == "VOZ":
            self.kw_threads_frame.pack_forget()
            self.kw_voz_frame.pack(side="left")
        else:
            self.kw_voz_frame.pack_forget()
            self.kw_threads_frame.pack(side="left")

    def refresh_keyword_history(self):
        for item in self.kw_history_tree.get_children():
            self.kw_history_tree.delete(item)
        try:
            history = load_keyword_history()
            for platform in ["voz", "threads"]:
                for kw in history.get(platform, []):
                    self.kw_history_tree.insert("", "end", values=(platform.upper(), kw))
        except Exception:
            pass

    def _reuse_history_keyword(self):
        selected = self.kw_history_tree.selection()
        if not selected:
            messagebox.showwarning("Nhắc nhở", "Hãy chọn 1 keyword từ lịch sử.")
            return
        item = self.kw_history_tree.item(selected[0])
        platform = item["values"][0]
        keyword = item["values"][1]
        self.kw_entry.delete(0, "end")
        self.kw_entry.insert(0, keyword)
        self.kw_platform_var.set(platform)
        self._on_kw_platform_change(platform)

    def kw_log_message(self, message):
        self.kw_log_textbox.configure(state="normal")
        self.kw_log_textbox.insert("end", f"{message}\n")
        self.kw_log_textbox.see("end")
        self.kw_log_textbox.configure(state="disabled")

    def kw_handle_new_data(self, batch):
        """Called from crawler thread – schedule GUI update."""
        self.after(0, self._kw_append_to_table, batch)
        self.kw_extracted_data.extend(batch)

    def _kw_append_to_table(self, batch):
        for item in batch:
            display_text = str(item.get("text", "")).replace("\n", "  ")
            self.kw_tree_data.insert("", "end", values=(item.get("id", ""), display_text))
        if len(self.kw_tree_data.get_children()) > 0:
            self.kw_tree_data.yview_moveto(1)
        wh_count = get_warehouse_count()
        self.kw_status_label.configure(
            text=f"Thu thập: {len(self.kw_extracted_data)} | Warehouse: {wh_count} dòng",
            text_color="green"
        )

    def _set_kw_gui_state(self, running):
        if running:
            self.kw_run_button.configure(state="disabled", text="Đang chạy...")
            self.kw_stop_button.configure(state="normal")
            self.kw_entry.configure(state="disabled")
            self.kw_platform_menu.configure(state="disabled")
            self.kw_chk_dec.configure(state="disabled")
            self.kw_chk_fil.configure(state="disabled")
            self.kw_chk_nor.configure(state="disabled")
            self.kw_chk_seg.configure(state="disabled")
            self.kw_is_running = True
        else:
            self.kw_run_button.configure(state="normal", text="Bắt Đầu")
            self.kw_stop_button.configure(state="disabled", text="Dừng")
            self.kw_entry.configure(state="normal")
            self.kw_platform_menu.configure(state="normal")
            self.kw_chk_dec.configure(state="normal")
            self.kw_chk_fil.configure(state="normal")
            self.kw_chk_nor.configure(state="normal")
            self.kw_chk_seg.configure(state="normal")
            self.kw_is_running = False
            self.refresh_file_list()
            self.refresh_keyword_history()

    def _keyword_crawl_thread(self, keyword, platform, u_dec, u_fil, u_nor, u_seg,
                               max_threads, max_pages, max_posts, max_scroll):
        crawler = None
        try:
            # Init NLP preprocessor
            preprocessor = None
            if u_dec or u_fil or u_nor or u_seg:
                self.after(0, self.kw_log_message, "Đang khởi tạo bộ tiền xử lý NLP...")
                preprocessor = VietnameseCommentPreprocessor()

            if platform == "VOZ":
                crawler = VOZCrawler(
                    keyword=keyword,
                    max_threads=max_threads,
                    max_pages=max_pages,
                    log_callback=lambda msg: self.after(0, self.kw_log_message, msg),
                    stop_event=self.kw_stop_event,
                    data_callback=self.kw_handle_new_data,
                    preprocessor=preprocessor,
                    use_decoder=u_dec,
                    use_filter=u_fil,
                    use_normalizer=u_nor,
                    use_segmentor=u_seg,
                )
            else:
                crawler = ThreadsCrawler(
                    keyword=keyword,
                    max_posts=max_posts,
                    max_scroll=max_scroll,
                    log_callback=lambda msg: self.after(0, self.kw_log_message, msg),
                    stop_event=self.kw_stop_event,
                    data_callback=self.kw_handle_new_data,
                    preprocessor=preprocessor,
                    use_decoder=u_dec,
                    use_filter=u_fil,
                    use_normalizer=u_nor,
                    use_segmentor=u_seg,
                )

            self.kw_active_crawler = crawler
            crawler.crawl_keyword(keyword)
            self.after(0, self.kw_log_message, f"--- HOÀN TẤT. Tổng {len(self.kw_extracted_data)} bình luận. ---")
        except Exception as e:
            self.after(0, self.kw_log_message, f"Lỗi không xác định: {str(e)}")
        finally:
            if crawler:
                try:
                    crawler.close()
                except Exception:
                    pass
            self.kw_active_crawler = None
            self.after(0, self._set_kw_gui_state, False)

    def start_keyword_crawling(self):
        if self.kw_is_running:
            return
        keyword = self.kw_entry.get().strip()
        if not keyword:
            messagebox.showwarning("Lỗi", "Vui lòng nhập keyword!")
            return

        platform = self.kw_platform_var.get()

        # Check if keyword already in history – warn but allow
        history = load_keyword_history()
        platform_key = platform.lower()
        kw_already_crawled = keyword in history.get(platform_key, [])

        u_dec = self.kw_dec_var.get()
        u_fil = self.kw_fil_var.get()
        u_nor = self.kw_nor_var.get()
        u_seg = self.kw_seg_var.get()

        max_threads = int(self.kw_max_threads_var.get() or 10)
        max_pages = int(self.kw_max_pages_var.get() or 50)
        max_posts = int(self.kw_max_posts_var.get() or 10)
        max_scroll = int(self.kw_max_scroll_var.get() or 30)

        self.kw_extracted_data = []
        self.kw_tree_data.delete(*self.kw_tree_data.get_children())
        self.kw_log_textbox.configure(state="normal")
        self.kw_log_textbox.delete("0.0", "end")
        self.kw_log_textbox.configure(state="disabled")

        self.kw_stop_event.clear()
        self._set_kw_gui_state(True)

        if kw_already_crawled:
            self.kw_log_message(f"⚠ Keyword '{keyword}' đã có trong lịch sử {platform}. Vẫn tiếp tục cào...")
        self.kw_log_message(f"Bắt đầu cào {platform} với keyword: {keyword}")

        thread = threading.Thread(
            target=self._keyword_crawl_thread,
            args=(keyword, platform, u_dec, u_fil, u_nor, u_seg,
                  max_threads, max_pages, max_posts, max_scroll),
            daemon=True,
        )
        thread.start()

    def stop_keyword_crawling(self):
        if self.kw_is_running:
            self.kw_log_message("Đang gửi lệnh yêu cầu dừng...")
            self.kw_stop_event.set()
            self.kw_stop_button.configure(state="disabled", text="Đang dừng...")
            # Also try to kill the browser directly for faster stop
            if self.kw_active_crawler:
                try:
                    self.kw_active_crawler.force_kill_driver()
                except Exception:
                    pass

    # ---------------------------------------------------------
    # TAB 3: WAREHOUSE MANAGER
    # ---------------------------------------------------------
    def _setup_warehouse_tab(self):
        tab = self.tab_warehouse
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # --- Header / toolbar ---
        header = ctk.CTkFrame(tab)
        header.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Row 1: search + stats
        row1 = ctk.CTkFrame(header, fg_color="transparent")
        row1.pack(fill="x", padx=5, pady=(5, 2))

        ctk.CTkButton(row1, text="⟳ Tải dữ liệu", width=110, command=self.wh_load_data).pack(side="left", padx=5)

        ctk.CTkLabel(row1, text="Tìm kiếm:").pack(side="left", padx=(15, 5))
        self.wh_search_var = ctk.StringVar()
        self.wh_search_entry = ctk.CTkEntry(row1, textvariable=self.wh_search_var, placeholder_text="Nhập text để lọc...", width=250)
        self.wh_search_entry.pack(side="left", padx=5)
        ctk.CTkButton(row1, text="Lọc", width=60, command=self.wh_filter_data).pack(side="left", padx=5)
        ctk.CTkButton(row1, text="Xóa lọc", width=70, command=self.wh_clear_filter).pack(side="left", padx=5)

        self.wh_stats_label = ctk.CTkLabel(row1, text="Warehouse: 0 dòng", text_color="gray")
        self.wh_stats_label.pack(side="right", padx=10)

        # Row 2: preprocessing options + actions
        row2 = ctk.CTkFrame(header, fg_color="transparent")
        row2.pack(fill="x", padx=5, pady=(2, 5))

        ctk.CTkLabel(row2, text="Tiền xử lý:").pack(side="left", padx=(5, 5))

        self.wh_dec_var = ctk.BooleanVar(value=True)
        self.wh_fil_var = ctk.BooleanVar(value=True)
        self.wh_nor_var = ctk.BooleanVar(value=True)
        self.wh_seg_var = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(row2, text="Decoder", variable=self.wh_dec_var).pack(side="left", padx=4)
        ctk.CTkCheckBox(row2, text="Filter", variable=self.wh_fil_var).pack(side="left", padx=4)
        ctk.CTkCheckBox(row2, text="Normalizer", variable=self.wh_nor_var).pack(side="left", padx=4)
        ctk.CTkCheckBox(row2, text="VnCoreNLP", variable=self.wh_seg_var).pack(side="left", padx=4)

        ctk.CTkButton(row2, text="▶ Chạy Preprocessing", width=150,
                       fg_color="#2563EB", hover_color="#1D4ED8",
                       command=self.wh_run_preprocessing).pack(side="left", padx=(15, 5))
        ctk.CTkButton(row2, text="Xóa trùng lặp", width=110,
                       fg_color="#7C3AED", hover_color="#6D28D9",
                       command=self.wh_remove_duplicates).pack(side="left", padx=5)
        ctk.CTkButton(row2, text="Xuất CSV", width=90,
                       fg_color="#059669", hover_color="#047857",
                       command=self.wh_export_csv).pack(side="left", padx=5)
        ctk.CTkButton(row2, text="Xóa dòng chọn", width=100,
                       fg_color="red", hover_color="darkred",
                       command=self.wh_delete_selected).pack(side="right", padx=5)

        # --- Data table ---
        tf = ctk.CTkFrame(tab)
        tf.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        tf.grid_columnconfigure(0, weight=1)
        tf.grid_rowconfigure(0, weight=1)

        self.wh_tree = ttk.Treeview(tf, columns=("id", "text"), show="headings", selectmode="extended")
        self.wh_tree.heading("id", text="ID")
        self.wh_tree.heading("text", text="Nội dung bình luận")
        self.wh_tree.column("id", width=60, anchor="center")
        self.wh_tree.column("text", width=800, anchor="w")

        wh_scroll = ttk.Scrollbar(tf, orient="vertical", command=self.wh_tree.yview)
        self.wh_tree.configure(yscrollcommand=wh_scroll.set)
        self.wh_tree.grid(row=0, column=0, sticky="nsew")
        wh_scroll.grid(row=0, column=1, sticky="ns")

        # Status bar
        self.wh_status_label = ctk.CTkLabel(tab, text="Trạng thái: Sẵn sàng", text_color="gray")
        self.wh_status_label.grid(row=2, column=0, pady=5, sticky="w", padx=10)

        # Internal data cache
        self._wh_all_rows = []  # list of {"id": int, "text": str}
        self._wh_is_processing = False

        # Initial load
        self.wh_load_data()

    def wh_load_data(self):
        """Load warehouse.csv into the table."""
        self._wh_all_rows = read_warehouse()
        self._wh_display_rows(self._wh_all_rows)
        self.wh_stats_label.configure(text=f"Warehouse: {len(self._wh_all_rows)} dòng")
        self.wh_status_label.configure(text=f"Đã tải {len(self._wh_all_rows)} dòng từ warehouse.csv", text_color="green")

    def _wh_display_rows(self, rows):
        """Populate the treeview with a list of row dicts."""
        self.wh_tree.delete(*self.wh_tree.get_children())
        for row in rows:
            display_text = str(row.get("text", "")).replace("\n", "  ")
            self.wh_tree.insert("", "end", values=(row.get("id", ""), display_text))

    def wh_filter_data(self):
        """Filter displayed rows by search term."""
        query = self.wh_search_var.get().strip().lower()
        if not query:
            self._wh_display_rows(self._wh_all_rows)
            return
        filtered = [r for r in self._wh_all_rows if query in r.get("text", "").lower()]
        self._wh_display_rows(filtered)
        self.wh_status_label.configure(text=f"Hiển thị {len(filtered)}/{len(self._wh_all_rows)} dòng (lọc: '{query}')", text_color="blue")

    def wh_clear_filter(self):
        """Clear search filter and show all data."""
        self.wh_search_var.set("")
        self._wh_display_rows(self._wh_all_rows)
        self.wh_status_label.configure(text=f"Hiển thị tất cả {len(self._wh_all_rows)} dòng", text_color="green")

    def wh_delete_selected(self):
        """Delete selected rows from warehouse."""
        selected = self.wh_tree.selection()
        if not selected:
            messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 dòng để xóa.")
            return
        ids_to_delete = set()
        for item in selected:
            vals = self.wh_tree.item(item)["values"]
            if vals:
                ids_to_delete.add(int(vals[0]))
        count_before = len(self._wh_all_rows)
        if not messagebox.askyesno("Xác nhận", f"Xóa {len(ids_to_delete)} dòng khỏi warehouse.csv?"):
            return
        self._wh_all_rows = [r for r in self._wh_all_rows if r["id"] not in ids_to_delete]
        # Re-assign IDs
        for i, row in enumerate(self._wh_all_rows):
            row["id"] = i + 1
        overwrite_warehouse(self._wh_all_rows)
        self._wh_display_rows(self._wh_all_rows)
        removed = count_before - len(self._wh_all_rows)
        self.wh_stats_label.configure(text=f"Warehouse: {len(self._wh_all_rows)} dòng")
        self.wh_status_label.configure(text=f"Đã xóa {removed} dòng.", text_color="orange")

    def wh_remove_duplicates(self):
        """Remove duplicate texts from warehouse."""
        seen = set()
        unique = []
        for row in self._wh_all_rows:
            text = row.get("text", "").strip()
            if text and text not in seen:
                seen.add(text)
                unique.append(row)
        removed = len(self._wh_all_rows) - len(unique)
        if removed == 0:
            messagebox.showinfo("Thông báo", "Không có dòng trùng lặp nào.")
            return
        # Re-assign IDs
        for i, row in enumerate(unique):
            row["id"] = i + 1
        self._wh_all_rows = unique
        overwrite_warehouse(self._wh_all_rows)
        self._wh_display_rows(self._wh_all_rows)
        self.wh_stats_label.configure(text=f"Warehouse: {len(self._wh_all_rows)} dòng")
        self.wh_status_label.configure(text=f"Đã xóa {removed} dòng trùng lặp.", text_color="green")

    def wh_export_csv(self):
        """Export current warehouse data to a timestamped CSV."""
        if not self._wh_all_rows:
            messagebox.showwarning("Nhắc nhở", "Warehouse trống, không có gì để xuất.")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(os.getcwd(), f"warehouse_export_{timestamp}.csv")
        try:
            with open(export_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text"])
                writer.writeheader()
                writer.writerows(self._wh_all_rows)
            messagebox.showinfo("Thành công", f"Đã xuất {len(self._wh_all_rows)} dòng ra:\n{os.path.basename(export_path)}")
            self.refresh_file_list()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất CSV: {e}")

    def wh_run_preprocessing(self):
        """Run NLP preprocessing on all warehouse rows in a background thread."""
        if self._wh_is_processing:
            messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
            return
        if not self._wh_all_rows:
            messagebox.showwarning("Nhắc nhở", "Warehouse trống.")
            return

        u_dec = self.wh_dec_var.get()
        u_fil = self.wh_fil_var.get()
        u_nor = self.wh_nor_var.get()
        u_seg = self.wh_seg_var.get()

        if not (u_dec or u_fil or u_nor or u_seg):
            messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 bước tiền xử lý.")
            return

        self._wh_is_processing = True
        self.wh_status_label.configure(text="Đang khởi tạo NLP pipeline... Vui lòng đợi.", text_color="orange")

        def _process():
            try:
                preprocessor = VietnameseCommentPreprocessor()
                total = len(self._wh_all_rows)
                kept = []
                removed_count = 0

                for idx, row in enumerate(self._wh_all_rows):
                    text = row.get("text", "")
                    if not text.strip():
                        removed_count += 1
                        continue
                    result = preprocessor.process_comment(
                        text, use_decoder=u_dec, use_filter=u_fil,
                        use_normalizer=u_nor, use_segmentor=u_seg,
                    )
                    if result["is_valid"]:
                        kept.append({"id": len(kept) + 1, "text": result["cleaned_text"]})
                    else:
                        removed_count += 1

                    if (idx + 1) % 500 == 0:
                        self.after(0, lambda i=idx+1: self.wh_status_label.configure(
                            text=f"Đang xử lý... {i}/{total}", text_color="orange"))

                self._wh_all_rows = kept
                overwrite_warehouse(kept)
                self.after(0, self._wh_display_rows, kept)
                self.after(0, lambda: self.wh_stats_label.configure(text=f"Warehouse: {len(kept)} dòng"))
                self.after(0, lambda: self.wh_status_label.configure(
                    text=f"Hoàn tất! Giữ {len(kept)}/{total} dòng (loại {removed_count} dòng).",
                    text_color="green"))
            except Exception as e:
                self.after(0, lambda: self.wh_status_label.configure(
                    text=f"Lỗi: {str(e)}", text_color="red"))
            finally:
                self._wh_is_processing = False

        threading.Thread(target=_process, daemon=True).start()

    # ---------------------------------------------------------
    # TAB 4: FILE MANAGER
    # ---------------------------------------------------------
    def _setup_file_manager_tab(self):
        tab = self.tab_files
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkButton(header, text="Làm mới danh sách", command=self.refresh_file_list).pack(side="left", padx=5)
        ctk.CTkButton(header, text="Mở thư mục hiện tại", command=self.open_current_folder).pack(side="left", padx=5)
        ctk.CTkButton(header, text="Xóa File Chọn", command=self.delete_selected_file, fg_color="red").pack(side="right", padx=5)

        # Treeview for files
        tf = ctk.CTkFrame(tab)
        tf.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        tf.grid_columnconfigure(0, weight=1)
        tf.grid_rowconfigure(0, weight=1)

        self.tree_files = ttk.Treeview(tf, columns=("filename", "size", "mtime"), show="headings")
        self.tree_files.heading("filename", text="Tên File")
        self.tree_files.heading("size", text="Kích thước (KB)")
        self.tree_files.heading("mtime", text="Thời gian sửa đổi")
        self.tree_files.column("filename", width=300)
        self.tree_files.column("size", width=100, anchor="e")
        self.tree_files.column("mtime", width=150)
        
        scrollbar_f = ttk.Scrollbar(tf, orient="vertical", command=self.tree_files.yview)
        self.tree_files.configure(yscrollcommand=scrollbar_f.set)
        
        self.tree_files.grid(row=0, column=0, sticky="nsew")
        scrollbar_f.grid(row=0, column=1, sticky="ns")
        
        # Initial load
        self.refresh_file_list()

    def refresh_file_list(self):
        for item in self.tree_files.get_children():
            self.tree_files.delete(item)
            
        csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
        
        for f in sorted(csv_files, key=os.path.getmtime, reverse=True):
            name = os.path.basename(f)
            size_kb = f"{os.path.getsize(f) / 1024:.1f}"
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
            self.tree_files.insert("", "end", values=(name, size_kb, mtime))

    def delete_selected_file(self):
        selected = self.tree_files.selection()
        if not selected:
            messagebox.showwarning("Nhắc nhở", "Bạn chưa chọn file cần xóa")
            return
            
        item = self.tree_files.item(selected[0])
        filename = item['values'][0]
        full_path = os.path.join(os.getcwd(), filename)
        
        if messagebox.askyesno("Xác nhận", f"Bạn có chắc chắn muốn xóa vĩnh viễn file:\n{filename}?"):
            try:
                os.remove(full_path)
                self.refresh_file_list()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xóa file: {e}")

    def open_current_folder(self):
        folder = os.getcwd()
        try:
            os.startfile(folder)
        except AttributeError:
            os.system(f'explorer "{folder}"')

    # ---------------------------------------------------------
    # TAB 4: CONFIG MANAGER
    # ---------------------------------------------------------
    def _setup_config_tab(self):
        tab = self.tab_config
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header for selection
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(header, text="Đang mở file:").pack(side="left", padx=5)
        self.config_vars = ["abbreviations.json", "emoji_vi.json", "profanity_list.json", "nlp_pipeline/config.py"]
        self.config_dropdown = ctk.CTkOptionMenu(header, values=self.config_vars, command=self.on_config_file_change)
        self.config_dropdown.pack(side="left", padx=5)

        # Workspace Container
        self.config_workspace = ctk.CTkFrame(tab)
        self.config_workspace.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.config_workspace.grid_columnconfigure(0, weight=1)
        self.config_workspace.grid_rowconfigure(0, weight=1)
        
        # --- JSON Mapping Editor Frame ---
        self.json_editor_frame = ctk.CTkFrame(self.config_workspace, fg_color="transparent")
        self.json_editor_frame.grid_columnconfigure(0, weight=1)
        self.json_editor_frame.grid_rowconfigure(1, weight=1)
        
        # Top controls for JSON
        jf_top = ctk.CTkFrame(self.json_editor_frame, fg_color="transparent")
        jf_top.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ctk.CTkLabel(jf_top, text="Từ khóa:").pack(side="left", padx=2)
        self.entry_key = ctk.CTkEntry(jf_top, width=150)
        self.entry_key.pack(side="left", padx=2)
        
        ctk.CTkLabel(jf_top, text="Giá trị thay thế:").pack(side="left", padx=2)
        self.entry_val = ctk.CTkEntry(jf_top, width=200)
        self.entry_val.pack(side="left", padx=2)
        
        ctk.CTkButton(jf_top, text="Thêm/Sửa", width=80, command=self.add_or_update_json_entry).pack(side="left", padx=5)
        ctk.CTkButton(jf_top, text="Xóa", width=60, fg_color="red", hover_color="darkred", command=self.delete_json_entry).pack(side="left", padx=5)
        ctk.CTkButton(jf_top, text="Lưu Cấu Hình (Disk)", width=120, fg_color="green", hover_color="darkgreen", command=self.save_json_file).pack(side="right", padx=5)
        
        # Treeview for JSON
        tr_frame = ctk.CTkFrame(self.json_editor_frame)
        tr_frame.grid(row=1, column=0, sticky="nsew")
        tr_frame.grid_columnconfigure(0, weight=1)
        tr_frame.grid_rowconfigure(0, weight=1)
        
        self.tree_json = ttk.Treeview(tr_frame, columns=("key", "val"), show="headings")
        self.tree_json.heading("key", text="Từ Khóa (Key)")
        self.tree_json.heading("val", text="Giá Trị Thay Thế (Value)")
        self.tree_json.column("key", width=200, anchor="w")
        self.tree_json.column("val", width=400, anchor="w")
        
        scroll_j = ttk.Scrollbar(tr_frame, orient="vertical", command=self.tree_json.yview)
        self.tree_json.configure(yscrollcommand=scroll_j.set)
        self.tree_json.grid(row=0, column=0, sticky="nsew")
        scroll_j.grid(row=0, column=1, sticky="ns")
        
        self.tree_json.bind('<<TreeviewSelect>>', self.on_json_tree_select)
        
        # Current active JSON data
        self.current_json_data = {}
        
        # --- Python Config Editor Frame ---
        self.py_editor_frame = ctk.CTkFrame(self.config_workspace, fg_color="transparent")
        self.py_editor_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.py_editor_frame, text="Quản lý biến trong config.py", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
        
        # Dynamically build entry fields for known python configs
        self.py_config_entries = {}
        target_keys = ["MIN_CHAR_LENGTH", "MIN_VIETNAMESE_RATIO", "MAX_SPAM_REPEAT", "MAX_REPEAT_CHARS", "MAX_REPEAT_PUNCTUATION", "MAX_REPEAT_ICON_CHARS"]
        
        row_idx = 1
        for k in target_keys:
            ctk.CTkLabel(self.py_editor_frame, text=k + ":").grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
            e = ctk.CTkEntry(self.py_editor_frame, width=200)
            e.grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
            self.py_config_entries[k] = e
            row_idx += 1
            
        ctk.CTkButton(self.py_editor_frame, text="Lưu config.py", fg_color="green", hover_color="darkgreen", command=self.save_py_config).grid(row=row_idx, column=0, columnspan=2, pady=20)
        
        # Load initial
        self.on_config_file_change(self.config_vars[0])

    def on_config_file_change(self, filename):
        if filename.endswith(".json"):
            self.py_editor_frame.grid_forget()
            self.json_editor_frame.grid(row=0, column=0, sticky="nsew")
            self.load_json_config(filename)
        else:
            self.json_editor_frame.grid_forget()
            self.py_editor_frame.grid(row=0, column=0, sticky="nsew")
            self.load_py_config(filename)

    def _get_config_path(self, filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if "nlp_pipeline" in filename:
            # The dropdown passes "nlp_pipeline/config.py"
            return os.path.join(base_dir, "nlp_pipeline", "config.py")
        else:
            # json map path is in mappings/
            return os.path.join(base_dir, "mappings", filename)

    # --- JSON Helpers ---
    def load_json_config(self, filename):
        path = self._get_config_path(filename)
        self.current_json_data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.current_json_data = data
                    elif isinstance(data, list):
                        # For profanity list -> turn into dict mapping to True
                        for item in data:
                            self.current_json_data[str(item)] = ""
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc JSON: {e}")
        
        self.refresh_json_tree()
        self.entry_key.delete(0, 'end')
        self.entry_val.delete(0, 'end')

    def refresh_json_tree(self):
        for item in self.tree_json.get_children():
            self.tree_json.delete(item)
        for k, v in self.current_json_data.items():
            self.tree_json.insert("", "end", values=(k, v))

    def on_json_tree_select(self, event):
        selected = self.tree_json.selection()
        if selected:
            item = self.tree_json.item(selected[0])
            self.entry_key.delete(0, 'end')
            self.entry_key.insert(0, item['values'][0])
            self.entry_val.delete(0, 'end')
            # Handle empty values safely
            val = item['values'][1] if len(item['values']) > 1 else ""
            self.entry_val.insert(0, val)

    def add_or_update_json_entry(self):
        k = self.entry_key.get().strip()
        v = self.entry_val.get().strip()
        if not k:
            messagebox.showwarning("Nhắc nhở", "Từ khóa không được để trống")
            return
        self.current_json_data[k] = v
        self.refresh_json_tree()
        self.entry_key.delete(0, 'end')
        self.entry_val.delete(0, 'end')

    def delete_json_entry(self):
        selected = self.tree_json.selection()
        if not selected:
            messagebox.showwarning("Nhắc nhở", "Hãy chọn 1 dòng để xóa")
            return
        item = self.tree_json.item(selected[0])
        k = str(item['values'][0])
        if k in self.current_json_data:
            del self.current_json_data[k]
            self.refresh_json_tree()
            self.entry_key.delete(0, 'end')
            self.entry_val.delete(0, 'end')

    def save_json_file(self):
        filename = self.config_dropdown.get()
        path = self._get_config_path(filename)
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                # If profanity list, we might want to save exactly as list
                if filename == "profanity_list.json":
                    res = list(self.current_json_data.keys())
                    json.dump(res, f, ensure_ascii=False, indent=4)
                else:
                    json.dump(self.current_json_data, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Thành công", f"Đã cập nhật file {filename}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể ghi file: {str(e)}")

    # --- Python Config Helpers ---
    def load_py_config(self, filename):
        path = self._get_config_path(filename)
        if not os.path.exists(path):
            messagebox.showwarning("Lỗi", f"Không tìm thấy file {path}")
            return
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Parse logic
        for k, entry in self.py_config_entries.items():
            pattern = rf"^[ \t]*{k}[ \t]*=[ \t]*([0-9\.]+)"
            match = re.search(pattern, content, re.MULTILINE)
            entry.delete(0, 'end')
            if match:
                entry.insert(0, match.group(1))

    def save_py_config(self):
        filename = self.config_dropdown.get()
        path = self._get_config_path(filename)
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                
            for k, entry in self.py_config_entries.items():
                val = entry.get().strip()
                if not val: continue
                # float or int parsing just to validate visually
                pattern = rf"^([ \t]*{k}[ \t]*=)[ \t]*([0-9\.]+)"
                # Replace exact definition
                if re.search(pattern, content, re.MULTILINE):
                    content = re.sub(pattern, rf"\g<1> {val}", content, flags=re.MULTILINE)
                    
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                
            messagebox.showinfo("Thành công", "Cập nhật config.py thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu config.py: {e}")

    # ---------------------------------------------------------
    # CRAWLER EXECUTION LOGIC
    # ---------------------------------------------------------
    def log_message(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"{message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def handle_new_data(self, batch):
        self.after(0, self._append_to_table, batch)
        if self.current_output_file:
            try:
                file_exists = os.path.exists(self.current_output_file)
                with open(self.current_output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['id', 'text'])
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(batch)
            except Exception as e:
                self.after(0, self.log_message, f"Lỗi ghi CSV: {e}")
        # Also append to warehouse.csv
        try:
            append_to_warehouse(batch)
        except Exception as e:
            self.after(0, self.log_message, f"Lỗi ghi warehouse: {e}")
        self.extracted_data.extend(batch)

    def _append_to_table(self, batch):
        for item in batch:
            display_text = item['text'].replace('\n', '  ')
            self.tree_data.insert("", "end", values=(item['id'], display_text))
        
        if len(self.tree_data.get_children()) > 0:
            self.tree_data.yview_moveto(1)
            
        self.status_label.configure(text=f"Đã thu thập & lọc: {len(self.extracted_data)} bình luận", text_color="green")

    def set_gui_state(self, running):
        if running:
            self.run_button.configure(state="disabled", text="Đang chạy...")
            self.stop_button.configure(state="normal")
            self.url_entry.configure(state="disabled")
            self.chk_dec.configure(state="disabled")
            self.chk_fil.configure(state="disabled")
            self.chk_nor.configure(state="disabled")
            self.chk_seg.configure(state="disabled")
            self.is_running = True
        else:
            self.run_button.configure(state="normal", text="Bắt Đầu")
            self.stop_button.configure(state="disabled")
            self.url_entry.configure(state="normal")
            self.chk_dec.configure(state="normal")
            self.chk_fil.configure(state="normal")
            self.chk_nor.configure(state="normal")
            self.chk_seg.configure(state="normal")
            self.is_running = False
            self.refresh_file_list()

    def _crawl_thread(self, url, headless, u_dec, u_fil, u_nor, u_seg):
        try:
            extract_comments_stream(
                url_input=url, 
                headless=headless, 
                use_decoder=u_dec,
                use_filter=u_fil,
                use_normalizer=u_nor,
                use_segmentor=u_seg,
                log_callback=lambda msg: self.after(0, self.log_message, msg),
                data_callback=self.handle_new_data,
                stop_event=self.stop_event
            )
            self.after(0, self.log_message, f"--- HOÀN TẤT. Đã lưu tổng {len(self.extracted_data)} mục. ---")
        except Exception as e:
            self.after(0, self.log_message, f"Lỗi không xác định: {str(e)}")
        finally:
            self.after(0, self.set_gui_state, False)

    def start_crawling(self):
        if self.is_running: return
            
        url_input = self.url_entry.get().strip()
        if not url_input:
            messagebox.showwarning("Lỗi", "Vui lòng nhập ít nhất một URL!")
            return

        headless = self.headless_var.get()
        u_dec = self.use_decoder_var.get()
        u_fil = self.use_filter_var.get()
        u_nor = self.use_normalizer_var.get()
        u_seg = self.use_segmentor_var.get()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_output_file = os.path.join(os.getcwd(), f"comments_{timestamp}.csv")
        
        self.extracted_data = []
        self.tree_data.delete(*self.tree_data.get_children())
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("0.0", "end")
        self.log_textbox.configure(state="disabled")
        
        self.stop_event.clear()
        self.set_gui_state(True)
        self.log_message(f"Tạo file lưu trữ: {self.current_output_file}")
        
        thread = threading.Thread(target=self._crawl_thread, 
                                  args=(url_input, headless, u_dec, u_fil, u_nor, u_seg), 
                                  daemon=True)
        thread.start()

    def stop_crawling(self):
        if self.is_running:
            self.log_message("Đang gửi lệnh yêu cầu dừng (vui lòng chờ vài giây)...")
            self.stop_event.set()
            self.stop_button.configure(state="disabled", text="Đang dừng...")

if __name__ == "__main__":
    app = App()
    app.mainloop()
