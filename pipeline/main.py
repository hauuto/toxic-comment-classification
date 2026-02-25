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
from google_drive import upload_warehouse, download_warehouse
from lmstudio_classifier import LMStudioClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Data manager")
        self.geometry("1100x800")
        self.minsize(950, 650)

        # Background threading variables
        self.extracted_data = []
        self.is_running = False
        self.stop_event = threading.Event()

        # Create Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_crawler = self.tabview.add("Facebook Crawler")
        self.tab_keyword = self.tabview.add("Keyword Crawler")
        self.tab_warehouse = self.tabview.add("Warehouse Manager")
        self.tab_labeling = self.tabview.add("Auto Labeling")
        self.tab_label_mgr = self.tabview.add("Label Manager")
        self.tab_files = self.tabview.add("File Manager")
        self.tab_config = self.tabview.add("Config Manager")

        self._setup_crawler_tab()
        self._setup_keyword_crawler_tab()
        self._setup_warehouse_tab()
        self._setup_labeling_tab()
        self._setup_label_manager_tab()
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
        self.kw_entry = ctk.CTkEntry(input_frame, placeholder_text="Nhập từ khóa (nhiều keyword cách nhau bằng dấu ;)")
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

    def _keyword_crawl_thread(self, keywords, platform, u_dec, u_fil, u_nor, u_seg,
                               max_threads, max_pages, max_posts, max_scroll):
        crawler = None
        try:
            # Init NLP preprocessor once for all keywords
            preprocessor = None
            if u_dec or u_fil or u_nor or u_seg:
                self.after(0, self.kw_log_message, "Đang khởi tạo bộ tiền xử lý NLP...")
                preprocessor = VietnameseCommentPreprocessor()

            log_cb = lambda msg: self.after(0, self.kw_log_message, msg)

            for kw_idx, keyword in enumerate(keywords, start=1):
                if self.kw_stop_event.is_set():
                    self.after(0, self.kw_log_message, "Đã nhận lệnh DỪNG.")
                    break

                self.after(0, self.kw_log_message,
                           f"\n{'='*50}\n[{kw_idx}/{len(keywords)}] Keyword: {keyword}\n{'='*50}")

                # Close previous crawler before creating a new one
                if crawler:
                    try: crawler.close()
                    except Exception: pass

                if platform == "VOZ":
                    crawler = VOZCrawler(
                        keyword=keyword,
                        max_threads=max_threads,
                        max_pages=max_pages,
                        log_callback=log_cb,
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
                        log_callback=log_cb,
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

            self.after(0, self.kw_log_message,
                       f"\n--- HOÀN TẤT TẤT CẢ {len(keywords)} KEYWORD. Tổng {len(self.kw_extracted_data)} bình luận. ---")
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
        raw_input = self.kw_entry.get().strip()
        if not raw_input:
            messagebox.showwarning("Lỗi", "Vui lòng nhập keyword!")
            return

        # Parse multiple keywords separated by ;
        keywords = [kw.strip() for kw in raw_input.split(";") if kw.strip()]
        if not keywords:
            messagebox.showwarning("Lỗi", "Vui lòng nhập ít nhất một keyword hợp lệ!")
            return

        platform = self.kw_platform_var.get()

        # Check which keywords already in history – warn but allow
        history = load_keyword_history()
        platform_key = platform.lower()
        already_crawled = [kw for kw in keywords if kw in history.get(platform_key, [])]

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

        if already_crawled:
            self.kw_log_message(f"⚠ Các keyword đã có trong lịch sử {platform}: {', '.join(already_crawled)}. Vẫn tiếp tục cào...")
        self.kw_log_message(f"Bắt đầu cào {platform} với {len(keywords)} keyword: {'; '.join(keywords)}")

        thread = threading.Thread(
            target=self._keyword_crawl_thread,
            args=(keywords, platform, u_dec, u_fil, u_nor, u_seg,
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

        # Row 3: Google Drive sync
        row3 = ctk.CTkFrame(header, fg_color="transparent")
        row3.pack(fill="x", padx=5, pady=(2, 5))

        ctk.CTkLabel(row3, text="☁ Google Drive:", font=("Arial", 13, "bold")).pack(side="left", padx=(5, 10))
        ctk.CTkButton(row3, text="⬆ Upload lên Drive", width=150,
                       fg_color="#EA580C", hover_color="#C2410C",
                       command=self.wh_upload_drive).pack(side="left", padx=5)
        ctk.CTkButton(row3, text="⬇ Tải từ Drive", width=150,
                       fg_color="#0284C7", hover_color="#0369A1",
                       command=self.wh_download_drive).pack(side="left", padx=5)

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

    def wh_upload_drive(self):
        """Upload warehouse.csv to Google Drive in background thread."""
        if self._wh_is_processing:
            messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
            return
        if not self._wh_all_rows:
            messagebox.showwarning("Nhắc nhở", "Warehouse trống, không có gì để upload.")
            return

        self._wh_is_processing = True
        self.wh_status_label.configure(text="⬆ Đang upload warehouse.csv lên Google Drive...", text_color="orange")

        def _upload():
            try:
                def _log(msg):
                    self.after(0, lambda: self.wh_status_label.configure(text=msg, text_color="orange"))

                upload_warehouse(log_callback=_log)
                self.after(0, lambda: self.wh_status_label.configure(
                    text=f"✓ Đã upload warehouse.csv lên Google Drive ({len(self._wh_all_rows)} dòng).",
                    text_color="green"))
            except FileNotFoundError as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                self.after(0, lambda: self.wh_status_label.configure(
                    text="✗ Upload thất bại: thiếu credentials.", text_color="red"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi Upload", f"Không thể upload lên Drive:\n{str(e)}"))
                self.after(0, lambda: self.wh_status_label.configure(
                    text=f"✗ Upload thất bại: {str(e)[:80]}", text_color="red"))
            finally:
                self._wh_is_processing = False

        threading.Thread(target=_upload, daemon=True).start()

    def wh_download_drive(self):
        """Download warehouse.csv from Google Drive in background thread."""
        if self._wh_is_processing:
            messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
            return

        if self._wh_all_rows:
            if not messagebox.askyesno(
                "Xác nhận",
                f"Warehouse hiện có {len(self._wh_all_rows)} dòng.\n"
                "Tải từ Drive sẽ GHI ĐÈ toàn bộ dữ liệu local.\n\n"
                "Bạn có muốn tiếp tục?"
            ):
                return

        self._wh_is_processing = True
        self.wh_status_label.configure(text="⬇ Đang tải warehouse.csv từ Google Drive...", text_color="orange")

        def _download():
            try:
                def _log(msg):
                    self.after(0, lambda: self.wh_status_label.configure(text=msg, text_color="orange"))

                success = download_warehouse(log_callback=_log)
                if success:
                    self.after(0, self.wh_load_data)
                    self.after(0, lambda: self.wh_status_label.configure(
                        text="✓ Đã tải warehouse.csv từ Google Drive và cập nhật.",
                        text_color="green"))
                else:
                    self.after(0, lambda: self.wh_status_label.configure(
                        text="✗ Không tìm thấy warehouse.csv trên Google Drive.",
                        text_color="red"))
            except FileNotFoundError as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                self.after(0, lambda: self.wh_status_label.configure(
                    text="✗ Download thất bại: thiếu credentials.", text_color="red"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi Download", f"Không thể tải từ Drive:\n{str(e)}"))
                self.after(0, lambda: self.wh_status_label.configure(
                    text=f"✗ Download thất bại: {str(e)[:80]}", text_color="red"))
            finally:
                self._wh_is_processing = False

        threading.Thread(target=_download, daemon=True).start()

    # ---------------------------------------------------------
    # TAB: AUTO LABELING (LM Studio)
    # ---------------------------------------------------------
    def _get_labeled_data_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_data.csv")

    def _setup_labeling_tab(self):
        tab = self.tab_labeling
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        conn_frame = ctk.CTkFrame(tab)
        conn_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        conn_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(conn_frame, text="LM Studio Endpoint:").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.lbl_endpoint_var = ctk.StringVar(value="http://localhost:1234")
        self.lbl_endpoint_entry = ctk.CTkEntry(conn_frame, textvariable=self.lbl_endpoint_var, width=350)
        self.lbl_endpoint_entry.grid(row=0, column=1, padx=5, pady=8, sticky="w")

        ctk.CTkLabel(conn_frame, text="Model:").grid(row=0, column=2, padx=(15, 5), pady=8, sticky="w")
        self.lbl_model_var = ctk.StringVar(value="")
        self.lbl_model_entry = ctk.CTkEntry(conn_frame, textvariable=self.lbl_model_var, width=250,
                                             placeholder_text="(để trống = model đang load)")
        self.lbl_model_entry.grid(row=0, column=3, padx=5, pady=8, sticky="w")

        self.lbl_test_btn = ctk.CTkButton(conn_frame, text="🔌 Test Connection", width=140,
                                           command=self._lbl_test_connection)
        self.lbl_test_btn.grid(row=0, column=4, padx=10, pady=8)

        ctrl_frame = ctk.CTkFrame(tab)
        ctrl_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(ctrl_frame, text="Batch size:").pack(side="left", padx=(10, 5))
        self.lbl_batch_var = ctk.StringVar(value="5")
        ctk.CTkEntry(ctrl_frame, textvariable=self.lbl_batch_var, width=60).pack(side="left", padx=(0, 15))

        self.lbl_start_btn = ctk.CTkButton(ctrl_frame, text="▶ Bắt Đầu Gán Nhãn", width=160,
                                            fg_color="#2563EB", hover_color="#1D4ED8",
                                            command=self._lbl_start_labeling)
        self.lbl_start_btn.pack(side="left", padx=5)

        self.lbl_stop_btn = ctk.CTkButton(ctrl_frame, text="⏹ Dừng", width=80,
                                           fg_color="red", hover_color="darkred",
                                           state="disabled", command=self._lbl_stop_labeling)
        self.lbl_stop_btn.pack(side="left", padx=5)

        self.lbl_progress_var = ctk.DoubleVar(value=0.0)
        self.lbl_progress = ctk.CTkProgressBar(ctrl_frame, variable=self.lbl_progress_var, width=250)
        self.lbl_progress.pack(side="left", padx=(15, 5))
        self.lbl_progress.set(0)

        self.lbl_progress_text = ctk.CTkLabel(ctrl_frame, text="0 / 0", text_color="gray")
        self.lbl_progress_text.pack(side="left", padx=5)

        work_frame = ctk.CTkFrame(tab)
        work_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        work_frame.grid_columnconfigure(1, weight=3)
        work_frame.grid_columnconfigure(0, weight=1)
        work_frame.grid_rowconfigure(0, weight=1)

        self.lbl_log_textbox = ctk.CTkTextbox(work_frame, corner_radius=5)
        self.lbl_log_textbox.grid(row=0, column=0, padx=(5, 5), pady=5, sticky="nsew")
        self.lbl_log_textbox.insert("0.0", "Sẵn sàng. Hãy kiểm tra kết nối LM Studio trước khi bắt đầu.\n")
        self.lbl_log_textbox.configure(state="disabled")

        table_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
        table_frame.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.lbl_tree = ttk.Treeview(table_frame, columns=("id", "text", "label"), show="headings")
        self.lbl_tree.heading("id", text="ID")
        self.lbl_tree.heading("text", text="Nội dung bình luận")
        self.lbl_tree.heading("label", text="Nhãn")
        self.lbl_tree.column("id", width=50, anchor="center")
        self.lbl_tree.column("text", width=500, anchor="w")
        self.lbl_tree.column("label", width=100, anchor="center")

        lbl_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.lbl_tree.yview)
        self.lbl_tree.configure(yscrollcommand=lbl_scroll.set)
        self.lbl_tree.grid(row=0, column=0, sticky="nsew")
        lbl_scroll.grid(row=0, column=1, sticky="ns")

        self.lbl_status_label = ctk.CTkLabel(tab, text="Trạng thái: Đang chờ lệnh", text_color="gray")
        self.lbl_status_label.grid(row=3, column=0, pady=5, sticky="w", padx=10)

        self._lbl_is_running = False
        self._lbl_stop_event = threading.Event()

    def _lbl_log(self, msg):
        self.lbl_log_textbox.configure(state="normal")
        self.lbl_log_textbox.insert("end", f"{msg}\n")
        self.lbl_log_textbox.see("end")
        self.lbl_log_textbox.configure(state="disabled")

    def _lbl_test_connection(self):
        base_url = self.lbl_endpoint_var.get().strip()
        if not base_url:
            messagebox.showwarning("Lỗi", "Vui lòng nhập endpoint!")
            return
        self._lbl_log(f"🔌 Đang kiểm tra kết nối tới {base_url} ...")
        self.lbl_test_btn.configure(state="disabled", text="Đang kiểm tra...")

        def _test():
            result = LMStudioClassifier.test_connection(base_url)
            if result["ok"]:
                models_str = ", ".join(result["models"]) if result["models"] else "(không có model nào)"
                self.after(0, self._lbl_log, f"✅ Kết nối thành công! Models: {models_str}")
                self.after(0, lambda: self.lbl_status_label.configure(
                    text=f"✅ LM Studio đang chạy — {len(result['models'])} model(s)", text_color="green"))
                if len(result["models"]) == 1:
                    self.after(0, lambda: self.lbl_model_var.set(result["models"][0]))
            else:
                self.after(0, self._lbl_log, f"❌ Lỗi: {result['error']}")
                self.after(0, lambda: self.lbl_status_label.configure(
                    text="❌ Không thể kết nối LM Studio", text_color="red"))
            self.after(0, lambda: self.lbl_test_btn.configure(state="normal", text="🔌 Test Connection"))
        threading.Thread(target=_test, daemon=True).start()

    def _lbl_start_labeling(self):
        if self._lbl_is_running:
            return
        base_url = self.lbl_endpoint_var.get().strip()
        if not base_url:
            messagebox.showwarning("Lỗi", "Vui lòng nhập endpoint!")
            return
        rows = read_warehouse()
        if not rows:
            messagebox.showwarning("Lỗi", "Warehouse trống! Hãy crawl dữ liệu trước.")
            return
        try:
            batch_size = max(1, min(int(self.lbl_batch_var.get()), 20))
        except ValueError:
            batch_size = 5
        model_name = self.lbl_model_var.get().strip()

        self.lbl_tree.delete(*self.lbl_tree.get_children())
        self.lbl_log_textbox.configure(state="normal")
        self.lbl_log_textbox.delete("0.0", "end")
        self.lbl_log_textbox.configure(state="disabled")
        self.lbl_progress.set(0)
        self.lbl_progress_text.configure(text=f"0 / {len(rows)}")

        self._lbl_stop_event.clear()
        self._lbl_is_running = True
        self.lbl_start_btn.configure(state="disabled", text="Đang chạy...")
        self.lbl_stop_btn.configure(state="normal")
        self.lbl_endpoint_entry.configure(state="disabled")
        self.lbl_model_entry.configure(state="disabled")

        endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._lbl_log(f"🚀 Bắt đầu gán nhãn {len(rows)} bình luận")
        self._lbl_log(f"   Endpoint: {endpoint}")
        self._lbl_log(f"   Model: {model_name or '(auto)'}")
        self._lbl_log(f"   Batch size: {batch_size}")
        self._lbl_log("=" * 50)

        def _labeling_thread():
            import pandas as pd
            classifier = LMStudioClassifier(endpoint=endpoint, model=model_name, timeout=120)
            labeled_path = self._get_labeled_data_path()
            if os.path.exists(labeled_path):
                try:
                    os.remove(labeled_path)
                except Exception:
                    pass
            file_exists = False
            total = len(rows)
            processed = 0
            label_counts = {}
            buffer_rows = []
            buffer_tasks = []

            def _flush_batch():
                nonlocal processed, file_exists
                predictions = classifier.predict(buffer_tasks)
                csv_rows = []
                for row_i, pred in zip(buffer_rows, predictions):
                    label = pred["result"][0]["value"]["choices"][0]
                    label_counts[label] = label_counts.get(label, 0) + 1
                    processed += 1
                    csv_rows.append({"id": row_i.get("id", processed), "text": row_i.get("text", ""), "label": label})
                    rid = row_i.get("id", processed)
                    dt = str(row_i.get("text", "")).replace("\n", "  ")[:100]
                    lb = label
                    self.after(0, lambda r=rid, d=dt, l=lb: self.lbl_tree.insert("", "end", values=(r, d, l)))
                pd.DataFrame(csv_rows).to_csv(labeled_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")
                file_exists = True
                p = processed / total
                pc = processed
                self.after(0, lambda v=p: self.lbl_progress.set(v))
                self.after(0, lambda v=pc, t=total: self.lbl_progress_text.configure(text=f"{v} / {t}"))
                stats_str = " | ".join(f"{k}: {v}" for k, v in label_counts.items())
                self.after(0, self._lbl_log, f"   ✓ Batch xong. [{stats_str}]")

            try:
                for row in rows:
                    if self._lbl_stop_event.is_set():
                        self.after(0, self._lbl_log, "🛑 Đã nhận lệnh DỪNG. Dữ liệu đã gán được lưu.")
                        break
                    buffer_rows.append(row)
                    buffer_tasks.append({"data": {"text": str(row.get("text", ""))}})
                    if len(buffer_tasks) < batch_size:
                        continue
                    self.after(0, self._lbl_log, f"📤 Gửi batch {processed + 1}–{processed + len(buffer_tasks)} / {total} ...")
                    _flush_batch()
                    buffer_rows.clear()
                    buffer_tasks.clear()
                if buffer_tasks and not self._lbl_stop_event.is_set():
                    self.after(0, self._lbl_log, f"📤 Gửi batch cuối {processed + 1}–{processed + len(buffer_tasks)} / {total} ...")
                    _flush_batch()
                self.after(0, self._lbl_log, "\n" + "=" * 50)
                self.after(0, self._lbl_log, f"✅ HOÀN TẤT: {processed}/{total} bình luận đã gán nhãn")
                for lbl, cnt in label_counts.items():
                    self.after(0, self._lbl_log, f"   {lbl}: {cnt}")
                self.after(0, self._lbl_log, f"📂 Đã lưu: {labeled_path}")
                final_msg = f"✅ Hoàn tất — {processed} dòng đã gán nhãn → labeled_data.csv"
                self.after(0, lambda: self.lbl_status_label.configure(text=final_msg, text_color="green"))
            except Exception as e:
                err_msg = str(e)
                self.after(0, self._lbl_log, f"❌ Lỗi: {err_msg}")
                self.after(0, lambda: self.lbl_status_label.configure(text=f"❌ Lỗi: {err_msg[:80]}", text_color="red"))
            finally:
                self._lbl_is_running = False
                self.after(0, lambda: self.lbl_start_btn.configure(state="normal", text="▶ Bắt Đầu Gán Nhãn"))
                self.after(0, lambda: self.lbl_stop_btn.configure(state="disabled"))
                self.after(0, lambda: self.lbl_endpoint_entry.configure(state="normal"))
                self.after(0, lambda: self.lbl_model_entry.configure(state="normal"))
                self.after(0, self.refresh_file_list)
        threading.Thread(target=_labeling_thread, daemon=True).start()

    def _lbl_stop_labeling(self):
        if self._lbl_is_running:
            self._lbl_log("⏹ Đang gửi lệnh dừng...")
            self._lbl_stop_event.set()
            self.lbl_stop_btn.configure(state="disabled", text="Đang dừng...")

    # ---------------------------------------------------------
    # TAB: LABEL MANAGER
    # ---------------------------------------------------------
    def _setup_label_manager_tab(self):
        tab = self.tab_label_mgr
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(tab)
        header.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        row1 = ctk.CTkFrame(header, fg_color="transparent")
        row1.pack(fill="x", padx=5, pady=(5, 2))
        ctk.CTkButton(row1, text="⟳ Tải dữ liệu", width=110, command=self._lm_load_data).pack(side="left", padx=5)
        ctk.CTkLabel(row1, text="Tìm kiếm:").pack(side="left", padx=(15, 5))
        self.lm_search_var = ctk.StringVar()
        ctk.CTkEntry(row1, textvariable=self.lm_search_var, placeholder_text="Nhập text để lọc...", width=200).pack(side="left", padx=5)
        ctk.CTkButton(row1, text="Lọc", width=60, command=self._lm_filter_data).pack(side="left", padx=5)
        ctk.CTkLabel(row1, text="Nhãn:").pack(side="left", padx=(10, 5))
        self.lm_label_filter_var = ctk.StringVar(value="Tất cả")
        self.lm_label_filter = ctk.CTkOptionMenu(row1, values=["Tất cả", "Clean", "Spam", "Hate Speech", "Harassment", "Obscene"],
                                                   variable=self.lm_label_filter_var, command=lambda _: self._lm_filter_data(), width=120)
        self.lm_label_filter.pack(side="left", padx=5)
        ctk.CTkButton(row1, text="Xóa lọc", width=70, command=self._lm_clear_filter).pack(side="left", padx=5)
        self.lm_stats_label = ctk.CTkLabel(row1, text="Labeled: 0 dòng", text_color="gray")
        self.lm_stats_label.pack(side="right", padx=10)

        row2 = ctk.CTkFrame(header, fg_color="transparent")
        row2.pack(fill="x", padx=5, pady=(2, 5))
        ctk.CTkButton(row2, text="Xuất CSV", width=90, fg_color="#059669", hover_color="#047857", command=self._lm_export_csv).pack(side="left", padx=5)
        ctk.CTkLabel(row2, text="Sửa nhãn →").pack(side="left", padx=(20, 5))
        self.lm_edit_label_var = ctk.StringVar(value="Clean")
        ctk.CTkOptionMenu(row2, values=["Clean", "Spam", "Hate Speech", "Harassment", "Obscene"],
                           variable=self.lm_edit_label_var, width=120).pack(side="left", padx=2)
        ctk.CTkButton(row2, text="Áp dụng", width=80, fg_color="#7C3AED", hover_color="#6D28D9", command=self._lm_edit_label).pack(side="left", padx=5)
        ctk.CTkButton(row2, text="Xóa dòng chọn", width=110, fg_color="red", hover_color="darkred", command=self._lm_delete_selected).pack(side="right", padx=5)

        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)

        tf = ctk.CTkFrame(main_frame)
        tf.grid(row=0, column=0, padx=(5, 3), pady=5, sticky="nsew")
        tf.grid_columnconfigure(0, weight=1)
        tf.grid_rowconfigure(0, weight=1)
        self.lm_tree = ttk.Treeview(tf, columns=("id", "text", "label"), show="headings", selectmode="extended")
        self.lm_tree.heading("id", text="ID")
        self.lm_tree.heading("text", text="Nội dung bình luận")
        self.lm_tree.heading("label", text="Nhãn")
        self.lm_tree.column("id", width=50, anchor="center")
        self.lm_tree.column("text", width=400, anchor="w")
        self.lm_tree.column("label", width=100, anchor="center")
        lm_scroll = ttk.Scrollbar(tf, orient="vertical", command=self.lm_tree.yview)
        self.lm_tree.configure(yscrollcommand=lm_scroll.set)
        self.lm_tree.grid(row=0, column=0, sticky="nsew")
        lm_scroll.grid(row=0, column=1, sticky="ns")

        chart_frame = ctk.CTkFrame(main_frame)
        chart_frame.grid(row=0, column=1, padx=(3, 5), pady=5, sticky="nsew")
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)
        self.lm_chart_frame = chart_frame
        self._lm_canvas = None

        self.lm_status_label = ctk.CTkLabel(tab, text="Trạng thái: Sẵn sàng", text_color="gray")
        self.lm_status_label.grid(row=2, column=0, pady=5, sticky="w", padx=10)
        self._lm_all_rows = []
        self._lm_load_data()

    def _lm_load_data(self):
        labeled_path = self._get_labeled_data_path()
        self._lm_all_rows = []
        if os.path.isfile(labeled_path):
            try:
                with open(labeled_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._lm_all_rows.append({"id": row.get("id", ""), "text": row.get("text", ""), "label": row.get("label", "Clean")})
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc labeled_data.csv: {e}")
        self._lm_display_rows(self._lm_all_rows)
        self._lm_update_chart()
        self.lm_stats_label.configure(text=f"Labeled: {len(self._lm_all_rows)} dòng")
        if self._lm_all_rows:
            self.lm_status_label.configure(text=f"Đã tải {len(self._lm_all_rows)} dòng từ labeled_data.csv", text_color="green")
        else:
            self.lm_status_label.configure(text="Chưa có dữ liệu labeled_data.csv", text_color="gray")

    def _lm_display_rows(self, rows):
        self.lm_tree.delete(*self.lm_tree.get_children())
        for row in rows:
            self.lm_tree.insert("", "end", values=(row.get("id", ""), str(row.get("text", "")).replace("\n", "  "), row.get("label", "")))

    def _lm_filter_data(self):
        query = self.lm_search_var.get().strip().lower()
        label_filter = self.lm_label_filter_var.get()
        filtered = self._lm_all_rows
        if query:
            filtered = [r for r in filtered if query in r.get("text", "").lower()]
        if label_filter != "Tất cả":
            filtered = [r for r in filtered if r.get("label", "") == label_filter]
        self._lm_display_rows(filtered)
        self.lm_status_label.configure(text=f"Hiển thị {len(filtered)}/{len(self._lm_all_rows)} dòng", text_color="blue")

    def _lm_clear_filter(self):
        self.lm_search_var.set("")
        self.lm_label_filter_var.set("Tất cả")
        self._lm_display_rows(self._lm_all_rows)
        self.lm_status_label.configure(text=f"Hiển thị tất cả {len(self._lm_all_rows)} dòng", text_color="green")

    def _lm_save_data(self):
        labeled_path = self._get_labeled_data_path()
        try:
            with open(labeled_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
                writer.writeheader()
                writer.writerows(self._lm_all_rows)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file: {e}")

    def _lm_delete_selected(self):
        selected = self.lm_tree.selection()
        if not selected:
            messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 dòng để xóa.")
            return
        ids_to_delete = set()
        for item in selected:
            vals = self.lm_tree.item(item)["values"]
            if vals:
                ids_to_delete.add(str(vals[0]))
        if not messagebox.askyesno("Xác nhận", f"Xóa {len(ids_to_delete)} dòng?"):
            return
        self._lm_all_rows = [r for r in self._lm_all_rows if str(r["id"]) not in ids_to_delete]
        self._lm_save_data()
        self._lm_display_rows(self._lm_all_rows)
        self._lm_update_chart()
        self.lm_stats_label.configure(text=f"Labeled: {len(self._lm_all_rows)} dòng")
        self.lm_status_label.configure(text=f"Đã xóa {len(ids_to_delete)} dòng.", text_color="orange")

    def _lm_edit_label(self):
        selected = self.lm_tree.selection()
        if not selected:
            messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 dòng để sửa nhãn.")
            return
        new_label = self.lm_edit_label_var.get()
        ids_to_edit = set()
        for item in selected:
            vals = self.lm_tree.item(item)["values"]
            if vals:
                ids_to_edit.add(str(vals[0]))
        changed = 0
        for row in self._lm_all_rows:
            if str(row["id"]) in ids_to_edit:
                row["label"] = new_label
                changed += 1
        self._lm_save_data()
        self._lm_display_rows(self._lm_all_rows)
        self._lm_update_chart()
        self.lm_status_label.configure(text=f"Đã sửa {changed} dòng → '{new_label}'", text_color="green")

    def _lm_export_csv(self):
        if not self._lm_all_rows:
            messagebox.showwarning("Nhắc nhở", "Không có dữ liệu để xuất.")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(os.getcwd(), f"labeled_export_{timestamp}.csv")
        try:
            with open(export_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
                writer.writeheader()
                writer.writerows(self._lm_all_rows)
            messagebox.showinfo("Thành công", f"Đã xuất {len(self._lm_all_rows)} dòng ra:\n{os.path.basename(export_path)}")
            self.refresh_file_list()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất CSV: {e}")

    def _lm_update_chart(self):
        label_colors = {"Clean": "#22C55E", "Spam": "#F59E0B", "Hate Speech": "#EF4444", "Harassment": "#8B5CF6", "Obscene": "#EC4899"}
        counts = {lbl: 0 for lbl in label_colors}
        for row in self._lm_all_rows:
            lbl = row.get("label", "Clean")
            counts[lbl] = counts.get(lbl, 0) + 1
        total = sum(counts.values())
        if self._lm_canvas:
            self._lm_canvas.get_tk_widget().destroy()
            self._lm_canvas = None
        fig = Figure(figsize=(4, 4), dpi=100, facecolor="#2B2B2B")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#2B2B2B")
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [label_colors.get(lbl, "#6B7280") for lbl in labels]
        bars = ax.barh(labels, values, color=colors, edgecolor="#444", height=0.6)
        max_val = max(values) if values and max(values) > 0 else 1
        for bar, val in zip(bars, values):
            pct = f"{val / total * 100:.1f}%" if total > 0 else "0%"
            ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val}  ({pct})", va="center", ha="left", fontsize=9, color="white", fontweight="bold")
        ax.set_title(f"Phân Bố Nhãn (n={total})", fontsize=12, color="white", fontweight="bold", pad=10)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#555")
        ax.spines["left"].set_color("#555")
        ax.xaxis.label.set_color("white")
        if max_val > 0:
            ax.set_xlim(0, max_val * 1.35)
        fig.tight_layout()
        self._lm_canvas = FigureCanvasTkAgg(fig, master=self.lm_chart_frame)
        self._lm_canvas.draw()
        self._lm_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        plt.close(fig)

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
            self.after(0, self.log_message, f"--- HOÀN TẤT. Đã lưu tổng {len(self.extracted_data)} mục vào warehouse. ---")
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

        self.extracted_data = []
        self.tree_data.delete(*self.tree_data.get_children())
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("0.0", "end")
        self.log_textbox.configure(state="disabled")

        self.stop_event.clear()
        self.set_gui_state(True)
        self.log_message("Dữ liệu sẽ được lưu trực tiếp vào warehouse.csv")

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
