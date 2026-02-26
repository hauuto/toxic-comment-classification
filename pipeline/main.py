import os
import csv
import glob
import time
import json
import re
import threading
import concurrent.futures
import shutil
from datetime import datetime, timezone
import customtkinter as ctk
from tkinter import messagebox, ttk

from crawler import extract_comments_stream, VOZCrawler, ThreadsCrawler, load_keyword_history
from nlp_pipeline.warehouse import (
    append_to_warehouse,
    get_warehouse_count,
    get_warehouse_clusters,
    read_warehouse,
    read_warehouse_cluster,
    overwrite_warehouse,
    CLUSTER_SIZE_DEFAULT,
)
from nlp_pipeline import VietnameseCommentPreprocessor
from google_drive import upload_warehouse, download_warehouse, upload_labeled_data, download_labeled_data
from lmstudio_classifier import (LMStudioClassifier, TIER1_LABELS, TIER2_LABELS,
                                  TIER3_TOXIC_LABELS, TIER3_CLEAN_LABELS, TIER3_ALL_LABELS)
from gemini_hierarchical_classifier import GeminiHierarchicalClassifier
from nlp_pipeline.word_segmentor import WordSegmentor
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

        # Shared NLP resources (initialized once at app startup)
        self._shared_vncorenlp_segmentor: WordSegmentor | None = None
        self._preprocessor_cache: dict[str, VietnameseCommentPreprocessor] = {}
        self._startup_error: str | None = None

        # Defer building the full UI until VnCoreNLP is initialized
        self.withdraw()
        self._show_startup_loading()
        threading.Thread(target=self._init_vncorenlp_startup, daemon=True).start()

    # ---------------------------------------------------------
    # STARTUP: VnCoreNLP init (single shared instance)
    # ---------------------------------------------------------
    def _show_startup_loading(self):
        self._loading = ctk.CTkToplevel(self)
        self._loading.title("Đang khởi động")
        self._loading.geometry("520x200")
        self._loading.resizable(False, False)
        self._loading.attributes("-topmost", True)

        try:
            self._loading.grab_set()
        except Exception:
            pass

        frame = ctk.CTkFrame(self._loading)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="Đang khởi tạo VnCoreNLP (chỉ 1 lần khi mở app)...",
            font=("Arial", 15, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self._loading_status = ctk.CTkLabel(frame, text="Vui lòng đợi...", text_color="gray")
        self._loading_status.pack(anchor="w", pady=(0, 10))

        self._loading_bar = ctk.CTkProgressBar(frame)
        self._loading_bar.pack(fill="x", pady=(5, 10))
        self._loading_bar.configure(mode="indeterminate")
        self._loading_bar.start()

    def _init_vncorenlp_startup(self):
        try:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            self._shared_vncorenlp_segmentor = WordSegmentor(
                backend="vncorenlp",
                vncorenlp_dir=models_dir,
                auto_download=False,
            )
            # Cache a preprocessor wired to the shared VnCoreNLP instance
            self._preprocessor_cache["vncorenlp"] = VietnameseCommentPreprocessor(
                segmentor=self._shared_vncorenlp_segmentor,
                segmentor_backend="vncorenlp",
                vncorenlp_dir=models_dir,
            )
        except Exception as e:
            self._startup_error = str(e)
        finally:
            self.after(0, self._finish_startup)

    def _finish_startup(self):
        try:
            self._loading_bar.stop()
        except Exception:
            pass
        try:
            self._loading.destroy()
        except Exception:
            pass

        self.deiconify()

        if self._startup_error:
            messagebox.showwarning(
                "Cảnh báo",
                "Không thể khởi tạo VnCoreNLP lúc khởi động. "
                "Bạn vẫn có thể chọn Underthesea để tách từ.\n\n"
                f"Chi tiết: {self._startup_error}",
            )

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

    def _get_preprocessor(self, segmentor_backend: str) -> VietnameseCommentPreprocessor:
        backend = (segmentor_backend or "").lower().strip() or "vncorenlp"
        if backend in self._preprocessor_cache:
            return self._preprocessor_cache[backend]

        if backend == "whitespace":
            pp = VietnameseCommentPreprocessor(segmentor_backend="whitespace")
            self._preprocessor_cache[backend] = pp
            return pp

        if backend == "underthesea":
            pp = VietnameseCommentPreprocessor(segmentor_backend="underthesea")
            self._preprocessor_cache[backend] = pp
            return pp

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if self._shared_vncorenlp_segmentor is not None:
            pp = VietnameseCommentPreprocessor(
                segmentor=self._shared_vncorenlp_segmentor,
                segmentor_backend="vncorenlp",
                vncorenlp_dir=models_dir,
            )
        else:
            pp = VietnameseCommentPreprocessor(segmentor_backend="vncorenlp", vncorenlp_dir=models_dir)
        self._preprocessor_cache["vncorenlp"] = pp
        return pp

    # ---------------------------------------------------------
    # CSV helpers (robust encoding)
    # ---------------------------------------------------------
    @staticmethod
    def _read_csv_dicts_with_fallback(path: str) -> tuple[list[dict], list[str], str]:
        """Read a CSV file into list[dict] with encoding fallback.

        Returns: (rows, fieldnames, encoding_used)
        """
        encodings = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"]
        last_err: Exception | None = None
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = list(reader.fieldnames or [])
                return rows, fieldnames, enc
            except UnicodeDecodeError as e:
                last_err = e
                continue
            except Exception as e:
                # Non-decoding errors shouldn't be masked by fallback.
                raise e

        # Last resort: replace invalid bytes to avoid crashing UI.
        _ = last_err
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        return rows, fieldnames, "utf-8(replace)"

    @staticmethod
    def _backup_file(path: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak_path = f"{path}.bak_{ts}"
        try:
            shutil.copy2(path, bak_path)
        except Exception:
            # Best-effort backup
            pass
        return bak_path

    @staticmethod
    def _looks_mojibake(s: str) -> bool:
        # Common UTF-8->cp1252/latin1 mojibake fragments seen in Vietnamese text.
        patterns = ("ï»¿", "Ã", "Â", "Ä", "áº", "Æ°", "â€")
        return any(p in s for p in patterns)

    @staticmethod
    def _fix_mojibake_text(s: str) -> str:
        # Try to undo UTF-8 bytes that were incorrectly decoded as latin-1/cp1252 and then re-encoded.
        # This is safe because it only applies when s is encodable in latin-1.
        out = s
        for _ in range(2):
            if not App._looks_mojibake(out):
                break
            try:
                out2 = out.encode("latin-1").decode("utf-8")
            except Exception:
                break
            if out2 == out:
                break
            out = out2
        return out

    @staticmethod
    def _normalize_fieldname(name: str) -> str:
        # Strip BOM and whitespace. Some tools may embed BOM (U+FEFF) into the header name.
        return (name or "").replace("\ufeff", "").strip()

    @staticmethod
    def _sanitize_csv_row(row: dict) -> dict:
        # DictReader stores extra columns under key None.
        if row is None:
            return {}
        if None in row:
            try:
                row = dict(row)
                row.pop(None, None)
            except Exception:
                return {}

        cleaned: dict[str, str] = {}
        for k, v in row.items():
            if k is None:
                continue
            key = str(k)
            if App._looks_mojibake(key):
                key = App._fix_mojibake_text(key)
            key = App._normalize_fieldname(key)
            if not key:
                continue

            if v is None:
                cleaned[key] = ""
            else:
                val = str(v)
                if App._looks_mojibake(val):
                    val = App._fix_mojibake_text(val)
                cleaned[key] = val
        return cleaned

    @staticmethod
    def _rewrite_csv_utf8sig(path: str, rows: list[dict], fieldnames: list[str]) -> None:
        sanitized_rows: list[dict] = []
        for r in rows:
            if isinstance(r, dict):
                sanitized_rows.append(App._sanitize_csv_row(r))

        if not fieldnames:
            # Infer from sanitized rows
            keys: list[str] = []
            seen: set[str] = set()
            for r in sanitized_rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            fieldnames = keys
        else:
            normalized: list[str] = []
            for fn in fieldnames:
                if fn is None:
                    continue
                name = str(fn)
                if App._looks_mojibake(name):
                    name = App._fix_mojibake_text(name)
                name = App._normalize_fieldname(name)
                if name:
                    normalized.append(name)
            fieldnames = normalized
            fieldnames = [fn for fn in fieldnames if fn]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in sanitized_rows:
                writer.writerow(r)

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
        self.seg_backend_var = ctk.StringVar(value="VnCoreNLP")
        
        nlp_frame = ctk.CTkFrame(opt_frame, fg_color="transparent")
        nlp_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(nlp_frame, text="Pipeline Các Bước Tiền Xử Lý:").pack(side="left", padx=(0, 10))
        self.chk_dec = ctk.CTkCheckBox(nlp_frame, text="Decoder", variable=self.use_decoder_var)
        self.chk_dec.pack(side="left", padx=5)
        self.chk_fil = ctk.CTkCheckBox(nlp_frame, text="Filter", variable=self.use_filter_var)
        self.chk_fil.pack(side="left", padx=5)
        self.chk_nor = ctk.CTkCheckBox(nlp_frame, text="Normalizer", variable=self.use_normalizer_var)
        self.chk_nor.pack(side="left", padx=5)
        ctk.CTkLabel(nlp_frame, text="Tách từ:").pack(side="left", padx=(10, 5))
        self.seg_backend_menu = ctk.CTkOptionMenu(
            nlp_frame,
            values=["Tắt", "VnCoreNLP", "Underthesea"],
            variable=self.seg_backend_var,
            width=130,
        )
        self.seg_backend_menu.pack(side="left", padx=5)

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
        ctk.CTkLabel(self.kw_voz_frame, text="Workers:").pack(side="left", padx=(0, 5))
        self.kw_num_workers_var = ctk.StringVar(value="3")
        ctk.CTkEntry(self.kw_voz_frame, textvariable=self.kw_num_workers_var, width=40).pack(side="left", padx=(0, 15))

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
        self.kw_seg_backend_var = ctk.StringVar(value="VnCoreNLP")

        self.kw_chk_dec = ctk.CTkCheckBox(nlp_frame, text="Decoder", variable=self.kw_dec_var)
        self.kw_chk_dec.pack(side="left", padx=5)
        self.kw_chk_fil = ctk.CTkCheckBox(nlp_frame, text="Filter", variable=self.kw_fil_var)
        self.kw_chk_fil.pack(side="left", padx=5)
        self.kw_chk_nor = ctk.CTkCheckBox(nlp_frame, text="Normalizer", variable=self.kw_nor_var)
        self.kw_chk_nor.pack(side="left", padx=5)
        ctk.CTkLabel(nlp_frame, text="Tách từ:").pack(side="left", padx=(10, 5))
        self.kw_seg_backend_menu = ctk.CTkOptionMenu(
            nlp_frame,
            values=["Tắt", "VnCoreNLP", "Underthesea"],
            variable=self.kw_seg_backend_var,
            width=130,
        )
        self.kw_seg_backend_menu.pack(side="left", padx=5)

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
            self.kw_seg_backend_menu.configure(state="disabled")
            self.kw_is_running = True
        else:
            self.kw_run_button.configure(state="normal", text="Bắt Đầu")
            self.kw_stop_button.configure(state="disabled", text="Dừng")
            self.kw_entry.configure(state="normal")
            self.kw_platform_menu.configure(state="normal")
            self.kw_chk_dec.configure(state="normal")
            self.kw_chk_fil.configure(state="normal")
            self.kw_chk_nor.configure(state="normal")
            self.kw_seg_backend_menu.configure(state="normal")
            self.kw_is_running = False
            self.refresh_file_list()
            self.refresh_keyword_history()

    def _keyword_crawl_thread(
        self,
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
                self.after(0, self.kw_log_message, "Đang khởi tạo bộ tiền xử lý NLP...")

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
                        use_segmentor=use_segmentor,
                        num_workers=num_workers,
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
                        use_segmentor=use_segmentor,
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
        seg_choice = (self.kw_seg_backend_var.get() or "").strip()
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
            preprocessor = self._get_preprocessor(segmentor_backend)

        max_threads = int(self.kw_max_threads_var.get() or 10)
        max_pages = int(self.kw_max_pages_var.get() or 50)
        max_posts = int(self.kw_max_posts_var.get() or 10)
        max_scroll = int(self.kw_max_scroll_var.get() or 30)
        num_workers = int(self.kw_num_workers_var.get() or 3)

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
            args=(keywords, platform, u_dec, u_fil, u_nor, use_segmentor, preprocessor,
                  max_threads, max_pages, max_posts, max_scroll, num_workers),
            daemon=True,
        )
        thread.start()

    def stop_keyword_crawling(self):
        if self.kw_is_running:
            self.kw_log_message("Đang gửi lệnh yêu cầu dừng...")
            self.kw_stop_event.set()
            self.kw_stop_button.configure(state="disabled", text="Đang dừng...")
            # Also try to kill all browsers directly for faster stop
            if self.kw_active_crawler:
                try:
                    self.kw_active_crawler.close()
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
        self.wh_seg_backend_var = ctk.StringVar(value="Tắt")

        ctk.CTkCheckBox(row2, text="Decoder", variable=self.wh_dec_var).pack(side="left", padx=4)
        ctk.CTkCheckBox(row2, text="Filter", variable=self.wh_fil_var).pack(side="left", padx=4)
        ctk.CTkCheckBox(row2, text="Normalizer", variable=self.wh_nor_var).pack(side="left", padx=4)
        ctk.CTkLabel(row2, text="Tách từ:").pack(side="left", padx=(10, 5))
        self.wh_seg_backend_menu = ctk.CTkOptionMenu(
            row2,
            values=["Tắt", "VnCoreNLP", "Underthesea"],
            variable=self.wh_seg_backend_var,
            width=130,
        )
        self.wh_seg_backend_menu.pack(side="left", padx=4)

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
        try:
            self._lbl_refresh_clusters()
        except Exception:
            pass

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
        try:
            self._lbl_refresh_clusters()
        except Exception:
            pass

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
        try:
            self._lbl_refresh_clusters()
        except Exception:
            pass

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
        seg_choice = (self.wh_seg_backend_var.get() or "").strip()
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

        self._wh_is_processing = True
        self.wh_status_label.configure(text="Đang khởi tạo NLP pipeline... Vui lòng đợi.", text_color="orange")

        def _process():
            try:
                preprocessor = self._get_preprocessor(segmentor_backend)
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
                        use_normalizer=u_nor, use_segmentor=use_segmentor,
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
                self.after(0, lambda: self._lbl_refresh_clusters())
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

        ctk.CTkLabel(ctrl_frame, text="Workers:").pack(side="left", padx=(5, 5))
        self.lbl_workers_var = ctk.StringVar(value="4")
        ctk.CTkEntry(ctrl_frame, textvariable=self.lbl_workers_var, width=60).pack(side="left", padx=(0, 15))

        # Cluster selector (25k rows per cluster)
        ctk.CTkLabel(ctrl_frame, text="Cluster:").pack(side="left", padx=(5, 5))
        self.lbl_cluster_var = ctk.StringVar(value="")
        self.lbl_cluster_menu = ctk.CTkOptionMenu(
            ctrl_frame,
            values=["(đang tải...)"] ,
            variable=self.lbl_cluster_var,
            command=self._lbl_on_cluster_change,
            width=210,
        )
        self.lbl_cluster_menu.pack(side="left", padx=(0, 15))

        self.lbl_start_btn = ctk.CTkButton(ctrl_frame, text="▶ Bắt Đầu Gán Nhãn", width=160,
                                            fg_color="#2563EB", hover_color="#1D4ED8",
                                            command=self._lbl_start_labeling)
        self.lbl_start_btn.pack(side="left", padx=5)

        self.lbl_stop_btn = ctk.CTkButton(ctrl_frame, text="⏹ Dừng", width=80,
                                           fg_color="red", hover_color="darkred",
                                           state="disabled", command=self._lbl_stop_labeling)
        self.lbl_stop_btn.pack(side="left", padx=5)

        self.lbl_reset_btn = ctk.CTkButton(ctrl_frame, text="🔄 Reset", width=80,
                                            fg_color="#6B7280", hover_color="#4B5563",
                                            command=self._lbl_reset_labeled_data)
        self.lbl_reset_btn.pack(side="left", padx=5)

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
        self.lbl_log_textbox.insert("0.0", "Sẵn sàng. (Gemini: set GEMINI_API_KEY trong .env)\n")
        self.lbl_log_textbox.configure(state="disabled")

        table_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
        table_frame.grid(row=0, column=1, padx=(5, 5), pady=5, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.lbl_tree = ttk.Treeview(table_frame, columns=("id", "text", "tier1", "tier2", "tier3"), show="headings")
        self.lbl_tree.heading("id", text="ID")
        self.lbl_tree.heading("text", text="Nội dung bình luận")
        self.lbl_tree.heading("tier1", text="Tier1 Spam")
        self.lbl_tree.heading("tier2", text="Tier2 Toxic")
        self.lbl_tree.heading("tier3", text="Tier3 Labels")
        self.lbl_tree.column("id", width=40, anchor="center")
        self.lbl_tree.column("text", width=350, anchor="w")
        self.lbl_tree.column("tier1", width=80, anchor="center")
        self.lbl_tree.column("tier2", width=80, anchor="center")
        self.lbl_tree.column("tier3", width=150, anchor="center")

        lbl_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.lbl_tree.yview)
        self.lbl_tree.configure(yscrollcommand=lbl_scroll.set)
        self.lbl_tree.grid(row=0, column=0, sticky="nsew")
        lbl_scroll.grid(row=0, column=1, sticky="ns")

        self.lbl_status_label = ctk.CTkLabel(tab, text="Trạng thái: Đang chờ lệnh", text_color="gray")
        self.lbl_status_label.grid(row=3, column=0, pady=5, sticky="w", padx=10)

        self._lbl_is_running = False
        self._lbl_stop_event = threading.Event()

        # Init cluster options
        self._lbl_cluster_size = CLUSTER_SIZE_DEFAULT
        self._lbl_cluster_options: dict[str, dict] = {}
        self._lbl_refresh_clusters()

    def _lbl_log(self, msg):
        self.lbl_log_textbox.configure(state="normal")
        self.lbl_log_textbox.insert("end", f"{msg}\n")
        self.lbl_log_textbox.see("end")
        self.lbl_log_textbox.configure(state="disabled")

    def _lbl_cluster_history_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_history.json")

    def _lbl_load_cluster_history(self) -> dict:
        path = self._lbl_cluster_history_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _lbl_save_cluster_history(self, data: dict) -> None:
        path = self._lbl_cluster_history_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _lbl_refresh_clusters(self):
        """Recompute cluster list from current warehouse.csv and refresh the dropdown."""
        try:
            clusters = get_warehouse_clusters(cluster_size=self._lbl_cluster_size)
        except Exception:
            clusters = []

        try:
            total_rows = int(get_warehouse_count())
        except Exception:
            total_rows = 0

        options: dict[str, dict] = {}
        values: list[str] = []

        # Add an "All" option first so user can run the full warehouse.
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
            # IDs are typically 1-based sequential after overwrite/append.
            start_id = start_row + 1
            end_id = end_row + 1
            label = f"Cluster {idx + 1} ({start_id}–{end_id}) — {size}"
            options[label] = {"cluster_index": idx, "start_id": start_id, "end_id": end_id, "size": size}
            values.append(label)

        if not values:
            values = ["(warehouse trống)"]
            options = {values[0]: {"cluster_index": 0, "start_id": 0, "end_id": 0, "size": 0}}

        self._lbl_cluster_options = options
        try:
            self.lbl_cluster_menu.configure(values=values)
        except Exception:
            return

        # Restore last selection if possible
        history = self._lbl_load_cluster_history()
        last_idx = history.get("last_selected_cluster_index", 0)
        if not isinstance(last_idx, int):
            last_idx = 0

        # Pick selection
        selected = None
        for text, meta in options.items():
            if int(meta.get("cluster_index", 0)) == last_idx:
                selected = text
                break
        if selected is None:
            selected = values[0]

        try:
            self.lbl_cluster_var.set(selected)
        except Exception:
            pass

    def _lbl_on_cluster_change(self, selected_value: str):
        meta = self._lbl_cluster_options.get(selected_value, {})
        idx = int(meta.get("cluster_index", 0)) if meta else 0

        history = self._lbl_load_cluster_history()
        history["cluster_size"] = int(self._lbl_cluster_size)
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
        self._lbl_save_cluster_history(history)

        if meta and meta.get("size", 0):
            if meta.get("is_all") or idx < 0:
                self.lbl_status_label.configure(
                    text=f"Trạng thái: Đã chọn Toàn bộ (1–{meta.get('end_id')})",
                    text_color="gray",
                )
            else:
                self.lbl_status_label.configure(
                    text=f"Trạng thái: Đã chọn Cluster {idx + 1} ({meta.get('start_id')}–{meta.get('end_id')})",
                    text_color="gray",
                )

    def _lbl_test_connection(self):
        gemini_enabled = bool(os.getenv("GEMINI_API_KEY", "").strip())
        base_url = self.lbl_endpoint_var.get().strip()
        model_name = self.lbl_model_var.get().strip()
        if gemini_enabled:
            effective_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            self._lbl_log(f"🔌 Đang kiểm tra Gemini API (model={effective_model}) ...")
        else:
            if not base_url:
                messagebox.showwarning("Lỗi", "Vui lòng nhập endpoint!")
                return
            self._lbl_log(f"🔌 Đang kiểm tra kết nối tới {base_url} ...")

        self.lbl_test_btn.configure(state="disabled", text="Đang kiểm tra...")

        def _test():
            if gemini_enabled:
                result = GeminiHierarchicalClassifier.test_connection(model=model_name)
                if result["ok"]:
                    models_str = ", ".join(result["models"]) if result["models"] else "(unknown model)"
                    self.after(0, self._lbl_log, f"✅ Gemini OK! Model: {models_str}")
                    self.after(0, lambda: self.lbl_status_label.configure(
                        text="✅ Gemini API sẵn sàng", text_color="green"))
                    if result.get("models") and not model_name:
                        self.after(0, lambda: self.lbl_model_var.set(result["models"][0]))
                else:
                    self.after(0, self._lbl_log, f"❌ Lỗi Gemini: {result['error']}")
                    self.after(0, lambda: self.lbl_status_label.configure(
                        text="❌ Không thể kết nối Gemini", text_color="red"))
            else:
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
        gemini_enabled = bool(os.getenv("GEMINI_API_KEY", "").strip())
        base_url = self.lbl_endpoint_var.get().strip()
        if not gemini_enabled:
            if not base_url:
                messagebox.showwarning("Lỗi", "Vui lòng nhập endpoint!")
                return
        # Select cluster
        selected_cluster_text = (self.lbl_cluster_var.get() or "").strip()
        selected_meta = self._lbl_cluster_options.get(selected_cluster_text, {})
        cluster_index = int(selected_meta.get("cluster_index", 0)) if selected_meta else 0

        is_all = bool(selected_meta.get("is_all")) or cluster_index < 0
        if is_all:
            rows = read_warehouse()
            # UX: when choosing "Toàn bộ", start from the last position (newest rows)
            # instead of scanning top-down.
            try:
                rows = list(reversed(rows))
            except Exception:
                pass
        else:
            rows = read_warehouse_cluster(cluster_index=cluster_index, cluster_size=self._lbl_cluster_size)
        if not rows:
            messagebox.showwarning("Lỗi", "Warehouse trống! Hãy crawl dữ liệu trước.")
            return
        try:
            batch_size = max(1, min(int(self.lbl_batch_var.get()), 20))
        except ValueError:
            batch_size = 5

        try:
            workers = int((self.lbl_workers_var.get() or "").strip() or "1")
        except Exception:
            workers = 1
        workers = max(1, min(workers, 32))
        request_retries = 3 if gemini_enabled else 1
        model_name = self.lbl_model_var.get().strip()
        effective_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

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
        self.lbl_reset_btn.configure(state="disabled")
        self.lbl_endpoint_entry.configure(state="disabled")
        self.lbl_model_entry.configure(state="disabled")

        self._lbl_log(f"🚀 Bắt đầu gán nhãn {len(rows)} bình luận")
        if selected_meta and selected_meta.get("size", 0):
            if is_all:
                self._lbl_log(f"   Phạm vi: Toàn bộ (1–{selected_meta.get('end_id')})")
                self._lbl_log("   Thứ tự: từ CUỐI → ĐẦU (resume từ vị trí cuối cùng)")
            else:
                self._lbl_log(
                    f"   Cluster: {cluster_index + 1} ({selected_meta.get('start_id')}–{selected_meta.get('end_id')})"
                )
        if gemini_enabled:
            self._lbl_log("   Provider: Gemini (Google AI Studio)")
            self._lbl_log(f"   Model: {effective_model}")
        else:
            endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
            self._lbl_log("   Provider: LM Studio")
            self._lbl_log(f"   Endpoint: {endpoint}")
            self._lbl_log(f"   Model: {model_name or '(auto)'}")
        self._lbl_log(f"   Batch size: {batch_size}")
        self._lbl_log(f"   Workers: {workers}")
        self._lbl_log("=" * 50)

        def _labeling_thread():
            import pandas as pd
            # NOTE: For multi-threading, each worker thread will keep its own classifier instance
            # (requests.Session is not guaranteed thread-safe).
            endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"

            def _make_classifier():
                if gemini_enabled:
                    return GeminiHierarchicalClassifier(model=effective_model, timeout=120)
                return LMStudioClassifier(endpoint=endpoint, model=model_name, timeout=120)

            classifier_single = None
            if workers <= 1:
                classifier_single = _make_classifier()
            labeled_path = self._get_labeled_data_path()

            # --- Load existing labeled data for resume support ---
            labeled_ids = set()
            existing_rows = []
            label_counts = {}
            file_exists = False

            if os.path.exists(labeled_path):
                try:
                    raw_rows, fieldnames, enc_used = self._read_csv_dicts_with_fallback(labeled_path)
                    # If file is legacy encoding (e.g., cp1258), convert once to utf-8-sig to prevent future errors.
                    if enc_used not in ("utf-8-sig", "utf-8", "utf-8(replace)"):
                        self._backup_file(labeled_path)
                        self._rewrite_csv_utf8sig(labeled_path, raw_rows, fieldnames)
                        # Re-read using utf-8-sig for consistency
                        raw_rows, fieldnames, enc_used = self._read_csv_dicts_with_fallback(labeled_path)

                    for erow in raw_rows:
                        try:
                            eid = int(erow.get("id", 0))
                        except (ValueError, TypeError):
                            eid = 0
                        labeled_ids.add(eid)
                        existing_rows.append(erow)
                        # Rebuild label_counts from existing data
                        t1 = erow.get("tier1_spam", "")
                        t2 = erow.get("tier2_toxic", "")
                        t3_str = erow.get("tier3_labels", "")
                        if t1:
                            label_counts[t1] = label_counts.get(t1, 0) + 1
                        if t2:
                            label_counts[t2] = label_counts.get(t2, 0) + 1
                        if t3_str:
                            for lbl in str(t3_str).split("|"):
                                lbl = lbl.strip()
                                if lbl:
                                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                    file_exists = len(labeled_ids) > 0
                except Exception:
                    pass

            total = len(rows)

            # Cluster-scoped resume: only count/pre-fill rows belonging to current cluster
            cluster_ids = set(r.get("id", 0) for r in rows)
            existing_rows_cluster = []
            if existing_rows:
                for erow in existing_rows:
                    try:
                        eid = int(erow.get("id", 0))
                    except Exception:
                        continue
                    if eid in cluster_ids:
                        existing_rows_cluster.append(erow)

            skipped = len(existing_rows_cluster)

            # Pre-populate treeview with existing labeled rows (current cluster only)
            if existing_rows_cluster:
                for erow in existing_rows_cluster:
                    rid = erow.get("id", "")
                    dt = str(erow.get("text", "")).replace("\n", "  ")[:80]
                    s1 = erow.get("tier1_spam", "")
                    s2 = erow.get("tier2_toxic", "")
                    s3 = erow.get("tier3_labels", "")
                    self.after(0, lambda r=rid, d=dt, a1=s1, a2=s2, a3=s3:
                               self.lbl_tree.insert("", "end", values=(r, d, a1, a2, a3)))
                self.after(0, self._lbl_log,
                           f"♻ Tiếp tục từ {skipped}/{total} dòng đã gán nhãn trước đó")

            # Filter pending rows (skip already labeled)
            pending_rows = [r for r in rows if r.get("id") not in labeled_ids]

            if not pending_rows:
                self.after(0, self._lbl_log, f"✅ Tất cả {total} dòng đã được gán nhãn. Không cần làm gì thêm.")
                stats_str = " | ".join(f"{k}: {v}" for k, v in label_counts.items())
                if stats_str:
                    self.after(0, self._lbl_log, f"   Thống kê: [{stats_str}]")
                self.after(0, lambda: self.lbl_progress.set(1.0))
                self.after(0, lambda t=total: self.lbl_progress_text.configure(text=f"{t} / {t}"))
                final_msg = f"✅ Hoàn tất — {total} dòng đã gán nhãn → labeled_data.csv"
                self.after(0, lambda: self.lbl_status_label.configure(text=final_msg, text_color="green"))
                self._lbl_is_running = False
                self.after(0, lambda: self.lbl_start_btn.configure(state="normal", text="▶ Bắt Đầu Gán Nhãn"))
                self.after(0, lambda: self.lbl_stop_btn.configure(state="disabled"))
                self.after(0, lambda: self.lbl_reset_btn.configure(state="normal"))
                self.after(0, lambda: self.lbl_endpoint_entry.configure(state="normal"))
                self.after(0, lambda: self.lbl_model_entry.configure(state="normal"))
                self.after(0, self.refresh_file_list)
                return

            # Update progress to reflect already-labeled rows
            processed = skipped
            if skipped > 0:
                p = (skipped / total) if total else 0
                self.after(0, lambda v=p: self.lbl_progress.set(v))
                self.after(0, lambda v=skipped, t=total: self.lbl_progress_text.configure(text=f"{v} / {t}"))

            self.after(0, self._lbl_log, f"📋 Còn {len(pending_rows)} dòng cần gán nhãn")

            def _default_result() -> dict:
                return {
                    "tier1_spam": "Not Spam",
                    "tier2_toxic": "Clean",
                    "tier3_labels": ["Neutral"],
                }

            def _write_batch_results(batch_rows, predictions):
                nonlocal processed, file_exists
                csv_rows = []
                for row_i, pred in zip(batch_rows, predictions):
                    t1 = pred.get("tier1_spam", "Not Spam")
                    t2 = pred.get("tier2_toxic", "Clean")
                    t3_list = pred.get("tier3_labels", []) or []
                    t3 = "|".join(t3_list) if t3_list else ""

                    label_counts[t1] = label_counts.get(t1, 0) + 1
                    label_counts[t2] = label_counts.get(t2, 0) + 1
                    for lbl in t3_list:
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1

                    processed += 1
                    csv_rows.append({
                        "id": row_i.get("id", processed),
                        "text": row_i.get("text", ""),
                        "tier1_spam": t1,
                        "tier2_toxic": t2,
                        "tier3_labels": t3,
                    })

                    rid = row_i.get("id", processed)
                    dt = str(row_i.get("text", "")).replace("\n", "  ")[:80]
                    self.after(0, lambda r=rid, d=dt, s1=t1, s2=t2, s3=t3:
                               self.lbl_tree.insert("", "end", values=(r, d, s1, s2, s3)))

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
                self.after(0, lambda v=p: self.lbl_progress.set(v))
                self.after(0, lambda v=pc, t=total: self.lbl_progress_text.configure(text=f"{v} / {t}"))
                stats_str = " | ".join(f"{k}: {v}" for k, v in label_counts.items())
                self.after(0, self._lbl_log, f"   ✓ Batch xong. [{stats_str}]")

            def _split_batches():
                batches = []
                buf_rows = []
                buf_tasks = []
                for row in pending_rows:
                    if self._lbl_stop_event.is_set():
                        break
                    buf_rows.append(row)
                    buf_tasks.append({"data": {"text": str(row.get("text", ""))}})
                    if len(buf_tasks) >= batch_size:
                        batches.append((buf_rows, buf_tasks))
                        buf_rows = []
                        buf_tasks = []
                if buf_tasks and not self._lbl_stop_event.is_set():
                    batches.append((buf_rows, buf_tasks))
                return batches

            try:
                batches = _split_batches()
                if self._lbl_stop_event.is_set():
                    self.after(0, self._lbl_log, "🛑 Đã nhận lệnh DỪNG. Dữ liệu đã gán được lưu.")

                if workers <= 1:
                    # Sequential (backward compatible)
                    for i, (batch_rows, batch_tasks) in enumerate(batches):
                        if self._lbl_stop_event.is_set():
                            self.after(0, self._lbl_log, "🛑 Đã nhận lệnh DỪNG. Dữ liệu đã gán được lưu.")
                            break
                        self.after(
                            0,
                            self._lbl_log,
                            f"📤 Gửi batch {i + 1}/{len(batches)} (n={len(batch_tasks)}) ...",
                        )
                        try:
                            predictions = classifier_single.predict(
                                batch_tasks,
                                retries=request_retries,
                                strict=True,
                            )
                        except Exception as e:
                            # Hard-fail on request error: do not silently label defaults.
                            self._lbl_stop_event.set()
                            raise RuntimeError(f"Request thất bại ở batch {i + 1}/{len(batches)}: {e}") from e

                        if not isinstance(predictions, list) or len(predictions) != len(batch_tasks):
                            self._lbl_stop_event.set()
                            raise RuntimeError(
                                f"Response không hợp lệ ở batch {i + 1}/{len(batches)}: got {type(predictions)} len={getattr(predictions, '__len__', lambda: '?')()}"
                            )
                        _write_batch_results(batch_rows, predictions)
                else:
                    # Concurrent: keep ordered writes/UI updates, but run predict() in parallel.
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
                            if self._lbl_stop_event.is_set():
                                self.after(0, self._lbl_log, "🛑 Đã nhận lệnh DỪNG. Đang hủy các batch còn lại...")
                                break

                            while (not self._lbl_stop_event.is_set()) and next_to_submit < len(batches) and len(inflight) < max_inflight:
                                batch_rows, batch_tasks = batches[next_to_submit]
                                self.after(
                                    0,
                                    self._lbl_log,
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
                            except Exception as e:
                                # Hard-fail on request error: stop and surface the error.
                                self._lbl_stop_event.set()
                                raise RuntimeError(f"Request thất bại ở batch {next_to_write + 1}/{len(batches)}: {e}") from e

                            inflight.pop(next_to_write, None)
                            if not isinstance(predictions, list) or len(predictions) != len(batch_tasks):
                                self._lbl_stop_event.set()
                                raise RuntimeError(
                                    f"Response không hợp lệ ở batch {next_to_write + 1}/{len(batches)}: got {type(predictions)} len={getattr(predictions, '__len__', lambda: '?')()}"
                                )

                            _write_batch_results(batch_rows, predictions)
                            next_to_write += 1
                    finally:
                        ex.shutdown(wait=not self._lbl_stop_event.is_set(), cancel_futures=True)

                newly_labeled = processed - skipped
                if self._lbl_stop_event.is_set():
                    self.after(0, self._lbl_log, "\n" + "=" * 50)
                    self.after(0, self._lbl_log, f"🛑 ĐÃ DỪNG: {processed}/{total} bình luận đã được lưu ({newly_labeled} mới gán)")
                    stats_str = " | ".join(f"{k}: {v}" for k, v in label_counts.items())
                    if stats_str:
                        self.after(0, self._lbl_log, f"   Thống kê: [{stats_str}]")
                    self.after(0, self._lbl_log, f"📂 Đã lưu: {labeled_path}")
                    stop_msg = f"🛑 Đã dừng — {processed}/{total} dòng đã lưu"
                    self.after(0, lambda: self.lbl_status_label.configure(text=stop_msg, text_color="orange"))
                else:
                    self.after(0, self._lbl_log, "\n" + "=" * 50)
                    self.after(0, self._lbl_log, f"✅ HOÀN TẤT: {processed}/{total} bình luận đã gán nhãn ({newly_labeled} mới gán)")
                    for lbl, cnt in label_counts.items():
                        self.after(0, self._lbl_log, f"   {lbl}: {cnt}")
                    self.after(0, self._lbl_log, f"📂 Đã lưu: {labeled_path}")
                    final_msg = f"✅ Hoàn tất — {processed}/{total} dòng đã gán nhãn ({newly_labeled} mới) → labeled_data.csv"
                    self.after(0, lambda: self.lbl_status_label.configure(text=final_msg, text_color="green"))
            except Exception as e:
                err_msg = str(e)
                self.after(0, self._lbl_log, f"❌ Lỗi: {err_msg}")
                self.after(0, lambda: self.lbl_status_label.configure(text=f"❌ Lỗi: {err_msg[:80]}", text_color="red"))
            finally:
                self._lbl_is_running = False
                self.after(0, lambda: self.lbl_start_btn.configure(state="normal", text="▶ Bắt Đầu Gán Nhãn"))
                self.after(0, lambda: self.lbl_stop_btn.configure(state="disabled"))
                self.after(0, lambda: self.lbl_reset_btn.configure(state="normal"))
                self.after(0, lambda: self.lbl_endpoint_entry.configure(state="normal"))
                self.after(0, lambda: self.lbl_model_entry.configure(state="normal"))
                self.after(0, self.refresh_file_list)
        threading.Thread(target=_labeling_thread, daemon=True).start()

    def _lbl_stop_labeling(self):
        if self._lbl_is_running:
            self._lbl_log("⏹ Đang gửi lệnh dừng...")
            self._lbl_stop_event.set()
            self.lbl_stop_btn.configure(state="disabled", text="Đang dừng...")

    def _lbl_reset_labeled_data(self):
        """Xóa labeled_data.csv và reset toàn bộ tiến trình gán nhãn."""
        if self._lbl_is_running:
            messagebox.showwarning("Lỗi", "Không thể reset khi đang gán nhãn! Hãy dừng trước.")
            return
        labeled_path = self._get_labeled_data_path()
        if not os.path.exists(labeled_path):
            messagebox.showinfo("Thông báo", "Chưa có file labeled_data.csv để xóa.")
            return
        confirm = messagebox.askyesno(
            "Xác nhận Reset",
            "Bạn có chắc chắn muốn xóa toàn bộ dữ liệu đã gán nhãn?\n\n"
            "File labeled_data.csv sẽ bị xóa và bạn phải gán nhãn lại từ đầu.\n"
            "Hành động này KHÔNG THỂ hoàn tác!",
            icon="warning"
        )
        if not confirm:
            return
        try:
            os.remove(labeled_path)
            self.lbl_tree.delete(*self.lbl_tree.get_children())
            self.lbl_progress.set(0)
            self.lbl_progress_text.configure(text="0 / 0")
            self.lbl_log_textbox.configure(state="normal")
            self.lbl_log_textbox.delete("0.0", "end")
            self.lbl_log_textbox.configure(state="disabled")
            self._lbl_log("🔄 Đã reset — labeled_data.csv đã bị xóa.")
            self._lbl_log("Sẵn sàng gán nhãn lại từ đầu.")
            self.lbl_status_label.configure(text="🔄 Đã reset dữ liệu gán nhãn", text_color="orange")
            self.refresh_file_list()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xóa file: {e}")

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
        ctk.CTkEntry(row1, textvariable=self.lm_search_var, placeholder_text="Nhập text để lọc...", width=180).pack(side="left", padx=5)
        ctk.CTkButton(row1, text="Lọc", width=60, command=self._lm_filter_data).pack(side="left", padx=5)

        ctk.CTkLabel(row1, text="Tier1:").pack(side="left", padx=(10, 3))
        self.lm_t1_filter_var = ctk.StringVar(value="Tất cả")
        ctk.CTkOptionMenu(row1, values=["Tất cả"] + TIER1_LABELS,
                           variable=self.lm_t1_filter_var, command=lambda _: self._lm_filter_data(), width=100).pack(side="left", padx=2)

        ctk.CTkLabel(row1, text="Tier2:").pack(side="left", padx=(8, 3))
        self.lm_t2_filter_var = ctk.StringVar(value="Tất cả")
        ctk.CTkOptionMenu(row1, values=["Tất cả"] + TIER2_LABELS,
                           variable=self.lm_t2_filter_var, command=lambda _: self._lm_filter_data(), width=100).pack(side="left", padx=2)

        ctk.CTkLabel(row1, text="Tier3:").pack(side="left", padx=(8, 3))
        self.lm_t3_filter_var = ctk.StringVar(value="Tất cả")
        ctk.CTkOptionMenu(row1, values=["Tất cả"] + TIER3_ALL_LABELS,
                           variable=self.lm_t3_filter_var, command=lambda _: self._lm_filter_data(), width=120).pack(side="left", padx=2)

        ctk.CTkButton(row1, text="Xóa lọc", width=70, command=self._lm_clear_filter).pack(side="left", padx=5)
        self.lm_stats_label = ctk.CTkLabel(row1, text="Labeled: 0 dòng", text_color="gray")
        self.lm_stats_label.pack(side="right", padx=10)

        row2 = ctk.CTkFrame(header, fg_color="transparent")
        row2.pack(fill="x", padx=5, pady=(2, 5))
        ctk.CTkButton(row2, text="Xuất CSV", width=90, fg_color="#059669", hover_color="#047857", command=self._lm_export_csv).pack(side="left", padx=5)

        # --- Edit labels section ---
        ctk.CTkLabel(row2, text="Sửa →").pack(side="left", padx=(20, 3))
        ctk.CTkLabel(row2, text="T1:").pack(side="left", padx=(0, 2))
        self.lm_edit_t1_var = ctk.StringVar(value="—")
        ctk.CTkOptionMenu(row2, values=["—"] + TIER1_LABELS, variable=self.lm_edit_t1_var, width=95).pack(side="left", padx=2)
        ctk.CTkLabel(row2, text="T2:").pack(side="left", padx=(5, 2))
        self.lm_edit_t2_var = ctk.StringVar(value="—")
        ctk.CTkOptionMenu(row2, values=["—"] + TIER2_LABELS, variable=self.lm_edit_t2_var, width=85).pack(side="left", padx=2)
        ctk.CTkLabel(row2, text="T3:").pack(side="left", padx=(5, 2))
        self.lm_edit_t3_var = ctk.StringVar(value="—")
        ctk.CTkOptionMenu(row2, values=["—"] + TIER3_ALL_LABELS, variable=self.lm_edit_t3_var, width=120).pack(side="left", padx=2)
        ctk.CTkButton(row2, text="Áp dụng", width=80, fg_color="#7C3AED", hover_color="#6D28D9", command=self._lm_edit_label).pack(side="left", padx=5)

        # --- Drive sync + delete ---
        ctk.CTkButton(row2, text="Xóa dòng chọn", width=110, fg_color="red", hover_color="darkred", command=self._lm_delete_selected).pack(side="right", padx=5)
        ctk.CTkButton(row2, text="⬇ Drive", width=90, fg_color="#0284C7", hover_color="#0369A1",
                       command=self._lm_download_drive).pack(side="right", padx=3)
        ctk.CTkButton(row2, text="⬆ Drive", width=90, fg_color="#0284C7", hover_color="#0369A1",
                       command=self._lm_upload_drive).pack(side="right", padx=3)

        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)

        tf = ctk.CTkFrame(main_frame)
        tf.grid(row=0, column=0, padx=(5, 3), pady=5, sticky="nsew")
        tf.grid_columnconfigure(0, weight=1)
        tf.grid_rowconfigure(0, weight=1)
        self.lm_tree = ttk.Treeview(tf, columns=("id", "text", "tier1", "tier2", "tier3"), show="headings", selectmode="extended")
        self.lm_tree.heading("id", text="ID")
        self.lm_tree.heading("text", text="Nội dung bình luận")
        self.lm_tree.heading("tier1", text="Tier1 Spam")
        self.lm_tree.heading("tier2", text="Tier2 Toxic")
        self.lm_tree.heading("tier3", text="Tier3 Labels")
        self.lm_tree.column("id", width=40, anchor="center")
        self.lm_tree.column("text", width=300, anchor="w")
        self.lm_tree.column("tier1", width=80, anchor="center")
        self.lm_tree.column("tier2", width=80, anchor="center")
        self.lm_tree.column("tier3", width=130, anchor="center")
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
        self._lm_is_processing = False
        self._lm_load_data()

    def _lm_load_data(self):
        labeled_path = self._get_labeled_data_path()
        self._lm_all_rows = []
        if os.path.isfile(labeled_path):
            try:
                raw_rows, fieldnames, enc_used = self._read_csv_dicts_with_fallback(labeled_path)
                # Auto-convert only when we are reasonably confident (cp1258/cp1252).
                # Avoid destructive conversions from latin-1 or utf-8(replace).
                if enc_used in ("cp1258", "cp1252"):
                    bak = self._backup_file(labeled_path)
                    self._rewrite_csv_utf8sig(labeled_path, raw_rows, fieldnames)
                    self.lm_status_label.configure(text=f"Đã sửa encoding (backup: {os.path.basename(bak)})", text_color="orange")
                    raw_rows, fieldnames, enc_used = self._read_csv_dicts_with_fallback(labeled_path)

                # If file content looks like mojibake (double-encoded), repair once.
                need_repair = False
                for r in raw_rows[:200]:
                    if not isinstance(r, dict):
                        continue
                    for v in r.values():
                        if isinstance(v, str) and self._looks_mojibake(v):
                            need_repair = True
                            break
                    if need_repair:
                        break

                if need_repair:
                    bak = self._backup_file(labeled_path)
                    self._rewrite_csv_utf8sig(labeled_path, raw_rows, fieldnames)
                    self.lm_status_label.configure(text=f"Đã repair chữ Việt (backup: {os.path.basename(bak)})", text_color="orange")
                    raw_rows, fieldnames, enc_used = self._read_csv_dicts_with_fallback(labeled_path)

                for row in raw_rows:
                    if isinstance(row, dict):
                        row = self._sanitize_csv_row(row)
                    # New 3-tier format
                    if "tier1_spam" in row:
                        self._lm_all_rows.append({
                            "id": row.get("id", ""),
                            "text": row.get("text", ""),
                            "tier1_spam": row.get("tier1_spam", "Not Spam"),
                            "tier2_toxic": row.get("tier2_toxic", "Clean"),
                            "tier3_labels": row.get("tier3_labels", ""),
                        })
                    else:
                        # Backward compatibility: migrate old format (id, text, label)
                        old_label = row.get("label", "Clean")
                        t1, t2, t3 = self._migrate_old_label(str(old_label))
                        self._lm_all_rows.append({
                            "id": row.get("id", ""),
                            "text": row.get("text", ""),
                            "tier1_spam": t1,
                            "tier2_toxic": t2,
                            "tier3_labels": t3,
                        })
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc labeled_data.csv: {e}")
        self._lm_display_rows(self._lm_all_rows)
        self._lm_update_chart()
        self.lm_stats_label.configure(text=f"Labeled: {len(self._lm_all_rows)} dòng")
        if self._lm_all_rows:
            self.lm_status_label.configure(text=f"Đã tải {len(self._lm_all_rows)} dòng từ labeled_data.csv", text_color="green")
        else:
            self.lm_status_label.configure(text="Chưa có dữ liệu labeled_data.csv", text_color="gray")

    @staticmethod
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

    def _lm_display_rows(self, rows):
        self.lm_tree.delete(*self.lm_tree.get_children())
        for row in rows:
            self.lm_tree.insert("", "end", values=(
                row.get("id", ""),
                str(row.get("text", "")).replace("\n", "  "),
                row.get("tier1_spam", ""),
                row.get("tier2_toxic", ""),
                row.get("tier3_labels", ""),
            ))

    def _lm_filter_data(self):
        query = self.lm_search_var.get().strip().lower()
        t1_filter = self.lm_t1_filter_var.get()
        t2_filter = self.lm_t2_filter_var.get()
        t3_filter = self.lm_t3_filter_var.get()
        filtered = self._lm_all_rows
        if query:
            filtered = [r for r in filtered if query in r.get("text", "").lower()]
        if t1_filter != "Tất cả":
            filtered = [r for r in filtered if r.get("tier1_spam", "") == t1_filter]
        if t2_filter != "Tất cả":
            filtered = [r for r in filtered if r.get("tier2_toxic", "") == t2_filter]
        if t3_filter != "Tất cả":
            filtered = [r for r in filtered if t3_filter in r.get("tier3_labels", "").split("|")]
        self._lm_display_rows(filtered)
        self.lm_status_label.configure(text=f"Hiển thị {len(filtered)}/{len(self._lm_all_rows)} dòng", text_color="blue")

    def _lm_clear_filter(self):
        self.lm_search_var.set("")
        self.lm_t1_filter_var.set("Tất cả")
        self.lm_t2_filter_var.set("Tất cả")
        self.lm_t3_filter_var.set("Tất cả")
        self._lm_display_rows(self._lm_all_rows)
        self.lm_status_label.configure(text=f"Hiển thị tất cả {len(self._lm_all_rows)} dòng", text_color="green")

    def _lm_save_data(self):
        labeled_path = self._get_labeled_data_path()
        try:
            with open(labeled_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "tier1_spam", "tier2_toxic", "tier3_labels"])
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
        new_t1 = self.lm_edit_t1_var.get()
        new_t2 = self.lm_edit_t2_var.get()
        new_t3 = self.lm_edit_t3_var.get()
        if new_t1 == "—" and new_t2 == "—" and new_t3 == "—":
            messagebox.showwarning("Nhắc nhở", "Hãy chọn ít nhất 1 tier để sửa (T1, T2, hoặc T3).")
            return
        ids_to_edit = set()
        for item in selected:
            vals = self.lm_tree.item(item)["values"]
            if vals:
                ids_to_edit.add(str(vals[0]))
        changed = 0
        for row in self._lm_all_rows:
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
        self._lm_save_data()
        self._lm_display_rows(self._lm_all_rows)
        self._lm_update_chart()
        parts = []
        if new_t1 != "—":
            parts.append(f"T1→{new_t1}")
        if new_t2 != "—":
            parts.append(f"T2→{new_t2}")
        if new_t3 != "—":
            parts.append(f"T3+={new_t3}")
        self.lm_status_label.configure(text=f"Đã sửa {changed} dòng: {', '.join(parts)}", text_color="green")

    def _lm_export_csv(self):
        if not self._lm_all_rows:
            messagebox.showwarning("Nhắc nhở", "Không có dữ liệu để xuất.")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(os.getcwd(), f"labeled_export_{timestamp}.csv")
        try:
            with open(export_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "tier1_spam", "tier2_toxic", "tier3_labels"])
                writer.writeheader()
                writer.writerows(self._lm_all_rows)
            messagebox.showinfo("Thành công", f"Đã xuất {len(self._lm_all_rows)} dòng ra:\n{os.path.basename(export_path)}")
            self.refresh_file_list()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất CSV: {e}")

    def _lm_update_chart(self):
        total = len(self._lm_all_rows)

        # Count tiers
        t1_counts = {"Spam": 0, "Not Spam": 0}
        t2_counts = {"Toxic": 0, "Clean": 0}
        t3_counts = {}
        for lbl in TIER3_ALL_LABELS:
            t3_counts[lbl] = 0

        for row in self._lm_all_rows:
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

        if self._lm_canvas:
            self._lm_canvas.get_tk_widget().destroy()
            self._lm_canvas = None

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
                t1_values, labels=t1_labels, colors=t1_colors, autopct='%1.0f%%',
                startangle=90, textprops={"color": "white", "fontsize": 8})
            for at in autotexts:
                at.set_fontsize(7)
        ax1.set_title(f"Tier 1 – Spam Check (n={total})", fontsize=10, color="white", fontweight="bold", pad=5)

        # --- Tier 2: Pie ---
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.set_facecolor(bg_color)
        t2_labels = list(t2_counts.keys())
        t2_values = list(t2_counts.values())
        t2_colors = ["#EF4444", "#22C55E"]
        if sum(t2_values) > 0:
            wedges2, texts2, autotexts2 = ax2.pie(
                t2_values, labels=t2_labels, colors=t2_colors, autopct='%1.0f%%',
                startangle=90, textprops={"color": "white", "fontsize": 8})
            for at in autotexts2:
                at.set_fontsize(7)
        ax2.set_title("Tier 2 – Toxic Check", fontsize=10, color="white", fontweight="bold", pad=5)

        # --- Tier 3: Horizontal bar ---
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.set_facecolor(bg_color)
        t3_color_map = {
            "Hate Speech": "#EF4444", "Harassment": "#8B5CF6", "Obscene": "#EC4899",
            "Positive": "#22C55E", "Negative": "#F59E0B", "Neutral": "#6B7280",
        }
        t3_labels = list(t3_counts.keys())
        t3_values = list(t3_counts.values())
        t3_colors = [t3_color_map.get(lbl, "#6B7280") for lbl in t3_labels]
        bars = ax3.barh(t3_labels, t3_values, color=t3_colors, edgecolor="#444", height=0.55)
        max_val = max(t3_values) if t3_values and max(t3_values) > 0 else 1
        for bar, val in zip(bars, t3_values):
            if val > 0:
                ax3.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", ha="left", fontsize=8, color="white", fontweight="bold")
        ax3.set_title("Tier 3 – Multi-label", fontsize=10, color="white", fontweight="bold", pad=5)
        ax3.tick_params(colors="white", labelsize=7)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["bottom"].set_color("#555")
        ax3.spines["left"].set_color("#555")
        if max_val > 0:
            ax3.set_xlim(0, max_val * 1.25)

        fig.tight_layout(pad=1.5)
        self._lm_canvas = FigureCanvasTkAgg(fig, master=self.lm_chart_frame)
        self._lm_canvas.draw()
        self._lm_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        plt.close(fig)

    # ---------------------------------------------------------
    # LABEL MANAGER: Google Drive Sync
    # ---------------------------------------------------------
    def _lm_upload_drive(self):
        """Upload labeled_data.csv to Google Drive in background thread."""
        if self._lm_is_processing:
            messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
            return
        if not self._lm_all_rows:
            messagebox.showwarning("Nhắc nhở", "Chưa có dữ liệu labeled để upload.")
            return

        self._lm_is_processing = True
        self.lm_status_label.configure(text="⬆ Đang upload labeled_data.csv lên Google Drive...", text_color="orange")

        def _upload():
            try:
                def _log(msg):
                    self.after(0, lambda: self.lm_status_label.configure(text=msg, text_color="orange"))

                upload_labeled_data(log_callback=_log)
                self.after(0, lambda: self.lm_status_label.configure(
                    text=f"✓ Đã upload labeled_data.csv lên Google Drive ({len(self._lm_all_rows)} dòng).",
                    text_color="green"))
            except FileNotFoundError as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                self.after(0, lambda: self.lm_status_label.configure(
                    text="✗ Upload thất bại: thiếu credentials.", text_color="red"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi Upload", f"Không thể upload lên Drive:\n{str(e)}"))
                self.after(0, lambda: self.lm_status_label.configure(
                    text=f"✗ Upload thất bại: {str(e)[:80]}", text_color="red"))
            finally:
                self._lm_is_processing = False

        threading.Thread(target=_upload, daemon=True).start()

    def _lm_download_drive(self):
        """Download labeled_data.csv from Google Drive in background thread."""
        if self._lm_is_processing:
            messagebox.showwarning("Nhắc nhở", "Đang xử lý, vui lòng đợi...")
            return

        if self._lm_all_rows:
            if not messagebox.askyesno(
                "Xác nhận",
                f"Labeled data hiện có {len(self._lm_all_rows)} dòng.\n"
                "Tải từ Drive sẽ GHI ĐÈ toàn bộ dữ liệu local.\n\n"
                "Bạn có muốn tiếp tục?"
            ):
                return

        self._lm_is_processing = True
        self.lm_status_label.configure(text="⬇ Đang tải labeled_data.csv từ Google Drive...", text_color="orange")

        def _download():
            try:
                def _log(msg):
                    self.after(0, lambda: self.lm_status_label.configure(text=msg, text_color="orange"))

                success = download_labeled_data(log_callback=_log)
                if success:
                    self.after(0, self._lm_load_data)
                    self.after(0, lambda: self.lm_status_label.configure(
                        text="✓ Đã tải labeled_data.csv từ Google Drive và cập nhật.",
                        text_color="green"))
                else:
                    self.after(0, lambda: self.lm_status_label.configure(
                        text="✗ Không tìm thấy labeled_data.csv trên Google Drive.",
                        text_color="red"))
            except FileNotFoundError as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                self.after(0, lambda: self.lm_status_label.configure(
                    text="✗ Download thất bại: thiếu credentials.", text_color="red"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi Download", f"Không thể tải từ Drive:\n{str(e)}"))
                self.after(0, lambda: self.lm_status_label.configure(
                    text=f"✗ Download thất bại: {str(e)[:80]}", text_color="red"))
            finally:
                self._lm_is_processing = False

        threading.Thread(target=_download, daemon=True).start()

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
            self.after(0, self._lbl_refresh_clusters)
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
            self.seg_backend_menu.configure(state="disabled")
            self.is_running = True
        else:
            self.run_button.configure(state="normal", text="Bắt Đầu")
            self.stop_button.configure(state="disabled")
            self.url_entry.configure(state="normal")
            self.chk_dec.configure(state="normal")
            self.chk_fil.configure(state="normal")
            self.chk_nor.configure(state="normal")
            self.seg_backend_menu.configure(state="normal")
            self.is_running = False
            self.refresh_file_list()

    def _crawl_thread(self, url, headless, u_dec, u_fil, u_nor, use_segmentor, segmentor_backend, preprocessor):
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
        seg_choice = (self.seg_backend_var.get() or "").strip()
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
            preprocessor = self._get_preprocessor(segmentor_backend)

        self.extracted_data = []
        self.tree_data.delete(*self.tree_data.get_children())
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("0.0", "end")
        self.log_textbox.configure(state="disabled")

        self.stop_event.clear()
        self.set_gui_state(True)
        self.log_message("Dữ liệu sẽ được lưu trực tiếp vào warehouse.csv")

        thread = threading.Thread(target=self._crawl_thread,
                                  args=(url_input, headless, u_dec, u_fil, u_nor, use_segmentor, segmentor_backend, preprocessor),
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
