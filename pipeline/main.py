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

from ui_tabs import auto_labeling_tab as _auto_labeling_tab
from ui_tabs import crawler_tab as _crawler_tab
from ui_tabs import keyword_tab as _keyword_tab
from ui_tabs import warehouse_tab as _warehouse_tab
from ui_tabs import label_manager_tab as _label_manager_tab
from ui_tabs import file_manager_tab as _file_manager_tab
from ui_tabs import config_manager_tab as _config_manager_tab
from ui_tabs import pipeline_test_tab as _pipeline_test_tab

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
        self.tab_pipeline_test = self.tabview.add("Test Pipeline")

        self._setup_crawler_tab()
        self._setup_keyword_crawler_tab()
        self._setup_warehouse_tab()
        self._setup_labeling_tab()
        self._setup_label_manager_tab()
        self._setup_file_manager_tab()
        self._setup_config_tab()
        self._setup_pipeline_test_tab()

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
        return _crawler_tab._setup_crawler_tab(self)

    # ---------------------------------------------------------
    # TAB 2: KEYWORD CRAWLER (VOZ / Threads)
    # ---------------------------------------------------------
    def _setup_keyword_crawler_tab(self):
        return _keyword_tab._setup_keyword_crawler_tab(self)

    def _on_kw_platform_change(self, platform):
        return _keyword_tab._on_kw_platform_change(self, platform)

    def refresh_keyword_history(self):
        return _keyword_tab.refresh_keyword_history(self)

    def _reuse_history_keyword(self):
        return _keyword_tab._reuse_history_keyword(self)

    def kw_log_message(self, message):
        return _keyword_tab.kw_log_message(self, message)

    def kw_handle_new_data(self, batch):
        return _keyword_tab.kw_handle_new_data(self, batch)

    def _kw_append_to_table(self, batch):
        return _keyword_tab._kw_append_to_table(self, batch)

    def _set_kw_gui_state(self, running):
        return _keyword_tab._set_kw_gui_state(self, running)

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
        return _keyword_tab._keyword_crawl_thread(
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
            num_workers,
        )

    def start_keyword_crawling(self):
        return _keyword_tab.start_keyword_crawling(self)

    def stop_keyword_crawling(self):
        return _keyword_tab.stop_keyword_crawling(self)

    # ---------------------------------------------------------
    # TAB 3: WAREHOUSE MANAGER
    # ---------------------------------------------------------
    def _setup_warehouse_tab(self):
        return _warehouse_tab._setup_warehouse_tab(self)

    def wh_load_data(self):
        return _warehouse_tab.wh_load_data(self)

    def _wh_display_rows(self, rows):
        return _warehouse_tab._wh_display_rows(self, rows)

    def wh_filter_data(self):
        return _warehouse_tab.wh_filter_data(self)

    def wh_clear_filter(self):
        return _warehouse_tab.wh_clear_filter(self)

    def wh_delete_selected(self):
        return _warehouse_tab.wh_delete_selected(self)

    def wh_remove_duplicates(self):
        return _warehouse_tab.wh_remove_duplicates(self)

    def wh_export_csv(self):
        return _warehouse_tab.wh_export_csv(self)

    def wh_run_preprocessing(self):
        return _warehouse_tab.wh_run_preprocessing(self)

    def wh_upload_drive(self):
        return _warehouse_tab.wh_upload_drive(self)

    def wh_download_drive(self):
        return _warehouse_tab.wh_download_drive(self)

    # ---------------------------------------------------------
    # TAB: AUTO LABELING (LM Studio)
    # ---------------------------------------------------------
    def _get_labeled_data_path(self) -> str:
        return _auto_labeling_tab._get_labeled_data_path(self)

    def _setup_labeling_tab(self):
        return _auto_labeling_tab._setup_labeling_tab(self)

    def _lbl_log(self, msg):
        return _auto_labeling_tab._lbl_log(self, msg)

    def _lbl_cluster_history_path(self) -> str:
        return _auto_labeling_tab._lbl_cluster_history_path(self)

    def _lbl_load_cluster_history(self) -> dict:
        return _auto_labeling_tab._lbl_load_cluster_history(self)

    def _lbl_save_cluster_history(self, data: dict) -> None:
        return _auto_labeling_tab._lbl_save_cluster_history(self, data)

    def _lbl_refresh_clusters(self):
        return _auto_labeling_tab._lbl_refresh_clusters(self)

    def _lbl_on_cluster_change(self, selected_value: str):
        return _auto_labeling_tab._lbl_on_cluster_change(self, selected_value)

    def _lbl_test_connection(self):
        return _auto_labeling_tab._lbl_test_connection(self)

    def _lbl_start_labeling(self):
        return _auto_labeling_tab._lbl_start_labeling(self)

    def _lbl_stop_labeling(self):
        return _auto_labeling_tab._lbl_stop_labeling(self)

    def _lbl_reset_labeled_data(self):
        return _auto_labeling_tab._lbl_reset_labeled_data(self)

    # ---------------------------------------------------------
    # TAB: LABEL MANAGER
    # ---------------------------------------------------------
    def _setup_label_manager_tab(self):
        return _label_manager_tab._setup_label_manager_tab(self)

    def _lm_load_data(self):
        return _label_manager_tab._lm_load_data(self)

    @staticmethod
    def _migrate_old_label(old_label: str):
        return _label_manager_tab._migrate_old_label(old_label)

    def _lm_display_rows(self, rows):
        return _label_manager_tab._lm_display_rows(self, rows)

    def _lm_filter_data(self):
        return _label_manager_tab._lm_filter_data(self)

    def _lm_clear_filter(self):
        return _label_manager_tab._lm_clear_filter(self)

    def _lm_save_data(self):
        return _label_manager_tab._lm_save_data(self)

    def _lm_delete_selected(self):
        return _label_manager_tab._lm_delete_selected(self)

    def _lm_edit_label(self):
        return _label_manager_tab._lm_edit_label(self)

    def _lm_export_csv(self):
        return _label_manager_tab._lm_export_csv(self)

    def _lm_update_chart(self):
        return _label_manager_tab._lm_update_chart(self)

    # ---------------------------------------------------------
    # LABEL MANAGER: Google Drive Sync
    # ---------------------------------------------------------
    def _lm_upload_drive(self):
        return _label_manager_tab._lm_upload_drive(self)

    def _lm_download_drive(self):
        return _label_manager_tab._lm_download_drive(self)

    # ---------------------------------------------------------
    # TAB 4: FILE MANAGER
    # ---------------------------------------------------------
    def _setup_file_manager_tab(self):
        return _file_manager_tab._setup_file_manager_tab(self)

    def refresh_file_list(self):
        return _file_manager_tab.refresh_file_list(self)

    def delete_selected_file(self):
        return _file_manager_tab.delete_selected_file(self)

    def open_current_folder(self):
        return _file_manager_tab.open_current_folder(self)

    # ---------------------------------------------------------
    # TAB 4: CONFIG MANAGER
    # ---------------------------------------------------------
    def _setup_config_tab(self):
        return _config_manager_tab._setup_config_tab(self)

    # ---------------------------------------------------------
    # TAB: PIPELINE TEST (no filter)
    # ---------------------------------------------------------
    def _setup_pipeline_test_tab(self):
        return _pipeline_test_tab._setup_pipeline_test_tab(self)

    def run_pipeline_test(self):
        return _pipeline_test_tab.run_pipeline_test(self)

    def on_config_file_change(self, filename):
        return _config_manager_tab.on_config_file_change(self, filename)

    def _get_config_path(self, filename):
        return _config_manager_tab._get_config_path(self, filename)

    # --- JSON Helpers ---
    def load_json_config(self, filename):
        return _config_manager_tab.load_json_config(self, filename)

    def refresh_json_tree(self):
        return _config_manager_tab.refresh_json_tree(self)

    def on_json_tree_select(self, event):
        return _config_manager_tab.on_json_tree_select(self, event)

    def add_or_update_json_entry(self):
        return _config_manager_tab.add_or_update_json_entry(self)

    def delete_json_entry(self):
        return _config_manager_tab.delete_json_entry(self)

    def save_json_file(self):
        return _config_manager_tab.save_json_file(self)

    # --- Python Config Helpers ---
    def load_py_config(self, filename):
        return _config_manager_tab.load_py_config(self, filename)

    def save_py_config(self):
        return _config_manager_tab.save_py_config(self)

    # ---------------------------------------------------------
    # CRAWLER EXECUTION LOGIC
    # ---------------------------------------------------------
    def log_message(self, message):
        return _crawler_tab.log_message(self, message)

    def handle_new_data(self, batch):
        return _crawler_tab.handle_new_data(self, batch)

    def _append_to_table(self, batch):
        return _crawler_tab._append_to_table(self, batch)

    def set_gui_state(self, running):
        return _crawler_tab.set_gui_state(self, running)

    def _crawl_thread(self, url, headless, u_dec, u_fil, u_nor, use_segmentor, segmentor_backend, preprocessor):
        return _crawler_tab._crawl_thread(
            self,
            url,
            headless,
            u_dec,
            u_fil,
            u_nor,
            use_segmentor,
            segmentor_backend,
            preprocessor,
        )

    def start_crawling(self):
        return _crawler_tab.start_crawling(self)

    def stop_crawling(self):
        return _crawler_tab.stop_crawling(self)

if __name__ == "__main__":
    app = App()
    app.mainloop()
