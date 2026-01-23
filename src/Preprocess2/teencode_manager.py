import json
import os
import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import csv
from collections import Counter
from itertools import product


# Default Leet Speak mappings (letter -> list of replacements)
DEFAULT_LEET_MAP = {
    'a': ['4', '@', '/\\'],
    'b': ['8', '|3'],
    'c': ['(', '[', '<'],
    'd': ['|)', '|>'],
    'e': ['3', '€'],
    'f': ['|=', 'ph'],
    'g': ['9', '6', 'q'],
    'h': ['#', '|-|', '}{'],
    'i': ['1', '|', '!'],
    'j': ['_|'],
    'k': ['|<', '|{'],
    'l': ['1', '|_', '|'],
    'm': ['/\\/\\', '|\\/|'],
    'n': ['/\\/', '|\\|'],
    'o': ['0', '()'],
    'p': ['|>', '|*'],
    'q': ['9', '0_'],
    'r': ['|2', '|?', '12'],
    's': ['5', '$'],
    't': ['7', '+'],
    'u': ['|_|', '\\_/', 'v'],
    'v': ['\\/', '>'],
    'w': ['\\/\\/', '|/\\|', 'vv'],
    'x': ['><', '}{'],
    'y': ['`/', '¥'],
    'z': ['2', '7_'],
    # Vietnamese specific
    'á': ['a', 'a\''],
    'à': ['a', 'a`'],
    'ả': ['a', 'a?'],
    'ã': ['a', 'a~'],
    'ạ': ['a', 'a.'],
    'ă': ['a', 'aw'],
    'ắ': ['a', 'aw\''],
    'ằ': ['a', 'aw`'],
    'ẳ': ['a', 'aw?'],
    'ẵ': ['a', 'aw~'],
    'ặ': ['a', 'aw.'],
    'â': ['a', 'aa'],
    'ấ': ['a', 'aa\''],
    'ầ': ['a', 'aa`'],
    'ẩ': ['a', 'aa?'],
    'ẫ': ['a', 'aa~'],
    'ậ': ['a', 'aa.'],
    'é': ['e', 'e\''],
    'è': ['e', 'e`'],
    'ẻ': ['e', 'e?'],
    'ẽ': ['e', 'e~'],
    'ẹ': ['e', 'e.'],
    'ê': ['e', 'ee'],
    'ế': ['e', 'ee\''],
    'ề': ['e', 'ee`'],
    'ể': ['e', 'ee?'],
    'ễ': ['e', 'ee~'],
    'ệ': ['e', 'ee.'],
    'í': ['i', 'i\''],
    'ì': ['i', 'i`'],
    'ỉ': ['i', 'i?'],
    'ĩ': ['i', 'i~'],
    'ị': ['i', 'i.'],
    'ó': ['o', 'o\''],
    'ò': ['o', 'o`'],
    'ỏ': ['o', 'o?'],
    'õ': ['o', 'o~'],
    'ọ': ['o', 'o.'],
    'ô': ['o', 'oo'],
    'ố': ['o', 'oo\''],
    'ồ': ['o', 'oo`'],
    'ổ': ['o', 'oo?'],
    'ỗ': ['o', 'oo~'],
    'ộ': ['o', 'oo.'],
    'ơ': ['o', 'ow'],
    'ớ': ['o', 'ow\''],
    'ờ': ['o', 'ow`'],
    'ở': ['o', 'ow?'],
    'ỡ': ['o', 'ow~'],
    'ợ': ['o', 'ow.'],
    'ú': ['u', 'u\''],
    'ù': ['u', 'u`'],
    'ủ': ['u', 'u?'],
    'ũ': ['u', 'u~'],
    'ụ': ['u', 'u.'],
    'ư': ['u', 'uw'],
    'ứ': ['u', 'uw\''],
    'ừ': ['u', 'uw`'],
    'ử': ['u', 'uw?'],
    'ữ': ['u', 'uw~'],
    'ự': ['u', 'uw.'],
    'ý': ['y', 'y\''],
    'ỳ': ['y', 'y`'],
    'ỷ': ['y', 'y?'],
    'ỹ': ['y', 'y~'],
    'ỵ': ['y', 'y.'],
    'đ': ['d', 'dd', 'dj'],
}

class VietnameseDictionary:
    """Vietnamese dictionary using Viet74K from underthesea"""
    _instance = None
    _words = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._words = set()
        self._load_dictionary()
    
    def _load_dictionary(self):
        """Load Vietnamese dictionary from underthesea's Viet74K"""
        try:
            import underthesea
            package_path = os.path.dirname(underthesea.__file__)
            dict_path = os.path.join(package_path, "corpus", "data", "Viet74K.txt")
            
            if os.path.exists(dict_path):
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            self._words.add(word)
                            # Also add individual words from compound words
                            for w in word.split():
                                self._words.add(w)
                print(f"[VietnameseDictionary] Loaded {len(self._words)} words from Viet74K")
            else:
                print("[VietnameseDictionary] Viet74K.txt not found, using fallback")
                self._load_fallback()
        except ImportError:
            print("[VietnameseDictionary] underthesea not installed, using fallback")
            self._load_fallback()
    
    def _load_fallback(self):
        """Fallback common Vietnamese words"""
        common_words = {
            # Pronouns
            "tôi", "tao", "mình", "ta", "chúng", "họ", "nó", "anh", "chị", "em", "ông", "bà",
            "cô", "chú", "dì", "cậu", "mợ", "thím", "bác", "con", "cháu", "bạn", "các",
            # Common verbs
            "là", "có", "được", "làm", "đi", "đến", "về", "ra", "vào", "lên", "xuống",
            "ăn", "uống", "ngủ", "thức", "chạy", "đứng", "ngồi", "nằm", "bay", "bơi",
            "nói", "nghe", "nhìn", "thấy", "biết", "hiểu", "nghĩ", "nhớ", "quên", "yêu",
            "ghét", "thích", "muốn", "cần", "phải", "nên", "hãy", "đừng", "chớ",
            # Common adjectives
            "tốt", "xấu", "đẹp", "xinh", "hay", "dở", "giỏi", "kém", "nhanh", "chậm",
            "cao", "thấp", "to", "nhỏ", "lớn", "bé", "dài", "ngắn", "rộng", "hẹp",
            # Common nouns
            "người", "nhà", "trường", "lớp", "bài", "sách", "vở", "bút", "xe", "đường",
            "nước", "cơm", "thịt", "cá", "rau", "quả", "hoa", "cây", "chó", "mèo",
            # Conjunctions and particles
            "và", "với", "của", "cho", "từ", "trong", "ngoài", "trên", "dưới",
            "trước", "sau", "giữa", "bên", "cạnh", "gần", "xa", "vì", "nên", "nhưng",
            "mà", "thì", "nếu", "khi", "lúc", "để", "mặc", "dù", "tuy",
            # Question words
            "gì", "nào", "đâu", "sao", "tại", "bao", "mấy", "ai", "thế",
            # Numbers
            "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
            # Time
            "ngày", "đêm", "sáng", "trưa", "chiều", "tối", "hôm", "mai", "qua", "nay",
            # Particles
            "này", "kia", "đó", "đây", "ấy", "ạ", "à", "ừ", "ờ", "ơi", "nhé", "nha",
            "luôn", "vẫn", "cứ", "đang", "sẽ", "đã", "vừa", "mới", "lại", "nữa",
            "thôi", "vậy", "quá", "lắm", "rất", "cực", "siêu", "khá", "hơi",
        }
        self._words.update(common_words)
    
    def contains(self, word):
        """Check if word is in Vietnamese dictionary"""
        return word.lower() in self._words
    
    def add_word(self, word):
        """Add a word to the dictionary"""
        self._words.add(word.lower())
    
    def get_all_words(self):
        """Get all words in dictionary"""
        return self._words.copy()
    
    def __len__(self):
        return len(self._words)


class TeencodeManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Teencode Manager - Quản lý & Quét từ lạ")
        self.root.geometry("1300x850")
        
        self.json_path = os.path.join(os.path.dirname(__file__), "teencode.json")
        self.ignored_words_path = os.path.join(os.path.dirname(__file__), "ignored_words.json")
        
        self.data = self._load_data()
        self.vn_dict = VietnameseDictionary.get_instance()
        self.ignored_words = self._load_ignored_words()
        self.unknown_words = {}  # {word: count}
        self.selected_files = []
        
        self._create_ui()
        self._refresh_treeview()
    
    def _load_data(self):
        """Load teencode.json"""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_data(self):
        """Save data to teencode.json"""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
    
    def _load_ignored_words(self):
        """Load ignored words"""
        if os.path.exists(self.ignored_words_path):
            with open(self.ignored_words_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    
    def _save_ignored_words(self):
        """Save ignored words"""
        with open(self.ignored_words_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.ignored_words), f, ensure_ascii=False, indent=4)
    
    def _get_all_teencode_variants(self):
        """Get all teencode variants (to exclude from unknown words)"""
        variants = set()
        for standard, variant_list in self.data.items():
            variants.add(standard.lower())
            for v in variant_list:
                variants.add(v.lower())
        return variants
    
    def _create_ui(self):
        """Create the UI with tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Teencode Manager (original)
        self.tab_manager = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_manager, text="📚 Quản lý Teencode")
        self._create_manager_tab()
        
        # Tab 2: File Scanner
        self.tab_scanner = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_scanner, text="🔍 Quét từ lạ từ File")
        self._create_scanner_tab()
        
        # Tab 3: Leet Speak Generator
        self.tab_leet = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_leet, text="🔤 Sinh biến thể (Leet)")
        self._create_leet_tab()
    
    def _create_manager_tab(self):
        """Create the manager tab UI"""
        # Main frame
        main_frame = ttk.Frame(self.tab_manager, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Left Panel: Treeview ===
        left_frame = ttk.LabelFrame(main_frame, text="📚 Danh sách Teencode", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Search
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_frame, text="🔍 Tìm kiếm:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self._refresh_treeview())
        ttk.Entry(search_frame, textvariable=self.search_var, width=30).pack(side=tk.LEFT, padx=5)
        
        # Treeview
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, columns=("standard", "variants"), show="headings", height=15)
        self.tree.heading("standard", text="Từ chuẩn")
        self.tree.heading("variants", text="Các biến thể (teencode)")
        self.tree.column("standard", width=150)
        self.tree.column("variants", width=300)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        
        # === Right Panel: Edit Form ===
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Add/Edit form
        form_frame = ttk.LabelFrame(right_frame, text="✏️ Thêm / Sửa", padding="10")
        form_frame.pack(fill=tk.BOTH, pady=(0, 10), expand=True)
        
        # Từ chuẩn
        ttk.Label(form_frame, text="Từ chuẩn:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.standard_var = tk.StringVar()
        self.standard_var.trace('w', self._on_standard_changed)
        self.standard_entry = ttk.Entry(form_frame, width=25, textvariable=self.standard_var)
        self.standard_entry.grid(row=0, column=1, columnspan=2, pady=2, padx=5, sticky=tk.W)
        self._typing_from_selection = False  # Flag để phân biệt nhập tay vs chọn từ tree
        
        # Danh sách biến thể
        ttk.Label(form_frame, text="Danh sách biến thể:").grid(row=1, column=0, sticky=tk.NW, pady=2)
        
        variants_frame = ttk.Frame(form_frame)
        variants_frame.grid(row=1, column=1, columnspan=2, pady=2, padx=5, sticky=tk.NSEW)
        
        self.variants_listbox = tk.Listbox(variants_frame, height=6, width=20, selectmode=tk.SINGLE)
        variants_scrollbar = ttk.Scrollbar(variants_frame, orient=tk.VERTICAL, command=self.variants_listbox.yview)
        self.variants_listbox.configure(yscrollcommand=variants_scrollbar.set)
        self.variants_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        variants_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Thêm biến thể mới
        add_variant_frame = ttk.Frame(form_frame)
        add_variant_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky=tk.W)
        
        ttk.Label(add_variant_frame, text="Thêm biến thể:").pack(side=tk.LEFT)
        self.new_variant_entry = ttk.Entry(add_variant_frame, width=15)
        self.new_variant_entry.pack(side=tk.LEFT, padx=5)
        self.new_variant_entry.bind('<Return>', lambda e: self._add_variant())
        ttk.Button(add_variant_frame, text="➕", width=3, command=self._add_variant).pack(side=tk.LEFT)
        ttk.Button(add_variant_frame, text="🗑️", width=3, command=self._remove_variant).pack(side=tk.LEFT, padx=2)
        
        # Main action buttons
        btn_frame = ttk.Frame(form_frame)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="➕ Thêm từ mới", command=self._add_entry).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="💾 Cập nhật", command=self._update_entry).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑️ Xóa từ", command=self._delete_entry).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🔄 Làm mới", command=self._clear_form).pack(side=tk.LEFT, padx=2)
        
        # === Converter Test ===
        convert_frame = ttk.LabelFrame(right_frame, text="🔄 Thử nghiệm Convert", padding="10")
        convert_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(convert_frame, text="Nhập câu cần convert:").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(convert_frame, height=4, width=40)
        self.input_text.pack(fill=tk.X, pady=5)
        
        ttk.Button(convert_frame, text="🚀 Convert", command=self._convert_text).pack(pady=5)
        
        ttk.Label(convert_frame, text="Kết quả:").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(convert_frame, height=4, width=40)
        self.output_text.pack(fill=tk.X, pady=5)
        
        # Stats
        self.stats_label = ttk.Label(right_frame, text="")
        self.stats_label.pack(pady=5)
        self._update_stats()
    
    def _refresh_treeview(self):
        """Refresh the treeview with current data"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        search_term = self.search_var.get().lower()
        
        for standard, variants in self.data.items():
            variants_str = ", ".join(variants)
            if search_term:
                if search_term in standard.lower() or search_term in variants_str.lower():
                    self.tree.insert("", tk.END, values=(standard, variants_str))
            else:
                self.tree.insert("", tk.END, values=(standard, variants_str))
        
        self._update_stats()
    
    def _on_select(self, event):
        """Handle treeview selection"""
        selected = self.tree.selection()
        if selected:
            item = self.tree.item(selected[0])
            standard = item['values'][0]
            
            # Set flag to prevent _on_standard_changed from clearing variants
            self._typing_from_selection = True
            self.standard_var.set(standard)
            self._typing_from_selection = False
            
            # Load variants into listbox
            self.variants_listbox.delete(0, tk.END)
            if standard in self.data:
                for variant in self.data[standard]:
                    self.variants_listbox.insert(tk.END, variant)
    
    def _on_standard_changed(self, *args):
        """Handle when user types in standard entry"""
        if self._typing_from_selection:
            return  # Skip if this change is from tree selection
        
        standard = self.standard_var.get().strip()
        
        # Clear listbox first
        self.variants_listbox.delete(0, tk.END)
        
        # If this word exists in data, load its variants
        if standard in self.data:
            for variant in self.data[standard]:
                self.variants_listbox.insert(tk.END, variant)
    
    def _add_variant(self):
        """Add a variant to the listbox"""
        variant = self.new_variant_entry.get().strip()
        if not variant:
            return
        
        # Check if already exists
        current_variants = list(self.variants_listbox.get(0, tk.END))
        if variant.lower() in [v.lower() for v in current_variants]:
            messagebox.showwarning("Cảnh báo", f"Biến thể '{variant}' đã tồn tại!")
            return
        
        self.variants_listbox.insert(tk.END, variant)
        self.new_variant_entry.delete(0, tk.END)
    
    def _remove_variant(self):
        """Remove selected variant from listbox"""
        selected = self.variants_listbox.curselection()
        if selected:
            self.variants_listbox.delete(selected[0])
        else:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn biến thể cần xóa!")
    
    def _get_variants_list(self):
        """Get all variants from listbox"""
        return list(self.variants_listbox.get(0, tk.END))
    
    def _add_entry(self):
        """Add new entry"""
        standard = self.standard_entry.get().strip()
        variants = self._get_variants_list()
        
        if not standard:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập từ chuẩn!")
            return
        
        if not variants:
            messagebox.showwarning("Cảnh báo", "Vui lòng thêm ít nhất 1 biến thể!")
            return
        
        if standard in self.data:
            messagebox.showwarning("Cảnh báo", f"Từ '{standard}' đã tồn tại! Dùng 'Cập nhật' để sửa.")
            return
        
        self.data[standard] = variants
        self._save_data()
        self._refresh_treeview()
        self._clear_form()
        messagebox.showinfo("Thành công", f"Đã thêm '{standard}' với {len(variants)} biến thể!")
    
    def _update_entry(self):
        """Update existing entry"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một mục để cập nhật!")
            return
        
        old_standard = self.tree.item(selected[0])['values'][0]
        new_standard = self.standard_entry.get().strip()
        variants = self._get_variants_list()
        
        if not new_standard:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập từ chuẩn!")
            return
        
        if not variants:
            messagebox.showwarning("Cảnh báo", "Vui lòng thêm ít nhất 1 biến thể!")
            return
        
        # Remove old key if name changed
        if old_standard != new_standard and old_standard in self.data:
            del self.data[old_standard]
        
        self.data[new_standard] = variants
        self._save_data()
        self._refresh_treeview()
        messagebox.showinfo("Thành công", f"Đã cập nhật '{new_standard}'!")
    
    def _delete_entry(self):
        """Delete selected entry"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một mục để xóa!")
            return
        
        standard = self.tree.item(selected[0])['values'][0]
        
        if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa '{standard}'?"):
            if standard in self.data:
                del self.data[standard]
                self._save_data()
                self._refresh_treeview()
                self._clear_form()
                messagebox.showinfo("Thành công", f"Đã xóa '{standard}'!")
    
    def _clear_form(self):
        """Clear the form"""
        self._typing_from_selection = True  # Prevent auto-load when clearing
        self.standard_var.set("")
        self._typing_from_selection = False
        self.variants_listbox.delete(0, tk.END)
        self.new_variant_entry.delete(0, tk.END)
        for item in self.tree.selection():
            self.tree.selection_remove(item)
    
    def _convert_text(self):
        """Convert text using teencode dictionary"""
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập câu cần convert!")
            return
        
        # Build reverse dictionary
        reverse_dict = {}
        for standard, variants in self.data.items():
            for variant in variants:
                reverse_dict[variant.lower()] = standard.lower()
        
        # Convert
        words = input_text.split()
        converted_words = [reverse_dict.get(word.lower(), word) for word in words]
        result = ' '.join(converted_words)
        
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", result)
    
    def _update_stats(self):
        """Update statistics label"""
        total_words = len(self.data)
        total_variants = sum(len(v) for v in self.data.values())
        self.stats_label.config(text=f"📊 Tổng: {total_words} từ chuẩn | {total_variants} biến thể")
    
    # ==================== SCANNER TAB ====================
    
    def _create_scanner_tab(self):
        """Create file scanner tab"""
        main_frame = ttk.Frame(self.tab_scanner, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Top: File Selection ===
        file_frame = ttk.LabelFrame(main_frame, text="📁 Chọn file để quét", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X)
        
        ttk.Button(btn_row, text="📂 Chọn file CSV/TXT", command=self._select_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="🚀 Quét từ lạ", command=self._scan_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="🗑️ Xóa danh sách file", command=self._clear_files).pack(side=tk.LEFT, padx=2)
        
        # Info label
        info_label = ttk.Label(file_frame, text=f"📚 Từ điển tiếng Việt: {len(self.vn_dict)} từ", foreground="green")
        info_label.pack(side=tk.RIGHT, padx=10)
        
        # Selected files display
        self.files_listbox = tk.Listbox(file_frame, height=3)
        self.files_listbox.pack(fill=tk.X, pady=5)
        
        # Progress
        self.scan_progress = ttk.Progressbar(file_frame, mode='determinate')
        self.scan_progress.pack(fill=tk.X, pady=5)
        self.scan_status = ttk.Label(file_frame, text="Sẵn sàng quét...")
        self.scan_status.pack()
        
        # === Middle: Unknown Words List ===
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Unknown words list
        unknown_frame = ttk.LabelFrame(middle_frame, text="❓ Các từ lạ (không có trong từ điển TV & chưa có trong teencode)", padding="5")
        unknown_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Filter and sort
        filter_frame = ttk.Frame(unknown_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(filter_frame, text="🔍 Lọc:").pack(side=tk.LEFT)
        self.unknown_filter_var = tk.StringVar()
        self.unknown_filter_var.trace('w', lambda *args: self._refresh_unknown_list())
        ttk.Entry(filter_frame, textvariable=self.unknown_filter_var, width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_frame, text="Sắp xếp:").pack(side=tk.LEFT, padx=(10, 0))
        self.sort_var = tk.StringVar(value="count_desc")
        sort_combo = ttk.Combobox(filter_frame, textvariable=self.sort_var, width=15, state="readonly")
        sort_combo['values'] = ("count_desc", "count_asc", "alpha_asc", "alpha_desc")
        sort_combo.pack(side=tk.LEFT, padx=5)
        sort_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_unknown_list())
        
        # Treeview for unknown words
        unknown_tree_frame = ttk.Frame(unknown_frame)
        unknown_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.unknown_tree = ttk.Treeview(unknown_tree_frame, columns=("word", "count"), show="headings", height=15)
        self.unknown_tree.heading("word", text="Từ lạ")
        self.unknown_tree.heading("count", text="Số lần xuất hiện")
        self.unknown_tree.column("word", width=250)
        self.unknown_tree.column("count", width=120)
        
        unknown_scrollbar = ttk.Scrollbar(unknown_tree_frame, orient=tk.VERTICAL, command=self.unknown_tree.yview)
        self.unknown_tree.configure(yscrollcommand=unknown_scrollbar.set)
        
        self.unknown_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        unknown_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.unknown_tree.bind('<<TreeviewSelect>>', self._on_unknown_select)
        
        # Stats for unknown
        self.unknown_stats = ttk.Label(unknown_frame, text="📊 Chưa quét")
        self.unknown_stats.pack(pady=5)
        
        # Right: Quick action panel
        action_frame = ttk.LabelFrame(middle_frame, text="⚡ Xử lý nhanh", padding="10")
        action_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        action_frame.configure(width=380)
        
        # Selected word display
        ttk.Label(action_frame, text="Từ đang chọn:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.selected_unknown_var = tk.StringVar(value="(chưa chọn)")
        ttk.Label(action_frame, textvariable=self.selected_unknown_var, font=('Arial', 16), foreground='blue').pack(anchor=tk.W, pady=5)
        
        # Context display
        context_frame = ttk.LabelFrame(action_frame, text="📝 Ngữ cảnh (context)", padding="5")
        context_frame.pack(fill=tk.X, pady=5)
        self.context_text = scrolledtext.ScrolledText(context_frame, height=6, width=40, font=('Consolas', 10), wrap=tk.WORD)
        self.context_text.pack(fill=tk.X)
        self.context_text.config(state=tk.DISABLED)
        
        # Action buttons
        ttk.Separator(action_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Option 1: Add as teencode
        teen_frame = ttk.LabelFrame(action_frame, text="1️⃣ Đây là viết tắt → Thêm vào Teencode", padding="8")
        teen_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(teen_frame, text="Từ chuẩn (nghĩa gốc):").pack(anchor=tk.W)
        self.quick_standard_var = tk.StringVar()
        self.quick_standard_entry = ttk.Entry(teen_frame, textvariable=self.quick_standard_var, width=35, font=('Arial', 11))
        self.quick_standard_entry.pack(fill=tk.X, pady=2)
        self.quick_standard_entry.bind('<Return>', lambda e: self._quick_add_teencode())
        ttk.Button(teen_frame, text="✅ Thêm vào Teencode", command=self._quick_add_teencode).pack(pady=5)
        
        # Option 2: Add to Vietnamese dictionary
        vn_frame = ttk.LabelFrame(action_frame, text="2️⃣ Là từ tiếng Việt hợp lệ", padding="8")
        vn_frame.pack(fill=tk.X, pady=5)
        ttk.Label(vn_frame, text="(Thêm vào từ điển để không hiện lại)").pack(anchor=tk.W)
        ttk.Button(vn_frame, text="📖 Thêm vào từ điển TV", command=self._add_to_vn_dict).pack(pady=5)
        
        # Option 3: Ignore
        ignore_frame = ttk.LabelFrame(action_frame, text="3️⃣ Bỏ qua từ này", padding="8")
        ignore_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ignore_frame, text="(Từ không có nghĩa, rác, tiếng nước ngoài...)").pack(anchor=tk.W)
        ttk.Button(ignore_frame, text="🚫 Bỏ qua (không hiện lại)", command=self._ignore_word).pack(pady=5)
        
        # Bulk actions
        ttk.Separator(action_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        bulk_frame = ttk.LabelFrame(action_frame, text="🔧 Thao tác hàng loạt", padding="5")
        bulk_frame.pack(fill=tk.X, pady=5)
        
        bulk_btn_row = ttk.Frame(bulk_frame)
        bulk_btn_row.pack(fill=tk.X)
        ttk.Button(bulk_btn_row, text="🗑️ Xóa tất cả", command=self._clear_unknown).pack(side=tk.LEFT, padx=2)
        ttk.Button(bulk_btn_row, text="💾 Xuất file", command=self._export_unknown).pack(side=tk.LEFT, padx=2)
        ttk.Button(bulk_btn_row, text="🔄 Reset bỏ qua", command=self._clear_ignored_words).pack(side=tk.LEFT, padx=2)
    
    # ==================== SCANNER METHODS ====================
    
    def _select_files(self):
        """Select files to scan"""
        files = filedialog.askopenfilenames(
            title="Chọn file để quét",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if files:
            self.selected_files.extend(files)
            self._update_files_listbox()
    
    def _update_files_listbox(self):
        """Update files listbox display"""
        self.files_listbox.delete(0, tk.END)
        for f in self.selected_files:
            self.files_listbox.insert(tk.END, os.path.basename(f))
    
    def _clear_files(self):
        """Clear selected files"""
        self.selected_files = []
        self.files_listbox.delete(0, tk.END)
        self.scan_status.config(text="Đã xóa danh sách file")
    
    def _extract_words(self, text):
        """Extract words from text"""
        # Remove emoji codes like :la_hét:
        text = re.sub(r':[a-zA-Z_]+:', '', text)
        # Remove special chars, keep Vietnamese chars
        text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', ' ', text.lower())
        words = text.split()
        return [w for w in words if w]
    
    def _scan_files(self):
        """Scan files for unknown words"""
        if not self.selected_files:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file để quét!")
            return
        
        self.unknown_words = {}
        teencode_variants = self._get_all_teencode_variants()
        
        total_files = len(self.selected_files)
        self.scan_progress['maximum'] = total_files
        self.scan_progress['value'] = 0
        
        total_words_scanned = 0
        
        for i, filepath in enumerate(self.selected_files):
            self.scan_status.config(text=f"Đang quét: {os.path.basename(filepath)}...")
            self.root.update()
            
            try:
                count = self._scan_single_file(filepath, teencode_variants)
                total_words_scanned += count
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi đọc file {filepath}: {str(e)}")
            
            self.scan_progress['value'] = i + 1
            self.root.update()
        
        self.scan_status.config(text=f"✅ Hoàn thành! Quét {total_words_scanned} từ, tìm thấy {len(self.unknown_words)} từ lạ")
        self._refresh_unknown_list()
    
    def _scan_single_file(self, filepath, teencode_variants):
        """Scan a single file, return word count"""
        ext = os.path.splitext(filepath)[1].lower()
        word_count = 0
        
        texts = []
        if ext == '.csv':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    texts.extend(row)
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                texts = f.readlines()
        
        for text in texts:
            words = self._extract_words(text)
            word_count += len(words)
            
            for i, word in enumerate(words):
                word_lower = word.lower()
                
                # Skip if:
                # - Already in Vietnamese dict
                # - Already in teencode (standard or variant)
                # - In ignored words
                # - Is a number
                # - Too short (1 char) or too long (>20 chars)
                # - Contains underscore (compound words from tokenizer)
                if (self.vn_dict.contains(word_lower) or 
                    word_lower in teencode_variants or
                    word_lower in self.ignored_words or
                    word_lower.isdigit() or
                    len(word_lower) <= 1 or
                    len(word_lower) > 20 or
                    '_' in word_lower):
                    continue
                
                # Initialize word entry if not exists
                if word_lower not in self.unknown_words:
                    self.unknown_words[word_lower] = {"count": 0, "contexts": []}
                
                self.unknown_words[word_lower]["count"] += 1
                
                # Save context (3 words before and after), max 5 contexts per word
                if len(self.unknown_words[word_lower]["contexts"]) < 5:
                    start = max(0, i - 3)
                    end = min(len(words), i + 4)
                    context_words = words[start:end]
                    # Highlight the target word
                    target_idx = i - start
                    if 0 <= target_idx < len(context_words):
                        context_words[target_idx] = f"[{context_words[target_idx]}]"
                    context = " ".join(context_words)
                    if context not in self.unknown_words[word_lower]["contexts"]:
                        self.unknown_words[word_lower]["contexts"].append(context)
        
        return word_count
    
    def _refresh_unknown_list(self):
        """Refresh unknown words treeview"""
        for item in self.unknown_tree.get_children():
            self.unknown_tree.delete(item)
        
        filter_term = self.unknown_filter_var.get().lower()
        sort_mode = self.sort_var.get()
        
        # Filter
        items = [(word, data["count"]) for word, data in self.unknown_words.items() 
                 if filter_term in word.lower()]
        
        # Sort
        if sort_mode == "count_desc":
            items.sort(key=lambda x: x[1], reverse=True)
        elif sort_mode == "count_asc":
            items.sort(key=lambda x: x[1])
        elif sort_mode == "alpha_asc":
            items.sort(key=lambda x: x[0])
        else:  # alpha_desc
            items.sort(key=lambda x: x[0], reverse=True)
        
        for word, count in items:
            self.unknown_tree.insert("", tk.END, values=(word, count))
        
        self.unknown_stats.config(text=f"📊 Hiển thị: {len(items)} / {len(self.unknown_words)} từ lạ")
    
    def _on_unknown_select(self, event):
        """Handle unknown word selection"""
        selected = self.unknown_tree.selection()
        if selected:
            item = self.unknown_tree.item(selected[0])
            word = item['values'][0]
            self.selected_unknown_var.set(word)
            self.quick_standard_var.set("")
            self.quick_standard_entry.focus()
            
            # Display contexts
            self.context_text.config(state=tk.NORMAL)
            self.context_text.delete("1.0", tk.END)
            if word in self.unknown_words and self.unknown_words[word]["contexts"]:
                contexts = self.unknown_words[word]["contexts"]
                for i, ctx in enumerate(contexts, 1):
                    self.context_text.insert(tk.END, f"{i}. {ctx}\n")
            else:
                self.context_text.insert(tk.END, "(không có context)")
            self.context_text.config(state=tk.DISABLED)
    
    def _quick_add_teencode(self):
        """Quickly add selected unknown word as teencode variant"""
        variant = self.selected_unknown_var.get()
        standard = self.quick_standard_var.get().strip()
        
        if variant == "(chưa chọn)":
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một từ lạ!")
            return
        
        if not standard:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập từ chuẩn!")
            return
        
        # Add to teencode
        if standard in self.data:
            if variant.lower() not in [v.lower() for v in self.data[standard]]:
                self.data[standard].append(variant)
        else:
            self.data[standard] = [variant]
        
        self._save_data()
        self._refresh_treeview()
        
        # Remove from unknown and select next
        self._remove_current_and_select_next(variant)
        
        messagebox.showinfo("Thành công", f"Đã thêm '{variant}' → '{standard}' vào teencode!")
    
    def _add_to_vn_dict(self):
        """Add selected word to Vietnamese dictionary"""
        word = self.selected_unknown_var.get()
        
        if word == "(chưa chọn)":
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một từ lạ!")
            return
        
        self.vn_dict.add_word(word.lower())
        
        # Remove from unknown and select next
        self._remove_current_and_select_next(word)
    
    def _ignore_word(self):
        """Ignore selected word"""
        word = self.selected_unknown_var.get()
        
        if word == "(chưa chọn)":
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một từ lạ!")
            return
        
        self.ignored_words.add(word.lower())
        self._save_ignored_words()
        
        # Remove from unknown and select next
        self._remove_current_and_select_next(word)
    
    def _remove_current_and_select_next(self, word):
        """Remove word from unknown list and select next item"""
        if word.lower() in self.unknown_words:
            del self.unknown_words[word.lower()]
        
        # Get current selection index before refresh
        selected = self.unknown_tree.selection()
        current_index = 0
        if selected:
            current_index = self.unknown_tree.index(selected[0])
        
        self._refresh_unknown_list()
        
        # Select next item
        children = self.unknown_tree.get_children()
        if children:
            next_index = min(current_index, len(children) - 1)
            self.unknown_tree.selection_set(children[next_index])
            self.unknown_tree.focus(children[next_index])
            self.unknown_tree.see(children[next_index])
            # Trigger selection event
            self._on_unknown_select(None)
        else:
            self.selected_unknown_var.set("(chưa chọn)")
            self.quick_standard_var.set("")
            # Clear context
            self.context_text.config(state=tk.NORMAL)
            self.context_text.delete("1.0", tk.END)
            self.context_text.config(state=tk.DISABLED)
    
    def _clear_unknown(self):
        """Clear all unknown words"""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa tất cả từ lạ?"):
            self.unknown_words = {}
            self._refresh_unknown_list()
            self.selected_unknown_var.set("(chưa chọn)")
            # Clear context
            self.context_text.config(state=tk.NORMAL)
            self.context_text.delete("1.0", tk.END)
            self.context_text.config(state=tk.DISABLED)
    
    def _export_unknown(self):
        """Export unknown words to file"""
        if not self.unknown_words:
            messagebox.showwarning("Cảnh báo", "Không có từ lạ để xuất!")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                for word, data in sorted(self.unknown_words.items(), key=lambda x: x[1]["count"], reverse=True):
                    f.write(f"{word}\t{data['count']}\n")
                    for ctx in data["contexts"]:
                        f.write(f"  → {ctx}\n")
            messagebox.showinfo("Thành công", f"Đã xuất {len(self.unknown_words)} từ lạ!")
    
    def _clear_ignored_words(self):
        """Clear all ignored words"""
        if messagebox.askyesno("Xác nhận", f"Xóa {len(self.ignored_words)} từ đã bỏ qua?"):
            self.ignored_words = set()
            self._save_ignored_words()
            messagebox.showinfo("Thành công", "Đã reset danh sách từ bỏ qua!")
    
    # ==================== LEET SPEAK TAB ====================
    
    def _create_leet_tab(self):
        """Create Leet Speak Generator tab"""
        main_frame = ttk.Frame(self.tab_leet, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Left: Mapping Table ===
        left_frame = ttk.LabelFrame(main_frame, text="📋 Bảng ký hiệu thay thế", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Mapping treeview
        map_tree_frame = ttk.Frame(left_frame)
        map_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.leet_tree = ttk.Treeview(map_tree_frame, columns=("char", "replacements"), show="headings", height=20)
        self.leet_tree.heading("char", text="Ký tự gốc")
        self.leet_tree.heading("replacements", text="Các ký hiệu thay thế")
        self.leet_tree.column("char", width=80)
        self.leet_tree.column("replacements", width=250)
        
        leet_scrollbar = ttk.Scrollbar(map_tree_frame, orient=tk.VERTICAL, command=self.leet_tree.yview)
        self.leet_tree.configure(yscrollcommand=leet_scrollbar.set)
        
        self.leet_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        leet_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load default mappings
        self.leet_map = DEFAULT_LEET_MAP.copy()
        self._refresh_leet_tree()
        
        # Add/Edit mapping
        edit_frame = ttk.Frame(left_frame)
        edit_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(edit_frame, text="Ký tự:").pack(side=tk.LEFT)
        self.leet_char_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.leet_char_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(edit_frame, text="Thay thế (cách bởi dấu phẩy):").pack(side=tk.LEFT, padx=(10, 0))
        self.leet_replace_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.leet_replace_var, width=25).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(edit_frame, text="➕ Thêm/Sửa", command=self._add_leet_mapping).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_frame, text="🗑️ Xóa", command=self._delete_leet_mapping).pack(side=tk.LEFT, padx=2)
        
        self.leet_tree.bind('<<TreeviewSelect>>', self._on_leet_select)
        
        # === Right: Generator ===
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Input word
        input_frame = ttk.LabelFrame(right_frame, text="📝 Nhập từ chuẩn để sinh biến thể", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Từ chuẩn:").pack(anchor=tk.W)
        self.leet_input_var = tk.StringVar()
        input_entry = ttk.Entry(input_frame, textvariable=self.leet_input_var, font=('Arial', 14))
        input_entry.pack(fill=tk.X, pady=5)
        input_entry.bind('<Return>', lambda e: self._generate_leet_variants())
        
        # Options
        options_frame = ttk.Frame(input_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(options_frame, text="Số biến thể tối đa:").pack(side=tk.LEFT)
        self.max_variants_var = tk.StringVar(value="50")
        ttk.Spinbox(options_frame, from_=10, to=500, textvariable=self.max_variants_var, width=8).pack(side=tk.LEFT, padx=5)
        
        self.include_original_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Bao gồm từ gốc", variable=self.include_original_var).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(input_frame, text="🔄 Sinh biến thể", command=self._generate_leet_variants).pack(pady=5)
        
        # Generated variants
        variants_frame = ttk.LabelFrame(right_frame, text="📋 Các biến thể được sinh", padding="5")
        variants_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.leet_variants_text = scrolledtext.ScrolledText(variants_frame, height=12, font=('Consolas', 10))
        self.leet_variants_text.pack(fill=tk.BOTH, expand=True)
        
        self.leet_count_label = ttk.Label(variants_frame, text="")
        self.leet_count_label.pack()
        
        # Action buttons
        action_frame = ttk.LabelFrame(right_frame, text="⚡ Thêm vào Teencode", padding="10")
        action_frame.pack(fill=tk.X)
        
        btn_row1 = ttk.Frame(action_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row1, text="✅ Thêm TẤT CẢ biến thể vào Teencode", 
                   command=self._add_all_leet_to_teencode).pack(fill=tk.X)
        
        btn_row2 = ttk.Frame(action_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row2, text="📋 Copy danh sách", command=self._copy_leet_variants).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="🗑️ Xóa", command=lambda: self.leet_variants_text.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=2)
        
        # Quick common words
        quick_frame = ttk.LabelFrame(right_frame, text="🚀 Sinh nhanh cho từ phổ biến", padding="5")
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        
        common_words = ["không", "được", "biết", "thích", "yêu", "ghét", "tốt", "xấu", "đẹp", "hay"]
        
        for i in range(0, len(common_words), 5):
            row = ttk.Frame(quick_frame)
            row.pack(fill=tk.X, pady=1)
            for word in common_words[i:i+5]:
                ttk.Button(row, text=word, width=8, 
                          command=lambda w=word: self._quick_generate(w)).pack(side=tk.LEFT, padx=1)
    
    def _refresh_leet_tree(self):
        """Refresh leet mapping treeview"""
        for item in self.leet_tree.get_children():
            self.leet_tree.delete(item)
        
        for char in sorted(self.leet_map.keys()):
            replacements = ", ".join(self.leet_map[char])
            self.leet_tree.insert("", tk.END, values=(char, replacements))
    
    def _on_leet_select(self, event):
        """Handle leet tree selection"""
        selected = self.leet_tree.selection()
        if selected:
            item = self.leet_tree.item(selected[0])
            char = item['values'][0]
            replacements = item['values'][1]
            self.leet_char_var.set(char)
            self.leet_replace_var.set(replacements)
    
    def _add_leet_mapping(self):
        """Add or update leet mapping"""
        char = self.leet_char_var.get().strip().lower()
        replacements = self.leet_replace_var.get().strip()
        
        if not char or len(char) != 1:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đúng 1 ký tự!")
            return
        
        if not replacements:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập các ký hiệu thay thế!")
            return
        
        # Parse replacements
        replace_list = [r.strip() for r in replacements.split(',') if r.strip()]
        self.leet_map[char] = replace_list
        self._refresh_leet_tree()
    
    def _delete_leet_mapping(self):
        """Delete selected leet mapping"""
        selected = self.leet_tree.selection()
        if not selected:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một mapping để xóa!")
            return
        
        char = self.leet_tree.item(selected[0])['values'][0]
        if char in self.leet_map:
            del self.leet_map[char]
            self._refresh_leet_tree()
    
    def _generate_leet_variants(self):
        """Generate leet speak variants for a word"""
        word = self.leet_input_var.get().strip().lower()
        if not word:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập từ cần sinh biến thể!")
            return
        
        max_variants = int(self.max_variants_var.get())
        variants = self._generate_variants_for_word(word, max_variants)
        
        # Display
        self.leet_variants_text.delete("1.0", tk.END)
        
        if self.include_original_var.get():
            self.leet_variants_text.insert(tk.END, f"{word}\n")
        
        for v in variants:
            if v != word:  # Don't duplicate original
                self.leet_variants_text.insert(tk.END, f"{v}\n")
        
        total = len(variants) + (1 if self.include_original_var.get() and word not in variants else 0)
        self.leet_count_label.config(text=f"📊 Đã sinh: {total} biến thể")
    
    def _generate_variants_for_word(self, word, max_variants=50):
        """Generate all possible leet variants for a word"""
        # Build list of possible replacements for each character
        char_options = []
        for char in word:
            options = [char]  # Always include original
            if char in self.leet_map:
                options.extend(self.leet_map[char])
            char_options.append(options)
        
        # Calculate total combinations
        total_combinations = 1
        for opts in char_options:
            total_combinations *= len(opts)
        
        variants = set()
        
        if total_combinations <= max_variants:
            # Generate all combinations
            for combo in product(*char_options):
                variants.add(''.join(combo))
        else:
            # Generate random sample - prioritize single substitutions first
            variants.add(word)  # Original
            
            # Single substitutions (replace one char at a time)
            for i, char in enumerate(word):
                if char in self.leet_map:
                    for replacement in self.leet_map[char]:
                        new_word = word[:i] + replacement + word[i+1:]
                        variants.add(new_word)
                        if len(variants) >= max_variants:
                            break
                if len(variants) >= max_variants:
                    break
            
            # Double substitutions if still room
            if len(variants) < max_variants:
                for i, char1 in enumerate(word):
                    if char1 in self.leet_map:
                        for j, char2 in enumerate(word):
                            if j > i and char2 in self.leet_map:
                                for r1 in self.leet_map[char1][:2]:  # Limit to first 2
                                    for r2 in self.leet_map[char2][:2]:
                                        new_word = word[:i] + r1 + word[i+1:j] + r2 + word[j+1:]
                                        variants.add(new_word)
                                        if len(variants) >= max_variants:
                                            break
                                    if len(variants) >= max_variants:
                                        break
                                if len(variants) >= max_variants:
                                    break
                            if len(variants) >= max_variants:
                                break
                        if len(variants) >= max_variants:
                            break
        
        return list(variants)[:max_variants]
    
    def _add_all_leet_to_teencode(self):
        """Add all generated variants to teencode"""
        word = self.leet_input_var.get().strip().lower()
        if not word:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập từ chuẩn!")
            return
        
        variants_text = self.leet_variants_text.get("1.0", tk.END).strip()
        if not variants_text:
            messagebox.showwarning("Cảnh báo", "Chưa có biến thể nào được sinh!")
            return
        
        variants = [v.strip() for v in variants_text.split('\n') if v.strip() and v.strip() != word]
        
        if not variants:
            messagebox.showwarning("Cảnh báo", "Không có biến thể nào để thêm!")
            return
        
        # Add to teencode
        if word in self.data:
            existing = set(v.lower() for v in self.data[word])
            new_variants = [v for v in variants if v.lower() not in existing]
            self.data[word].extend(new_variants)
            added = len(new_variants)
        else:
            self.data[word] = variants
            added = len(variants)
        
        self._save_data()
        self._refresh_treeview()
        
        messagebox.showinfo("Thành công", f"Đã thêm {added} biến thể cho từ '{word}'!")
    
    def _copy_leet_variants(self):
        """Copy variants to clipboard"""
        variants = self.leet_variants_text.get("1.0", tk.END).strip()
        if variants:
            self.root.clipboard_clear()
            self.root.clipboard_append(variants)
            messagebox.showinfo("Thành công", "Đã copy vào clipboard!")
    
    def _quick_generate(self, word):
        """Quick generate for common word"""
        self.leet_input_var.set(word)
        self._generate_leet_variants()


if __name__ == "__main__":
    root = tk.Tk()
    app = TeencodeManager(root)
    root.mainloop()
