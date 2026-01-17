import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class SpamKeywordManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Keyword Manager")
        self.root.geometry("700x600")
        
        self.json_path = None
        self.data = {}  # Format: {"category": ["keyword1", "keyword2", ...]}
        
        self._create_ui()
    
    def _load_data(self):
        """Load JSON file"""
        if self.json_path and os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_data(self):
        """Save data to JSON file"""
        if not self.json_path:
            messagebox.showwarning("Cảnh báo", "Chưa chọn file JSON!")
            return
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
    
    def _create_ui(self):
        """Create the UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === File Selection Bar ===
        file_frame = ttk.LabelFrame(main_frame, text="📂 Chọn file JSON", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path_var = tk.StringVar(value="Chưa chọn file...")
        ttk.Label(file_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="📁 Mở file", command=self._open_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="➕ Tạo mới", command=self._new_file).pack(side=tk.LEFT, padx=5)
        
        # === Content Frame ===
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Left Panel: Categories ===
        left_frame = ttk.LabelFrame(content_frame, text="📁 Danh mục", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Category listbox
        self.category_listbox = tk.Listbox(left_frame, height=15, width=20, selectmode=tk.SINGLE)
        cat_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.category_listbox.yview)
        self.category_listbox.configure(yscrollcommand=cat_scrollbar.set)
        self.category_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.category_listbox.bind('<<ListboxSelect>>', self._on_category_select)
        
        # Category buttons
        cat_btn_frame = ttk.Frame(left_frame)
        cat_btn_frame.pack(fill=tk.X, pady=5)
        
        self.new_cat_entry = ttk.Entry(cat_btn_frame, width=15)
        self.new_cat_entry.pack(side=tk.LEFT, padx=2)
        self.new_cat_entry.bind('<Return>', lambda e: self._add_category())
        ttk.Button(cat_btn_frame, text="➕", width=3, command=self._add_category).pack(side=tk.LEFT)
        ttk.Button(cat_btn_frame, text="🗑️", width=3, command=self._delete_category).pack(side=tk.LEFT, padx=2)
        
        # === Right Panel: Keywords ===
        right_frame = ttk.LabelFrame(content_frame, text="🏷️ Từ khóa spam", padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Current category label
        self.current_cat_var = tk.StringVar(value="Chưa chọn danh mục")
        ttk.Label(right_frame, textvariable=self.current_cat_var, font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Keywords listbox
        kw_frame = ttk.Frame(right_frame)
        kw_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.keyword_listbox = tk.Listbox(kw_frame, height=12, width=30, selectmode=tk.EXTENDED)
        kw_scrollbar = ttk.Scrollbar(kw_frame, orient=tk.VERTICAL, command=self.keyword_listbox.yview)
        self.keyword_listbox.configure(yscrollcommand=kw_scrollbar.set)
        self.keyword_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        kw_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add keyword
        add_kw_frame = ttk.Frame(right_frame)
        add_kw_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(add_kw_frame, text="Thêm từ khóa:").pack(side=tk.LEFT)
        self.new_kw_entry = ttk.Entry(add_kw_frame, width=20)
        self.new_kw_entry.pack(side=tk.LEFT, padx=5)
        self.new_kw_entry.bind('<Return>', lambda e: self._add_keyword())
        ttk.Button(add_kw_frame, text="➕", width=3, command=self._add_keyword).pack(side=tk.LEFT)
        ttk.Button(add_kw_frame, text="🗑️", width=3, command=self._delete_keyword).pack(side=tk.LEFT, padx=2)
        
        # Bulk add
        bulk_frame = ttk.LabelFrame(right_frame, text="📝 Thêm nhiều (mỗi dòng 1 từ)", padding="5")
        bulk_frame.pack(fill=tk.X, pady=5)
        
        self.bulk_text = tk.Text(bulk_frame, height=4, width=30)
        self.bulk_text.pack(fill=tk.X)
        ttk.Button(bulk_frame, text="➕ Thêm tất cả", command=self._bulk_add_keywords).pack(pady=5)
        
        # Stats
        self.stats_label = ttk.Label(main_frame, text="")
        self.stats_label.pack(pady=5)
        self._update_stats()
    
    def _open_file(self):
        """Open existing JSON file"""
        file_path = filedialog.askopenfilename(
            title="Chọn file JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(__file__)
        )
        if file_path:
            self.json_path = file_path
            self.file_path_var.set(file_path)
            self.data = self._load_data()
            self._refresh_categories()
            self.keyword_listbox.delete(0, tk.END)
            self.current_cat_var.set("Chưa chọn danh mục")
            self.root.title(f"Spam Keyword Manager - {os.path.basename(file_path)}")
            self._update_stats()
    
    def _new_file(self):
        """Create new JSON file"""
        file_path = filedialog.asksaveasfilename(
            title="Tạo file JSON mới",
            filetypes=[("JSON files", "*.json")],
            defaultextension=".json",
            initialdir=os.path.dirname(__file__)
        )
        if file_path:
            self.json_path = file_path
            self.file_path_var.set(file_path)
            self.data = {}
            self._save_data()
            self._refresh_categories()
            self.keyword_listbox.delete(0, tk.END)
            self.current_cat_var.set("Chưa chọn danh mục")
            self.root.title(f"Spam Keyword Manager - {os.path.basename(file_path)}")
            messagebox.showinfo("Thành công", f"Đã tạo file mới: {os.path.basename(file_path)}")
            self._update_stats()
    
    def _refresh_categories(self, keep_selection=False):
        """Refresh category listbox"""
        # Remember current selection
        selected_category = None
        if keep_selection:
            selected_category = self._get_selected_category()
        
        self.category_listbox.delete(0, tk.END)
        select_index = None
        for i, category in enumerate(self.data.keys()):
            count = len(self.data[category])
            self.category_listbox.insert(tk.END, f"{category} ({count})")
            if category == selected_category:
                select_index = i
        
        # Restore selection
        if select_index is not None:
            self.category_listbox.selection_set(select_index)
            self.category_listbox.see(select_index)
    
    def _on_category_select(self, event):
        """Handle category selection"""
        selected = self.category_listbox.curselection()
        if selected:
            # Get category name (remove count suffix)
            cat_text = self.category_listbox.get(selected[0])
            category = cat_text.rsplit(' (', 1)[0]
            
            self.current_cat_var.set(f"📁 {category}")
            
            # Load keywords
            self.keyword_listbox.delete(0, tk.END)
            if category in self.data:
                for keyword in self.data[category]:
                    self.keyword_listbox.insert(tk.END, keyword)
    
    def _get_selected_category(self):
        """Get currently selected category name"""
        selected = self.category_listbox.curselection()
        if selected:
            cat_text = self.category_listbox.get(selected[0])
            return cat_text.rsplit(' (', 1)[0]
        return None
    
    def _add_category(self):
        """Add new category"""
        category = self.new_cat_entry.get().strip()
        if not category:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập tên danh mục!")
            return
        
        if category in self.data:
            messagebox.showwarning("Cảnh báo", f"Danh mục '{category}' đã tồn tại!")
            return
        
        self.data[category] = []
        self._save_data()
        self._refresh_categories()
        self.new_cat_entry.delete(0, tk.END)
        self._update_stats()
        messagebox.showinfo("Thành công", f"Đã thêm danh mục '{category}'!")
    
    def _delete_category(self):
        """Delete selected category"""
        category = self._get_selected_category()
        if not category:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn danh mục cần xóa!")
            return
        
        if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa danh mục '{category}' và tất cả từ khóa trong đó?"):
            del self.data[category]
            self._save_data()
            self._refresh_categories()
            self.keyword_listbox.delete(0, tk.END)
            self.current_cat_var.set("Chưa chọn danh mục")
            self._update_stats()
            messagebox.showinfo("Thành công", f"Đã xóa danh mục '{category}'!")
    
    def _add_keyword(self):
        """Add keyword to current category"""
        category = self._get_selected_category()
        if not category:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn danh mục trước!")
            return
        
        keyword = self.new_kw_entry.get().strip()
        if not keyword:
            return
        
        if keyword in self.data[category]:
            messagebox.showwarning("Cảnh báo", f"Từ khóa '{keyword}' đã tồn tại!")
            return
        
        self.data[category].append(keyword)
        self._save_data()
        self.keyword_listbox.insert(tk.END, keyword)
        self.new_kw_entry.delete(0, tk.END)
        self._refresh_categories(keep_selection=True)  # Update count
        self._update_stats()
    
    def _delete_keyword(self):
        """Delete selected keywords"""
        category = self._get_selected_category()
        if not category:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn danh mục trước!")
            return
        
        selected = self.keyword_listbox.curselection()
        if not selected:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn từ khóa cần xóa!")
            return
        
        # Delete from end to start to maintain indices
        for index in reversed(selected):
            keyword = self.keyword_listbox.get(index)
            self.data[category].remove(keyword)
            self.keyword_listbox.delete(index)
        
        self._save_data()
        self._refresh_categories(keep_selection=True)  # Update count
        self._update_stats()
    
    def _bulk_add_keywords(self):
        """Add multiple keywords at once"""
        category = self._get_selected_category()
        if not category:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn danh mục trước!")
            return
        
        text = self.bulk_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập từ khóa!")
            return
        
        keywords = [kw.strip() for kw in text.split('\n') if kw.strip()]
        added = 0
        for kw in keywords:
            if kw not in self.data[category]:
                self.data[category].append(kw)
                self.keyword_listbox.insert(tk.END, kw)
                added += 1
        
        self._save_data()
        self._refresh_categories(keep_selection=True)
        self.bulk_text.delete("1.0", tk.END)
        self._update_stats()
        messagebox.showinfo("Thành công", f"Đã thêm {added}/{len(keywords)} từ khóa!")
    
    def _update_stats(self):
        """Update statistics"""
        total_categories = len(self.data)
        total_keywords = sum(len(v) for v in self.data.values())
        self.stats_label.config(text=f"📊 Tổng: {total_categories} danh mục | {total_keywords} từ khóa")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpamKeywordManager(root)
    root.mainloop()
