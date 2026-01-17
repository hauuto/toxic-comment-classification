import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class TeencodeManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Teencode Manager")
        self.root.geometry("950x700")
        
        self.json_path = os.path.join(os.path.dirname(__file__), "teencode.json")
        self.data = self._load_data()
        
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
    
    def _create_ui(self):
        """Create the UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
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


if __name__ == "__main__":
    root = tk.Tk()
    app = TeencodeManager(root)
    root.mainloop()
