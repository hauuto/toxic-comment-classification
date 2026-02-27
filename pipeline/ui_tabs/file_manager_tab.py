import glob
import os
import time

import customtkinter as ctk
from tkinter import messagebox, ttk


def _setup_file_manager_tab(app):
    tab = app.tab_files
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(1, weight=1)

    # Header
    header = ctk.CTkFrame(tab, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    ctk.CTkButton(header, text="Làm mới danh sách", command=app.refresh_file_list).pack(side="left", padx=5)
    ctk.CTkButton(header, text="Mở thư mục hiện tại", command=app.open_current_folder).pack(side="left", padx=5)
    ctk.CTkButton(header, text="Xóa File Chọn", command=app.delete_selected_file, fg_color="red").pack(side="right", padx=5)

    # Treeview for files
    tf = ctk.CTkFrame(tab)
    tf.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    tf.grid_columnconfigure(0, weight=1)
    tf.grid_rowconfigure(0, weight=1)

    app.tree_files = ttk.Treeview(tf, columns=("filename", "size", "mtime"), show="headings")
    app.tree_files.heading("filename", text="Tên File")
    app.tree_files.heading("size", text="Kích thước (KB)")
    app.tree_files.heading("mtime", text="Thời gian sửa đổi")
    app.tree_files.column("filename", width=300)
    app.tree_files.column("size", width=100, anchor="e")
    app.tree_files.column("mtime", width=150)

    scrollbar_f = ttk.Scrollbar(tf, orient="vertical", command=app.tree_files.yview)
    app.tree_files.configure(yscrollcommand=scrollbar_f.set)

    app.tree_files.grid(row=0, column=0, sticky="nsew")
    scrollbar_f.grid(row=0, column=1, sticky="ns")

    # Initial load
    app.refresh_file_list()


def refresh_file_list(app):
    for item in app.tree_files.get_children():
        app.tree_files.delete(item)

    csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))

    for f in sorted(csv_files, key=os.path.getmtime, reverse=True):
        name = os.path.basename(f)
        size_kb = f"{os.path.getsize(f) / 1024:.1f}"
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f)))
        app.tree_files.insert("", "end", values=(name, size_kb, mtime))


def delete_selected_file(app):
    selected = app.tree_files.selection()
    if not selected:
        messagebox.showwarning("Nhắc nhở", "Bạn chưa chọn file cần xóa")
        return

    item = app.tree_files.item(selected[0])
    filename = item["values"][0]
    full_path = os.path.join(os.getcwd(), filename)

    if messagebox.askyesno("Xác nhận", f"Bạn có chắc chắn muốn xóa vĩnh viễn file:\n{filename}?"):
        try:
            os.remove(full_path)
            app.refresh_file_list()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xóa file: {e}")


def open_current_folder(app):
    folder = os.getcwd()
    try:
        os.startfile(folder)
    except AttributeError:
        os.system(f'explorer "{folder}"')
