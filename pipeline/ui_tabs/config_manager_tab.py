import json
import os
import re

import customtkinter as ctk
from tkinter import messagebox, ttk


def _pipeline_dir() -> str:
    # ui_tabs/ is a subfolder of pipeline/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_config_tab(app):
    tab = app.tab_config
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(1, weight=1)

    # Header for selection
    header = ctk.CTkFrame(tab, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    ctk.CTkLabel(header, text="Đang mở file:").pack(side="left", padx=5)
    app.config_vars = ["abbreviations.json", "emoji_vi.json", "profanity_list.json", "nlp_pipeline/config.py"]
    app.config_dropdown = ctk.CTkOptionMenu(header, values=app.config_vars, command=app.on_config_file_change)
    app.config_dropdown.pack(side="left", padx=5)

    # Workspace Container
    app.config_workspace = ctk.CTkFrame(tab)
    app.config_workspace.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    app.config_workspace.grid_columnconfigure(0, weight=1)
    app.config_workspace.grid_rowconfigure(0, weight=1)

    # --- JSON Mapping Editor Frame ---
    app.json_editor_frame = ctk.CTkFrame(app.config_workspace, fg_color="transparent")
    app.json_editor_frame.grid_columnconfigure(0, weight=1)
    app.json_editor_frame.grid_rowconfigure(1, weight=1)

    # Top controls for JSON
    jf_top = ctk.CTkFrame(app.json_editor_frame, fg_color="transparent")
    jf_top.grid(row=0, column=0, sticky="ew", pady=(0, 5))

    ctk.CTkLabel(jf_top, text="Từ khóa:").pack(side="left", padx=2)
    app.entry_key = ctk.CTkEntry(jf_top, width=150)
    app.entry_key.pack(side="left", padx=2)

    ctk.CTkLabel(jf_top, text="Giá trị thay thế:").pack(side="left", padx=2)
    app.entry_val = ctk.CTkEntry(jf_top, width=200)
    app.entry_val.pack(side="left", padx=2)

    ctk.CTkButton(jf_top, text="Thêm/Sửa", width=80, command=app.add_or_update_json_entry).pack(side="left", padx=5)
    ctk.CTkButton(
        jf_top,
        text="Xóa",
        width=60,
        fg_color="red",
        hover_color="darkred",
        command=app.delete_json_entry,
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        jf_top,
        text="Lưu Cấu Hình (Disk)",
        width=120,
        fg_color="green",
        hover_color="darkgreen",
        command=app.save_json_file,
    ).pack(side="right", padx=5)

    # Treeview for JSON
    tr_frame = ctk.CTkFrame(app.json_editor_frame)
    tr_frame.grid(row=1, column=0, sticky="nsew")
    tr_frame.grid_columnconfigure(0, weight=1)
    tr_frame.grid_rowconfigure(0, weight=1)

    app.tree_json = ttk.Treeview(tr_frame, columns=("key", "val"), show="headings")
    app.tree_json.heading("key", text="Từ Khóa (Key)")
    app.tree_json.heading("val", text="Giá Trị Thay Thế (Value)")
    app.tree_json.column("key", width=200, anchor="w")
    app.tree_json.column("val", width=400, anchor="w")

    scroll_j = ttk.Scrollbar(tr_frame, orient="vertical", command=app.tree_json.yview)
    app.tree_json.configure(yscrollcommand=scroll_j.set)
    app.tree_json.grid(row=0, column=0, sticky="nsew")
    scroll_j.grid(row=0, column=1, sticky="ns")

    app.tree_json.bind("<<TreeviewSelect>>", app.on_json_tree_select)

    # Current active JSON data
    app.current_json_data = {}

    # --- Python Config Editor Frame ---
    app.py_editor_frame = ctk.CTkFrame(app.config_workspace, fg_color="transparent")
    app.py_editor_frame.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(app.py_editor_frame, text="Quản lý biến trong config.py", font=("Arial", 16, "bold")).grid(
        row=0, column=0, columnspan=2, pady=10, sticky="w"
    )

    # Dynamically build entry fields for known python configs
    app.py_config_entries = {}
    target_keys = [
        "MIN_CHAR_LENGTH",
        "MIN_VIETNAMESE_RATIO",
        "MAX_SPAM_REPEAT",
        "MAX_REPEAT_CHARS",
        "MAX_REPEAT_PUNCTUATION",
        "MAX_REPEAT_ICON_CHARS",
    ]

    row_idx = 1
    for k in target_keys:
        ctk.CTkLabel(app.py_editor_frame, text=k + ":").grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        e = ctk.CTkEntry(app.py_editor_frame, width=200)
        e.grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
        app.py_config_entries[k] = e
        row_idx += 1

    ctk.CTkButton(
        app.py_editor_frame,
        text="Lưu config.py",
        fg_color="green",
        hover_color="darkgreen",
        command=app.save_py_config,
    ).grid(row=row_idx, column=0, columnspan=2, pady=20)

    # Load initial
    app.on_config_file_change(app.config_vars[0])


def on_config_file_change(app, filename):
    if filename.endswith(".json"):
        app.py_editor_frame.grid_forget()
        app.json_editor_frame.grid(row=0, column=0, sticky="nsew")
        app.load_json_config(filename)
    else:
        app.json_editor_frame.grid_forget()
        app.py_editor_frame.grid(row=0, column=0, sticky="nsew")
        app.load_py_config(filename)


def _get_config_path(app, filename):
    base_dir = _pipeline_dir()
    if "nlp_pipeline" in filename:
        # The dropdown passes "nlp_pipeline/config.py"
        return os.path.join(base_dir, "nlp_pipeline", "config.py")
    else:
        # json map path is in mappings/
        return os.path.join(base_dir, "mappings", filename)


# --- JSON Helpers ---

def load_json_config(app, filename):
    path = app._get_config_path(filename)
    app.current_json_data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    app.current_json_data = data
                elif isinstance(data, list):
                    # For profanity list -> turn into dict mapping to True
                    for item in data:
                        app.current_json_data[str(item)] = ""
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc JSON: {e}")

    refresh_json_tree(app)
    app.entry_key.delete(0, "end")
    app.entry_val.delete(0, "end")


def refresh_json_tree(app):
    for item in app.tree_json.get_children():
        app.tree_json.delete(item)
    for k, v in app.current_json_data.items():
        app.tree_json.insert("", "end", values=(k, v))


def on_json_tree_select(app, event):
    selected = app.tree_json.selection()
    if selected:
        item = app.tree_json.item(selected[0])
        app.entry_key.delete(0, "end")
        app.entry_key.insert(0, item["values"][0])
        app.entry_val.delete(0, "end")
        # Handle empty values safely
        val = item["values"][1] if len(item["values"]) > 1 else ""
        app.entry_val.insert(0, val)


def add_or_update_json_entry(app):
    k = app.entry_key.get().strip()
    v = app.entry_val.get().strip()
    if not k:
        messagebox.showwarning("Nhắc nhở", "Từ khóa không được để trống")
        return
    app.current_json_data[k] = v
    refresh_json_tree(app)
    app.entry_key.delete(0, "end")
    app.entry_val.delete(0, "end")


def delete_json_entry(app):
    selected = app.tree_json.selection()
    if not selected:
        messagebox.showwarning("Nhắc nhở", "Hãy chọn 1 dòng để xóa")
        return
    item = app.tree_json.item(selected[0])
    k = str(item["values"][0])
    if k in app.current_json_data:
        del app.current_json_data[k]
        refresh_json_tree(app)
        app.entry_key.delete(0, "end")
        app.entry_val.delete(0, "end")


def save_json_file(app):
    filename = app.config_dropdown.get()
    path = app._get_config_path(filename)

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # If profanity list, we might want to save exactly as list
            if filename == "profanity_list.json":
                res = list(app.current_json_data.keys())
                json.dump(res, f, ensure_ascii=False, indent=4)
            else:
                json.dump(app.current_json_data, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("Thành công", f"Đã cập nhật file {filename}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể ghi file: {str(e)}")


# --- Python Config Helpers ---

def load_py_config(app, filename):
    path = app._get_config_path(filename)
    if not os.path.exists(path):
        messagebox.showwarning("Lỗi", f"Không tìm thấy file {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse logic
    for k, entry in app.py_config_entries.items():
        pattern = rf"^[ \t]*{k}[ \t]*=[ \t]*([0-9\.]+)"
        match = re.search(pattern, content, re.MULTILINE)
        entry.delete(0, "end")
        if match:
            entry.insert(0, match.group(1))


def save_py_config(app):
    filename = app.config_dropdown.get()
    path = app._get_config_path(filename)

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        for k, entry in app.py_config_entries.items():
            val = entry.get().strip()
            if not val:
                continue
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
