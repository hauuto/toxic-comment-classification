import customtkinter as ctk
from tkinter import messagebox


def _setup_pipeline_test_tab(app):
    tab = app.tab_pipeline_test
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(2, weight=1)

    header = ctk.CTkFrame(tab)
    header.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    header.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        header,
        text="Test Pipeline (không filter)",
        font=("Arial", 15, "bold"),
    ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))

    ctk.CTkLabel(
        header,
        text="Nhập text bên dưới và chạy qua Decoder → Normalizer → Segmentor. Kết quả là cleaned_text.",
        text_color="gray",
    ).grid(row=1, column=0, sticky="w", padx=10, pady=(2, 10))

    io_frame = ctk.CTkFrame(tab)
    io_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
    io_frame.grid_columnconfigure(0, weight=1)
    io_frame.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(io_frame, text="Input:").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
    ctk.CTkLabel(io_frame, text="Output:").grid(row=0, column=1, sticky="w", padx=10, pady=(10, 0))

    app.pipeline_test_input = ctk.CTkTextbox(io_frame, height=120)
    app.pipeline_test_input.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    app.pipeline_test_output = ctk.CTkTextbox(io_frame, height=120)
    app.pipeline_test_output.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
    app.pipeline_test_output.configure(state="disabled")

    action = ctk.CTkFrame(tab, fg_color="transparent")
    action.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
    action.grid_columnconfigure(0, weight=1)

    app.pipeline_test_run_btn = ctk.CTkButton(action, text="Chạy Pipeline", command=app.run_pipeline_test)
    app.pipeline_test_run_btn.grid(row=0, column=0, sticky="w")

    app.pipeline_test_status = ctk.CTkLabel(action, text="", text_color="gray")
    app.pipeline_test_status.grid(row=0, column=1, sticky="w", padx=(10, 0))


def run_pipeline_test(app):
    if not hasattr(app, "pipeline_test_input"):
        return

    text = (app.pipeline_test_input.get("0.0", "end") or "").strip()
    if not text:
        messagebox.showinfo("Thông báo", "Vui lòng nhập text để test.")
        return

    # Prefer shared VnCoreNLP if initialized; else fall back to underthesea/whitespace.
    backend = "vncorenlp" if getattr(app, "_shared_vncorenlp_segmentor", None) is not None else "underthesea"

    try:
        pre = app._get_preprocessor(backend)
    except Exception:
        backend = "whitespace"
        pre = app._get_preprocessor(backend)

    try:
        out = pre.process_comment(
            text,
            use_decoder=True,
            use_filter=False,
            use_normalizer=True,
            use_segmentor=True,
        )
        cleaned = out.get("cleaned_text", "")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không chạy được pipeline: {e}")
        return

    app.pipeline_test_output.configure(state="normal")
    app.pipeline_test_output.delete("0.0", "end")
    app.pipeline_test_output.insert("0.0", cleaned)
    app.pipeline_test_output.configure(state="disabled")
    app.pipeline_test_status.configure(text=f"Backend: {backend}")
