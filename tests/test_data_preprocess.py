import pandas as pd

from src.data_preprocess import (
    PreprocessConfig,
    build_resources,
    preprocess_df,
    preprocess_text,
    teencode_replace,
)


def test_teencode_replace_basic():
    m = {"tui": "tôi"}
    assert teencode_replace("tui", m) == "tôi"
    assert teencode_replace("Tui", m) == "Tôi"
    assert teencode_replace("TUI", m) == "TÔI"


def test_preprocess_text_no_vncorenlp_smoke():
    config = PreprocessConfig(
        enable_vncorenlp=False,
        require_vietnamese=False,
        len_threshold=0,
        space_threshold=0,
    )
    resources = build_resources(config)
    out = preprocess_text("Tui 🙂", config=config, resources=resources)
    assert isinstance(out, str)


def test_preprocess_df_filters_and_runs():
    df = pd.DataFrame({"text": ["ngắn", "đây là một câu tiếng Việt khá dài để qua lọc"]})
    config = PreprocessConfig(
        enable_vncorenlp=False,
        require_vietnamese=False,
        len_threshold=10,
        space_threshold=2,
    )
    resources = build_resources(config)
    out = preprocess_df(df, config=config, resources=resources, show_progress=False)
    assert len(out) == 1
    assert "text" in out.columns

