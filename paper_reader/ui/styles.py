"""Streamlit CSS."""

APP_CSS = """
<style>
:root {
    --app-bg: #f5f8fa;
    --surface: #ffffff;
    --surface-accent: #e8f3f3;
    --border: #dbe5e8;
    --border-strong: #c8d5da;
    --text: #17202a;
    --muted: #667085;
    --accent: #236f73;
    --accent-dark: #19585c;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(35, 111, 115, 0.09), transparent 30rem),
        linear-gradient(180deg, var(--app-bg) 0%, #ffffff 24rem);
    color: var(--text);
}
.block-container { max-width: 1680px; padding: 2rem 2.2rem 3.2rem; }
h1 { font-size: 2rem !important; margin-bottom: 0.2rem !important; }
h2 { font-size: 1.32rem !important; }
h3 { font-size: 1.12rem !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef6f6 0%, #f8fbfb 42%, #ffffff 100%);
    border-right: 1px solid var(--border);
}
.stTextInput input, [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border-strong) !important;
    border-radius: 8px !important;
    min-height: 2.45rem;
}
.stButton > button, .stDownloadButton > button {
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    box-shadow: none;
    font-weight: 620;
    min-height: 2.45rem;
}
[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}
[data-testid="stDataFrame"], [data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}
</style>
"""
