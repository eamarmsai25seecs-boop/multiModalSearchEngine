import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
from PIL import Image
import json
import time

# ── Page config
st.set_page_config(
    page_title="MultiModal Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 {
    color: #fff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #a78bfa;
    font-size: 1.05rem;
    margin: 0.5rem 0 0 0;
}

.result-card {
    background: #1e1b3a;
    border: 1px solid #3d3a6b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #7c3aed; }
.result-rank {
    display: inline-block;
    background: #7c3aed;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}
.result-score {
    float: right;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.result-text { color: #e2e8f0; font-size: 0.95rem; line-height: 1.6; }
.stat-box {
    background: #1e1b3a;
    border: 1px solid #3d3a6b;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-number { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.stat-label  { color: #64748b; font-size: 0.8rem; }
.mode-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}
.mode-text  { background: #1d4ed8; color: #bfdbfe; }
.mode-image { background: #065f46; color: #a7f3d0; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading search engine…")
def load_searcher():
    from core.search import MultiModalSearcher
    searcher = MultiModalSearcher(
        index_dir=str(ROOT / "indexes"),
        expand_queries=True,
    )
    if (ROOT / "indexes" / "metadata.pkl").exists():
        searcher.load()
    return searcher


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    search_mode = st.selectbox(
        "Search Mode",
        ["Text → Text", "Text → Images", "Image → Text", "Image → Images"],
        index=0,
    )

    top_k = st.slider("Results to retrieve", 5, 50, 10, step=5)

    st.markdown("---")
    st.markdown("### Index Status")
    index_dir = ROOT / "indexes"
    text_idx = index_dir / "text.faiss"
    img_idx  = index_dir / "image.faiss"
    st.markdown(f"{'✅' if text_idx.exists() else '❌'} Text index")
    st.markdown(f"{'✅' if img_idx.exists() else '❌'} Image index")

    if not text_idx.exists() and not img_idx.exists():
        st.warning("No indexes found. Run the setup commands first.")


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔍 MultiModal Neural Search Engine</h1>
    <p>Dense Retrieval · CLIP · Vector Similarity Search</p>
</div>
""", unsafe_allow_html=True)


# ── Main search area ───────────────────────────────────────────────────────────

mode_map = {
    "Text → Text":     "text_to_text",
    "Text → Images":   "text_to_image",
    "Image → Text":    "image_to_text",
    "Image → Images": "image_to_image",
    "Hybrid":          "hybrid",
}
mode_key = mode_map[search_mode]
is_image_query = "Image →" in search_mode

col_input, col_btn = st.columns([5, 1])

query_text = ""
query_image = None

if not is_image_query:
    with col_input:
        query_text = st.text_input(
            "Enter your query",
            placeholder="e.g. cat sitting on a laptop",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")
else:
    uploaded = st.file_uploader(
        "Upload a query image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )
    search_clicked = st.button("🔍 Search with Image", type="primary")
    if uploaded:
        query_image = Image.open(uploaded).convert("RGB")
        st.image(query_image, width=300, caption="Query Image")


# ── Run search ─────────────────────────────────────────────────────────────────

if search_clicked:
    has_query = (not is_image_query and query_text.strip()) or (is_image_query and query_image)

    if not has_query:
        st.warning("Please enter a query or upload an image.")
    else:
        try:
            searcher = load_searcher()
            query_input = query_text if not is_image_query else query_image

            with st.spinner("Searching…"):
                t0 = time.perf_counter()
                # Ensure reranker is disabled in the searcher object
                searcher.set_reranker("none")
                results = searcher.search(query=query_input, mode=mode_key, top_k=top_k)
                search_time = time.perf_counter() - t0

            # ── Stats row ──
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="stat-box"><div class="stat-number">{len(results)}</div>
                <div class="stat-label">Results</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="stat-box"><div class="stat-number">{search_time*1000:.0f}ms</div>
                <div class="stat-label">Search Time</div></div>""", unsafe_allow_html=True)
            with c3:
                text_count = sum(1 for r in results if r.modality == "text")
                st.markdown(f"""<div class="stat-box"><div class="stat-number">{text_count}</div>
                <div class="stat-label">Text Results</div></div>""", unsafe_allow_html=True)
            with c4:
                img_count = sum(1 for r in results if r.modality == "image")
                st.markdown(f"""<div class="stat-box"><div class="stat-number">{img_count}</div>
                <div class="stat-label">Image Results</div></div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"### Results for: *{query_text or 'image query'}*")

            if not results:
                st.info("No results found.")
            else:
                for r in results:
                    modality_badge = (
                        '<span class="mode-badge mode-text">TEXT</span>'
                        if r.modality == "text"
                        else '<span class="mode-badge mode-image">IMAGE</span>'
                    )

                    if r.modality == "text":
                        text_snippet = r.metadata.get("text", "") or r.metadata.get("caption", "")
                        st.markdown(f"""
                        <div class="result-card">
                            <span class="result-rank">#{r.rank}</span>{modality_badge}
                            <span class="result-score">score: {r.score:.4f}</span>
                            <div class="result-text">{text_snippet}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    else:  # image result
                        img_path = r.metadata.get("image_path", "")
                        caption = r.metadata.get("caption", r.metadata.get("text", "No caption"))
                        st.markdown(f"""
                        <div class="result-card">
                            <span class="result-rank">#{r.rank}</span>{modality_badge}
                            <span class="result-score">score: {r.score:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if img_path and Path(img_path).exists():
                            try:
                                img = Image.open(img_path)
                                st.image(img, width=350, caption=caption)
                            except Exception:
                                st.warning(f"Could not load image file: {img_path}")
                        else:
                            st.info(f"Caption: {caption}")

        except Exception as e:
            st.error(f"Search error: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem;'>"
    "MultiModal Neural Search Engine + CLIP + FAISS · "
    "MS COCO &amp; Flickr30k"
    "</div>",
    unsafe_allow_html=True,
)