# benchup_topics.py
# BenchUp topics (v2.3)
# - Correct OpenAlex field→domain mapping (4 domains)
# - Domain colors updated; single legend row above both plots
# - Search also matches city; no NameError on no-match
# - Clean match label "Name (City, Country)" (no extra parentheses)
# - Avg/min/max pubs rounded; topics "count" as int
# - Fields sort: SI / Count / Fixed domain order (alphabetical in-domain)
# - Benchmark results: primary ranking by shared topics (desc)
# - Keep sortable table and add a separate clickable OpenAlex link column
# - New filters: exclude selected country; restrict to Europe

import streamlit as st
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime

# -------- Page & light mode ----------
st.set_page_config(page_title="BenchUp topics", layout="wide")
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] { background: #ffffff !important; color: #111 !important; }
    h1, h2, h3, h4, h5, h6 { color: #111 !important; }
    .red-big { color:#d62728; font-size: 2.0rem; font-weight:700; }
    .note { font-size:0.92rem; color:#333; }
    .legend-row { display:flex; gap:16px; flex-wrap:wrap; margin: 4px 0 8px 0; }
    .legend-item { display:flex; align-items:center; gap:8px; }
    .legend-swatch { width:14px; height:14px; border-radius:2px; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "institutions_master.parquet"

# -------- Exact OpenAlex field → domain mapping ----------
FIELDS_TO_DOMAIN = {
    # 1) PHYSICAL SCIENCES
    "Chemical Engineering": "Physical Sciences",
    "Chemistry": "Physical Sciences",
    "Computer Science": "Physical Sciences",
    "Earth and Planetary Sciences": "Physical Sciences",
    "Energy": "Physical Sciences",
    "Engineering": "Physical Sciences",
    "Materials Science": "Physical Sciences",
    "Mathematics": "Physical Sciences",
    "Physics and Astronomy": "Physical Sciences",
    # 2) LIFE SCIENCES
    "Agricultural and Biological Sciences": "Life Sciences",
    "Biochemistry, Genetics and Molecular Biology": "Life Sciences",
    "Environmental Science": "Life Sciences",
    "Immunology and Microbiology": "Life Sciences",
    # 3) HEALTH SCIENCES
    "Dentistry": "Health Sciences",
    "Health Professions": "Health Sciences",
    "Medicine": "Health Sciences",
    "Neuroscience": "Health Sciences",
    "Nursing": "Health Sciences",
    "Pharmacology, Toxicology and Pharmaceutics": "Health Sciences",
    "Veterinary": "Health Sciences",
    # 4) SOCIAL SCIENCES
    "Arts and Humanities": "Social Sciences",
    "Business, Management and Accounting": "Social Sciences",
    "Decision Sciences": "Social Sciences",
    "Economics, Econometrics and Finance": "Social Sciences",
    "Psychology": "Social Sciences",
    "Social Sciences": "Social Sciences",
}
DOMAIN_NAMES = ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences", "Other"]

# -------- Domain colors (as requested) ----------
DOMAIN_COLORS = {
    "Health Sciences": "#F85C32",
    "Life Sciences": "#0CA750",
    "Physical Sciences": "#8190FF",
    "Social Sciences": "#FFCB3A",
    "Other": "#7f7f7f",
}

# Fixed domain order for the "Fixed domain order" sort
DOMAIN_FIXED_ORDER = ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences", "Other"]

# -------- Helpers ----------

def slugify_name(name: str) -> str:
    s = unidecode(str(name))
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s

def ts_now() -> str:
    # day-month-year+hour+minute, e.g. 240920251040
    return datetime.now().strftime("%d%m%Y%H%M")

def norm(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unidecode(str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def parse_fields_blob(blob: str):
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["name", "count", "share", "si"])
    parts = [p.strip() for p in str(blob).split("|")]
    rows = []
    for p in parts:
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", p)
        if not m:
            continue
        name = m.group(1).strip()
        inside = [x.strip() for x in re.split(r"[;,]", m.group(2))]
        count = to_int_safe(inside[0]) if len(inside) > 0 else np.nan
        share = to_float_safe(inside[1]) if len(inside) > 1 else np.nan
        si    = to_float_safe(inside[2]) if len(inside) > 2 else np.nan
        rows.append((name, count, share, si))
    return pd.DataFrame(rows, columns=["name", "count", "share", "si"])

def parse_topics_blob(blob: str):
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["name", "count", "share"])
    parts = [p.strip() for p in str(blob).split("|")]
    rows = []
    for p in parts:
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", p)
        if not m:
            continue
        name = m.group(1).strip()
        inside = [x.strip() for x in re.split(r"[;,]", m.group(2))]
        count = to_int_safe(inside[0]) if len(inside) > 0 else np.nan
        share = to_float_safe(inside[1]) if len(inside) > 1 else np.nan
        rows.append((name, count, share))
    return pd.DataFrame(rows, columns=["name", "count", "share"])

def to_int_safe(x):
    try: return int(str(x).replace(",", "").strip())
    except: return np.nan

def to_float_safe(x):
    try: return float(str(x).replace("%", "").replace(",", ".").strip())
    except: return np.nan

def estimate_domain(field_name: str) -> str:
    # exact mapping first; fallback "Other"
    return FIELDS_TO_DOMAIN.get(str(field_name).strip(), "Other")

def domain_color(field_name: str) -> str:
    return DOMAIN_COLORS.get(estimate_domain(field_name), DOMAIN_COLORS["Other"])

def build_search_index(df):
    def row_key(r):
        parts = [r.get("display_name", "")]
        if r.get("acronyms"): parts.append(r.get("acronyms"))
        if r.get("alternatives"): parts.append(r.get("alternatives"))
        if r.get("city"): parts.append(r.get("city"))
        return norm(" | ".join([str(p) for p in parts if str(p).strip()]))
    df = df.copy()
    df["__searchkey"] = df.apply(row_key, axis=1)
    return df

def avg_pubs_per_year(row):
    total = (to_int_safe(row.get("articles_2020_24")) or 0) \
            + (to_int_safe(row.get("book_chapters_2020_24")) or 0) \
            + (to_int_safe(row.get("books_2020_24")) or 0)
    return total / 5.0

def cosine_sim(vec_a, vec_b):
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

def build_vector(df_fields, all_keys, kind="si"):
    if kind == "share":
        d = {r["name"]: (r["share"] if not pd.isna(r["share"]) else 0.0) for _, r in df_fields.iterrows()}
    else:
        d = {r["name"]: (r["si"] if not pd.isna(r["si"]) else 0.0) for _, r in df_fields.iterrows()}
    return [d.get(k, 0.0) for k in all_keys]

def sort_fields_df(df_f: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df_f.empty: return df_f
    df = df_f.copy()
    if mode == "SI (desc)":
        return df.sort_values("si", ascending=False)
    if mode == "Count (desc)":
        return df.sort_values("count", ascending=False)
    # Fixed domain order, alphabetical within each domain
    df["domain"] = df["name"].apply(estimate_domain)
    df["__dom_rank"] = df["domain"].apply(lambda d: DOMAIN_FIXED_ORDER.index(d) if d in DOMAIN_FIXED_ORDER else len(DOMAIN_FIXED_ORDER))
    df = df.sort_values(["__dom_rank","name"], ascending=[True, True]).drop(columns=["__dom_rank"])
    return df

# static-threshold coloring for topics ratio (%)
def color_topics_count(val):
    # New discrete thresholds (percent, 0–5) with interpolation in-between
    # 0%:#FFFFFF, 1%:#e2e2e2, 2%:#a8a8a8, 3%:#727272, 4%:#404040, 5%:#141414
    thresholds = [0, 1, 2, 3, 4, 5]
    colors = ["#FFFFFF", "#e2e2e2", "#a8a8a8", "#727272", "#404040", "#141414"]

    if pd.isna(val):
        return "background-color: #FFFFFF; color: black"

    # Cap above 5% at the 5% color
    if val >= thresholds[-1]:
        return "background-color: #141414; color: white"

    # Find interval and interpolate color
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if lo <= val < hi:
            # linear interpolation between colors[i] and colors[i+1]
            c1 = np.array([int(colors[i][j:j+2], 16) for j in (1, 3, 5)])
            c2 = np.array([int(colors[i + 1][j:j+2], 16) for j in (1, 3, 5)])
            ratio = (val - lo) / (hi - lo) if hi > lo else 0.0
            c = (c1 * (1 - ratio) + c2 * ratio).astype(int)
            hex_color = '#%02x%02x%02x' % tuple(c)

            # Font color: black up to and including 2%, white from 3%+
            text_color = "black" if val <= 2 else "white"
            return f"background-color: {hex_color}; color: {text_color}"

    # Fallback (shouldn't hit)
    return "background-color: #FFFFFF; color: black"

# -------- Load data ----------
@st.cache_data(show_spinner=True)
def load_master():
    if not DATA_PATH.exists():
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()
    df = pd.read_parquet(str(DATA_PATH))
    needed = [
        "openalex_id", "display_name", "acronyms", "alternatives",
        "type", "country", "city",
        "articles_2020_24", "book_chapters_2020_24", "books_2020_24",
        "primary_fields", "fields",
        "primary_topics_top100", "topics_top100"
    ]
    cols = [c for c in needed if c in df.columns]
    df = df[cols].copy()
    df["avg_pubs_per_year"] = df.apply(avg_pubs_per_year, axis=1)
    df = build_search_index(df)
    return df

df = load_master()

# -------- Title ----------
st.markdown(
    """
    <h1 style="margin-bottom:0.2rem;font-weight:700;">
      Bench<span style="color:#d62728;">Up</span> <em style="font-weight:300;">topics</em>
    </h1>
    <div class="note">OpenAlex-native benchmarking by topics and thematic shape</div>
    """,
    unsafe_allow_html=True
)

# dataset size for baseline
N_DATASET = int((df["avg_pubs_per_year"] >= 100).sum())

st.markdown(
    f"""
    **Data source & snapshot** — BenchUp topics uses only OpenAlex data (via the API), from a snapshot taken on **September 13, 2025** and now stored locally.  
    The dataset contains **{N_DATASET:,} institutions worldwide** present in OpenAlex (with a ROR ID) and publishing on average **≥ 100 works per year (2020–2024)** — works include articles (incl. conference communications), chapters, and books ([latest OpenAlex types](https://docs.openalex.org/api-entities/works/work-object#type)).  
    **Disclaimer:** The data in BenchUp topics should be used as a starting point, not a definitive analysis. While institutions may share similarities based on these indicators, meaningful comparisons require contextual and qualitative assessment. BenchUp helps identify potential international benchmarks but does not provide absolute answers.
    """,
    unsafe_allow_html=True,
)

# -------- Search ----------
q = st.text_input("Find your institution (name, acronym, alternatives, city)", placeholder="e.g., CNRS Paris, University of Milan, MIT ...")

# make sure chosen exists even if no results
chosen = None

if q:
    nq = norm(q)
    m_starts = df[df["__searchkey"].str.startswith(nq)].copy()
    m_contains = df[df["__searchkey"].str.contains(nq)].copy()
    results = pd.concat([m_starts, m_contains]).drop_duplicates(subset=["openalex_id"])
    results = results.sort_values(["display_name"]).head(200)
else:
    results = df.head(0)

if len(results) == 0 and q:
    st.info("No match. Try fewer words or official acronym or the city name.")
else:
    st.write(f"Matches: {len(results)}")

    def label_for_row(r):
        name = r.display_name
        city = (r.city or "").strip()
        country = (r.country or "").strip()
        if city and country:
            return f"{name} ({city}, {country})"
        if country:
            return f"{name} ({country})"
        return name

    options = ["—"] + [label_for_row(r) for _, r in results.iterrows()]
    sel = st.selectbox("Select institution", options=options, index=0)
    if sel != "—":
        idx = int(results.index[[label_for_row(r) for _, r in results.iterrows()].index(sel)])
        chosen = results.loc[idx]

if chosen is not None:
    st.markdown("---")

    # For filenames
    inst_slug = slugify_name(chosen.display_name)

    # ===== Basic profile =====
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.subheader(chosen.display_name)
        st.write(f"**Type:** {chosen.type or '—'}")
        st.write(f"**Location:** {chosen.city or '—'}, {chosen.country or '—'}")

        avg = int(round(chosen.avg_pubs_per_year))
        st.markdown(f"<div class='red-big'>{avg:,}</div>", unsafe_allow_html=True)
        st.markdown(
            """<div class='note'>
            <strong>Average publications per year (2020–2024)</strong><br/>
            Publications include articles, book chapters, and books.
            </div>""",
            unsafe_allow_html=True
        )

    with c2:
        # ---- doc-type pie with non-overlapping labels + legend below ----
        arts = to_int_safe(chosen.articles_2020_24) or 0
        chaps = to_int_safe(chosen.book_chapters_2020_24) or 0
        books = to_int_safe(chosen.books_2020_24) or 0

        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        vals = [arts, chaps, books]
        labels = ["Articles", "Book chapters", "Books"]
        vals2, labels2 = [], []
        for v, l in zip(vals, labels):
            if v and v > 0:
                vals2.append(v); labels2.append(l)

        if sum(vals2) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            wedges, _ = ax.pie(
                vals2,
                startangle=90,
                labels=None,
                wedgeprops=dict(linewidth=1, edgecolor="white"),
                normalize=True,
            )
            total = float(sum(vals2))
            percents = [100.0 * v / total for v in vals2]

            # radii for label placement
            r_far = 1.35   # <1%
            r_out = 1.15   # 1–10%
            r_in  = 0.70   # >=10%

            for w, pct in zip(wedges, percents):
                ang = np.deg2rad((w.theta2 + w.theta1) / 2.0)
                x, y = np.cos(ang), np.sin(ang)
                if pct < 1.0:
                    ax.annotate(
                        f"{pct:.1f}%",
                        xy=(x, y),
                        xytext=(x * r_far, y * r_far),
                        textcoords="data",
                        ha="center", va="center",
                        fontsize=10,
                        arrowprops=dict(arrowstyle="-", lw=0.7),
                        clip_on=False,
                    )
                elif pct < 10.0:
                    ax.annotate(
                        f"{pct:.1f}%",
                        xy=(x, y),
                        xytext=(x * r_out, y * r_out),
                        textcoords="data",
                        ha="center", va="center",
                        fontsize=10,
                        arrowprops=dict(arrowstyle="-", lw=0.6),
                        clip_on=False,
                    )
                else:
                    ax.text(x * r_in, y * r_in, f"{pct:.1f}%", ha="center", va="center", fontsize=10)

            # Legend below
            fig.subplots_adjust(bottom=0.26)
            ax.legend(
                wedges, labels2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.10),
                frameon=False,
                ncol=3,
            )

        ax.axis("equal")
        st.pyplot(fig, use_container_width=False)

    with c3:
        st.write("**2020–24 totals**")
        st.metric("Articles", f"{arts:,}")
        st.metric("Chapters", f"{chaps:,}")
        st.metric("Books", f"{books:,}")

    # --- Global toggle to drive all visuals & benchmarking ---
    use_primary = st.toggle(
        "Use primary fields & topics",
        value=False,
        help="When ON, all visuals, topics, and benchmarking use PRIMARY fields/topics only."
    )

    # ===== Thematic profile =====
    st.markdown("### Thematic profile")

    # default sort = Fixed domain order
    sort_mode = st.radio(
        "Sort fields by",
        options=["SI (desc)", "Count (desc)", "Fixed domain order"],
        index=2,
        horizontal=True
    )

    # One legend row for both plots
    legend_items = "".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:{DOMAIN_COLORS[d]};"></span>{d}</div>'
        for d in ["Health Sciences","Life Sciences","Physical Sciences","Social Sciences"]
    )
    st.markdown(f'<div class="legend-row">{legend_items}</div>', unsafe_allow_html=True)

    # SI explanation
    st.markdown(
        f"""
        **What is SI?** The Specialization Index (SI) compares an institution’s thematic weight to the dataset average.  
        Example: *if an institution has an SI of 2.4 in Immunology, its relative share in that field is 2.4× the average institution in this dataset.*  
        Baseline: all **{N_DATASET:,} institutions** in this snapshot (world, ≥100 works/year between 2020–24, with ROR ID).
        """
    )

    df_fields_sel   = parse_fields_blob(chosen.primary_fields if use_primary else chosen.fields)
    topics_blob_sel = chosen.primary_topics_top100 if use_primary else chosen.topics_top100

    def plot_fields_share(df_f, title, host):
        with host:
            st.markdown(f"**{title}**")
            if df_f.empty:
                st.info("No field data"); return

            # Same sorting logic as SI plots
            df_f = sort_fields_df(df_f, sort_mode)

            # Prepare data
            df_f = df_f.copy()
            df_f["ratio_%"] = pd.to_numeric(df_f["share"], errors="coerce").fillna(0.0) * 100.0
            df_f["count"] = pd.to_numeric(df_f["count"], errors="coerce").fillna(0).astype(int)

            y = np.arange(len(df_f))
            colors = [domain_color(n) for n in df_f["name"]]

            fig, ax = plt.subplots(figsize=(6.8, 0.48*len(df_f)+1))
            ax.barh(y, df_f["ratio_%"], color=colors, alpha=0.85, zorder=3)

            # Fixed pixel gutter on the left for counts (same as SI plots)
            left_pad_px = 80
            offset_px   = 6
            xmax = max(1.0, float(df_f["ratio_%"].max() or 0) * 1.15)
            ax.set_xlim(0, xmax)

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = ax.get_window_extent(renderer=renderer)
            ax_width_px = bb.width
            data_per_px = (xmax - 0.0) / ax_width_px
            left_pad_data = left_pad_px * data_per_px
            offset_data   = offset_px   * data_per_px

            # Extend x-range to include left gutter space
            ax.set_xlim(-left_pad_data, xmax)

            # counts (font +1), same placement logic as SI
            for yi, cnt in enumerate(df_f["count"]):
                ax.text(
                    -left_pad_data + offset_data, yi,
                    f"{cnt:,}".replace(",", " "),
                    va="center", ha="left", fontsize=9, color="#444"
                )

            ax.set_xlabel("Share (%)")
            ax.set_yticks(y)
            ax.set_yticklabels(df_f["name"])
            ax.invert_yaxis()
            ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)

            st.pyplot(fig, use_container_width=True)

            # Optional CSV export for % view
            out = df_f[["name","count","share","ratio_%"]].copy()
            st.download_button(
                "Download fields % CSV",
                data=out.to_csv(index=False, encoding='utf-8-sig'),
                file_name=f"{inst_slug}_{'primary_fields_share' if 'primary' in title.lower() else 'fields_share'}_{ts_now()}.csv",
                mime="text/csv"
            )

    def plot_fields(df_f, title, host):
        with host:
            st.markdown(f"**{title}**")
            if df_f.empty:
                st.info("No field data"); return
            df_f = sort_fields_df(df_f, sort_mode)
            y = np.arange(len(df_f))
            sizes = np.clip((df_f["count"].fillna(0).astype(float) / (df_f["count"].max() or 1)) * 3000, 50, 3000)
            colors = [domain_color(n) for n in df_f["name"]]

            fig, ax = plt.subplots(figsize=(6.8, 0.48*len(df_f)+1))
            ax.scatter(df_f["si"], y, s=sizes, c=colors, alpha=0.85, edgecolors="none", zorder=3)
            ax.axvline(1.0, color="#444", linestyle="--", linewidth=1, zorder=1)

            # Fixed pixel gutter on the left for counts
            left_pad_px = 80
            offset_px   = 6
            xmax = max(1.1, float(df_f["si"].max() or 0) * 1.15)
            ax.set_xlim(0, xmax)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = ax.get_window_extent(renderer=renderer)
            ax_width_px = bb.width
            data_per_px = (xmax - 0.0) / ax_width_px
            left_pad_data = left_pad_px * data_per_px
            offset_data   = offset_px   * data_per_px
            ax.set_xlim(-left_pad_data, xmax)

            # counts (font +1)
            for yi, cnt in enumerate(df_f["count"].fillna(0).astype(int)):
                ax.text(-left_pad_data + offset_data, yi, f"{cnt:,}".replace(",", " "),
                        va="center", ha="left", fontsize=9, color="#444")

            ax.set_xlabel("SI (Specialization Index)")
            ax.set_yticks(y)
            ax.set_yticklabels(df_f["name"])
            ax.invert_yaxis()
            ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)

            st.pyplot(fig, use_container_width=True)

            out = df_f[["name","count","share","si"]].copy()
            st.download_button(
                "Download fields CSV",
                data=out.to_csv(index=False,encoding='utf-8-sig'),
                file_name=f"{inst_slug}_{'primary_fields' if 'primary' in title.lower() else 'fields'}_{ts_now()}.csv",
                mime="text/csv"
            )

    # --- NEW: One pair only: left = % share, right = SI (driven by the toggle)
    cp1, cp2 = st.columns(2)
    plot_fields_share(df_fields_sel, f"Fields (%) ({'primary' if use_primary else 'all'})", cp1)
    plot_fields(df_fields_sel,       f"Fields (SI) ({'primary' if use_primary else 'all'})", cp2)

    # Topics tables with your discrete coloring; counts as int
    def show_topics_table(blob, title, host):
        with host:
            st.markdown(f"**{title}**")
            df_topics = parse_topics_blob(blob)
            if df_topics.empty:
                st.info("No topic data"); return
            df_topics["ratio_%"] = (pd.to_numeric(df_topics["share"], errors="coerce").fillna(0.0) * 100.0)
            df_topics["count"] = pd.to_numeric(df_topics["count"], errors="coerce").fillna(0).astype(int)
            df_topics = df_topics.sort_values(["count","ratio_%"], ascending=False).reset_index(drop=True)
            df_topics.index = df_topics.index + 1
            styled = (
                df_topics[["name","count","ratio_%"]]
                .style.format({"ratio_%": "{:.2f}"})
                .applymap(color_topics_count, subset=["ratio_%"])
            )
            st.dataframe(styled, use_container_width=True)
            st.download_button(
                "Download topics CSV",
                data=df_topics.to_csv(index=True,encoding='utf-8-sig'),
                file_name=f"{inst_slug}_{'top100_primary_topics' if 'primary' in title.lower() else 'top100_topics'}_{ts_now()}.csv",
                mime="text/csv"
            )

    show_topics_table(
        topics_blob_sel,
        f"Topics ({'primary' if use_primary else 'all'}, top 100)",
        st.container()
    )

    # ===== Benchmarking =====
    st.markdown("---")
    st.header("Benchmarking")

    default_center = int(round(max(100.0, chosen.avg_pubs_per_year)))
    min_default = int(round(max(100.0, default_center - 300)))
    max_default = int(round(default_center + 300))

    cA, cB = st.columns(2)
    with cA:
        min_shared_topics = st.slider("Minimum shared topics among the top 100*", 1, 50, 15)
        min_sim_share = st.slider("Similarity (Fields % proximity)**", 0.0, 1.0, 0.90, 0.01)
    with cB:
        st.markdown(f"<div style='color:#d62728; font-weight:700;'>Selected avg: {default_center:,} pubs/yr</div>", unsafe_allow_html=True)
        min_pubs = st.number_input("Min average pubs/year", min_value=100, value=int(min_default), step=25)
        max_pubs = st.number_input("Max average pubs/year", min_value=100, value=int(max_default), step=25)

    st.caption(
        "*Shared topics are counted among the top-100 topics lists of both institutions."
        "**Fields % proximity uses cosine similarity (on relative shares across fields."
    )

    allowed_types = sorted([t for t in df["type"].dropna().unique()])
    type_filter = st.multiselect("Institution type (optional)", allowed_types, default=[])

    # Geo filters
    exclude_same_country = st.checkbox("Exclude institutions from the same country", value=False)
    EUROPE = {
        "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina",
        "Bulgaria","Croatia","Cyprus","Czechia","Denmark","Estonia","Finland","France","Georgia","Germany",
        "Greece","Hungary","Iceland","Ireland","Italy","Kosovo","Latvia","Liechtenstein","Lithuania",
        "Luxembourg","Malta","Moldova","Monaco","Montenegro","Netherlands","North Macedonia","Norway",
        "Poland","Portugal","Romania","Russia","Serbia","Slovakia","Slovenia","Spain","Sweden","Switzerland",
        "Turkey","Ukraine","United Kingdom","Faroe Islands"
    }
    restrict_europe = st.checkbox("Restrict results to Europe", value=False)

    # Target sets & vectors
    target_topics_df = parse_topics_blob(chosen.primary_topics_top100 if use_primary else chosen.topics_top100)
    target_topics_set = set(target_topics_df["name"].tolist())
    target_fields_df = parse_fields_blob(chosen.primary_fields if use_primary else chosen.fields)

    @st.cache_data(show_spinner=False)
    def global_field_keys(df_all: pd.DataFrame, col: str):
        keys = set()
        for s in df_all[col].fillna("").tolist():
            tmp = parse_fields_blob(s)
            keys.update(tmp["name"].tolist())
        return sorted(keys)

    field_col = "primary_fields" if use_primary else "fields"
    ALL_KEYS = global_field_keys(df, field_col)
    target_vec_share = build_vector(target_fields_df, ALL_KEYS, kind="share")
    target_vec_si    = build_vector(target_fields_df, ALL_KEYS, kind="si")

    def row_passes_basic_filters(r):
        apy = r["avg_pubs_per_year"]
        if apy < min_pubs or apy > max_pubs: return False
        if type_filter and r.get("type") not in set(type_filter): return False
        if exclude_same_country and (r.get("country") == chosen.country): return False
        if restrict_europe and (r.get("country") not in EUROPE): return False
        return True

    # --- Filtered OpenAlex works URLs (2020–2024; articles/chapters/books) ---
    TYPES_Q = "type:types/article|types/book-chapter|types/book"
    YEARS_Q = "publication_year:2020-2024"

    def oa_works_url(inst_id):
        iid = str(inst_id).lower()
        return f"https://openalex.org/works?page=1&filter=authorships.institutions.lineage:{iid},{TYPES_Q},{YEARS_Q}"

    def oa_copubs_url(a_id, b_id):
        a = str(a_id).lower()
        b = str(b_id).lower()
        return f"https://openalex.org/works?page=1&filter=authorships.institutions.lineage:{a},{TYPES_Q},{YEARS_Q},authorships.institutions.lineage:{b}"

    chosen_id = str(chosen.openalex_id).lower()

    rows = []
    for _, r in df.iterrows():
        if r.openalex_id == chosen.openalex_id:
            continue
        if not row_passes_basic_filters(r):
            continue

        # shared topics
        df_topics_r = parse_topics_blob(r.primary_topics_top100 if use_primary else r.topics_top100)
        shared_names = target_topics_set.intersection(set(df_topics_r["name"].tolist()))
        if len(shared_names) < min_shared_topics:
            continue

        # field vectors
        df_fields_r = parse_fields_blob(r.primary_fields if use_primary else r.fields)
        vec_share_r = build_vector(df_fields_r, ALL_KEYS, kind="share")

        sim_share = cosine_sim(target_vec_share, vec_share_r)
        if sim_share < min_sim_share:
            continue

        # TOPICS DETAIL (robust to NaNs)
        df_topics_r = df_topics_r.copy()
        df_topics_r["count"] = pd.to_numeric(df_topics_r["count"], errors="coerce").fillna(0).astype(int)
        df_topics_r["ratio_%"] = pd.to_numeric(df_topics_r["share"], errors="coerce").fillna(0.0) * 100.0
        shared_df = df_topics_r[df_topics_r["name"].isin(shared_names)].copy()
        shared_df = shared_df.sort_values(["count","ratio_%"], ascending=False)
        shared_detail = "; ".join(
            f"{n} ({int(c)}; {float(r_):0.2f}%)"
            for n, c, r_ in shared_df[["name","count","ratio_%"]].itertuples(index=False, name=None)
        )

        # FIELDS DETAIL (robust)
        fd = df_fields_r.copy()
        fd["count"] = pd.to_numeric(fd["count"], errors="coerce").fillna(0).astype(int)
        fd["ratio_%"] = pd.to_numeric(fd["share"], errors="coerce").fillna(0.0) * 100.0
        fd["si"] = pd.to_numeric(fd["si"], errors="coerce").fillna(0.0)
        fields_detail = "; ".join(
            f"{n} ({int(c)}; {ratio:0.2f}% ; SI {si:0.2f})"
            for n, c, ratio, si in fd[["name","count","ratio_%","si"]].itertuples(index=False, name=None)
        )

        arts_r = to_int_safe(r.articles_2020_24) or 0
        chaps_r = to_int_safe(r.book_chapters_2020_24) or 0
        books_r = to_int_safe(r.books_2020_24) or 0
        docs_detail = f"Articles {arts_r:,} | Chapters {chaps_r:,} | Books {books_r:,}"

        rows.append({
            "name": r.display_name,
            "OpenAlex": oa_works_url(r.openalex_id),             # filtered works for that institution
            "Copubs": oa_copubs_url(chosen_id, r.openalex_id),   # intersection with the chosen institution
            "type": r.type,
            "city": r.city,
            "country": r.country,
            "avg_pubs_per_year": int(round(r.avg_pubs_per_year)),
            "shared_topics_count": len(shared_names),
            # ORDER: % first, then SI
            "similarity_%": round(sim_share, 3),
            "shared_topics_detail": shared_detail,
            "fields_detail": fields_detail,
            "document_types": docs_detail,
            "openalex_id": r.openalex_id,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        st.info("No matches with the current filters.")
    else:
        # Rank: shared topics desc, then similarity_%
        res = res.sort_values(
            ["shared_topics_count","similarity_%","avg_pubs_per_year"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        show_details = st.checkbox("Show detailed columns (shared topics, fields, doc types)", value=False)

        base_cols = ["name","type","city","country","avg_pubs_per_year","shared_topics_count","similarity_%","Copubs","OpenAlex"]
        detail_cols = ["shared_topics_detail","fields_detail","document_types"]
        display_cols = (base_cols + detail_cols) if show_details else base_cols

        st.write(f"**{len(res)}** matching institutions")
        st.dataframe(
            res[display_cols],
            use_container_width=True,
            column_config={
                "Copubs": st.column_config.LinkColumn(
                    "Copubs",
                    help=f"Co-publications with {chosen.display_name} (2020–24; articles/chapters/books)"
                ),
                "OpenAlex": st.column_config.LinkColumn(
                    "OpenAlex",
                    help="Filtered works for this institution (2020–24; articles/chapters/books)"
                ),
                "avg_pubs_per_year": st.column_config.NumberColumn("Avg pubs/yr", format="%d"),
                "shared_topics_count": st.column_config.NumberColumn("Shared topics", format="%d"),
                "similarity_%": st.column_config.NumberColumn("Similarity (Fields %)", format="%.3f"),
            },
            hide_index=True,
        )

        exercise = f"{len(res)}_benchmarks"
        csv_cols = ["name","type","city","country","avg_pubs_per_year","shared_topics_count","similarity_%"] \
           + detail_cols + ["openalex_id","Copubs","OpenAlex"]
        st.download_button(
            "Download benchmark CSV",
            data=res[csv_cols].to_csv(index=False,encoding='utf-8-sig'),
            file_name=f"{inst_slug}_{exercise}_{ts_now()}.csv",
            mime="text/csv"
        )