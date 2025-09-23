# benchup_topics.py
# BenchUp topics (v2.1) – refinements
# - Force light mode (via optional config + CSS safety net)
# - Title styling: Bench<span style="color:red">Up</span> + italic thin "topics"
# - Match list label: "Name (City, Country)"
# - Avg pubs/yr highlighted in red, larger font, with explanatory sentence + link
# - Fields SI plots: dedicated left margin; show counts next to field names
# - Topic tables: yellow-ish gradient on ratio column
# - Benchmarking: one toggle for primary vs all (applies to both topics & fields)
# - Defaults: ±300 pubs window (min 100), min shared topics = 15
# - Benchmark results: extra (hidden-by-default) detail columns + present in CSV

import streamlit as st
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
import matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path

# -------- Page & light mode ----------
st.set_page_config(page_title="BenchUp topics", layout="wide")
# CSS safety net to keep it light-ish even if user forces dark theme
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] { background: #ffffff !important; color: #111 !important; }
    h1, h2, h3, h4, h5, h6 { color: #111 !important; }
    .red-big { color:#d62728; font-size: 2.0rem; font-weight:700; }
    .note { font-size:0.92rem; color:#333; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "institutions_master.parquet"  # robust relative path

# -------- Domain coloring (same as before) ----------
DOMAIN_COLORS = {
    "Health Sciences": "#d62728",
    "Life Sciences": "#bcbd22",
    "Physical Sciences": "#1f77b4",
    "Social Sciences": "#17becf",
    "Arts & Humanities": "#9467bd",
    "Engineering & Tech": "#ff7f0e"
}
FIELD_TO_DOMAIN_HINTS = {
    "Medicine": "Health Sciences",
    "Nursing": "Health Sciences",
    "Health Professions": "Health Sciences",
    "Biochemistry": "Life Sciences",
    "Agricultural": "Life Sciences",
    "Immunology": "Life Sciences",
    "Neuroscience": "Life Sciences",
    "Biology": "Life Sciences",
    "Chemistry": "Physical Sciences",
    "Physics": "Physical Sciences",
    "Earth and Planetary": "Physical Sciences",
    "Mathematics": "Physical Sciences",
    "Computer Science": "Engineering & Tech",
    "Engineering": "Engineering & Tech",
    "Materials Science": "Engineering & Tech",
    "Environmental Science": "Physical Sciences",
    "Energy": "Engineering & Tech",
    "Psychology": "Social Sciences",
    "Decision Sciences": "Social Sciences",
    "Economics": "Social Sciences",
    "Business": "Social Sciences",
    "Social Sciences": "Social Sciences",
    "Arts and Humanities": "Arts & Humanities",
}

# -------- Helpers ----------
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
    for key, dom in FIELD_TO_DOMAIN_HINTS.items():
        if key.lower() in field_name.lower():
            return dom
    return "Other"

def domain_color(field_name: str) -> str:
    dom = estimate_domain(field_name)
    return DOMAIN_COLORS.get(dom, "#7f7f7f")

def build_search_index(df):
    def row_key(r):
        parts = [r.get("display_name", "")]
        if r.get("acronyms"): parts.append(r.get("acronyms"))
        if r.get("alternatives"): parts.append(r.get("alternatives"))
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

def build_SI_vector(df_fields, all_keys):
    d = {r["name"]: (r["si"] if not pd.isna(r["si"]) else 0.0) for _, r in df_fields.iterrows()}
    return [d.get(k, 0.0) for k in all_keys]

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

# -------- Search ----------
q = st.text_input("Find your institution (name, acronym, alternatives)", placeholder="e.g., CNRS, MIT, \"Université Paris\" ...")
if q:
    nq = norm(q)
    m_starts = df[df["__searchkey"].str.startswith(nq)].copy()
    m_contains = df[df["__searchkey"].str.contains(nq)].copy()
    results = pd.concat([m_starts, m_contains]).drop_duplicates(subset=["openalex_id"])
    results = results.sort_values(["display_name"]).head(200)
else:
    results = df.head(0)

if len(results) == 0 and q:
    st.info("No match. Try fewer words or the official acronym.")
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
    chosen = None
    if sel != "—":
        idx = int(results.index[[label_for_row(r) for _, r in results.iterrows()].index(sel)])
        chosen = results.loc[idx]

# -------- Profile ----------
if chosen is not None:
    st.markdown("---")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.subheader(chosen.display_name)
        st.write(f"**Type:** {chosen.type or '—'}")
        st.write(f"**Location:** {chosen.city or '—'}, {chosen.country or '—'}")

        avg = chosen.avg_pubs_per_year
        st.markdown(f"<div class='red-big'>{avg:,.0f}</div>", unsafe_allow_html=True)
        st.markdown(
            """<div class='note'>
            <strong>Average publications per year (2020–2024)</strong><br/>
            Publications include articles, book chapters, and books, per the 
            <a href="https://docs.openalex.org/api-entities/works/work-object#type" target="_blank">
            latest nomenclature of OpenAlex</a>.
            </div>""",
            unsafe_allow_html=True
        )

    with c2:
        # doc-type pie
        arts = to_int_safe(chosen.articles_2020_24) or 0
        chaps = to_int_safe(chosen.book_chapters_2020_24) or 0
        books = to_int_safe(chosen.books_2020_24) or 0
        fig, ax = plt.subplots(figsize=(3.8,3.8))
        vals = [arts, chaps, books]
        labels = ["Articles", "Book chapters", "Books"]
        vals2, labels2 = [], []
        for v, l in zip(vals, labels):
            if v and v>0:
                vals2.append(v); labels2.append(l)
        if sum(vals2)==0:
            ax.text(0.5,0.5,"No data", ha="center", va="center")
        else:
            ax.pie(vals2, labels=labels2, autopct='%1.0f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=False)
    with c3:
        st.write("**2020–24 totals**")
        st.metric("Articles", f"{arts:,}")
        st.metric("Chapters", f"{chaps:,}")
        st.metric("Books", f"{books:,}")

    # ---- Thematic views
    st.markdown("### Thematic profile")

    df_fields_all = parse_fields_blob(chosen.fields)
    df_fields_prim = parse_fields_blob(chosen.primary_fields)

    # Fields scatter with counts on left margin
    def plot_fields(df_f, title, host):
        with host:
            st.markdown(f"**{title}**")
            if df_f.empty:
                st.info("No field data")
                return
            df_f = df_f.sort_values("si", ascending=False).reset_index(drop=True)
            y = np.arange(len(df_f))
            sizes = np.clip((df_f["count"].fillna(0).astype(float) / (df_f["count"].max() or 1)) * 3000, 50, 3000)
            colors = [domain_color(n) for n in df_f["name"]]

            # left margin for counts
            left_pad = -0.55  # space for 6-digit counts
            fig, ax = plt.subplots(figsize=(6.8, 0.48*len(df_f)+1))
            ax.scatter(df_f["si"], y, s=sizes, c=colors, alpha=0.85, edgecolors="none", zorder=3)
            ax.axvline(1.0, color="#444", linestyle="--", linewidth=1, zorder=1)

            # counts next to field names (left margin)
            # place text slightly to the right of left_pad
            for yi, (cnt, name) in enumerate(zip(df_f["count"].fillna(0).astype(int), df_f["name"])):
                ax.text(left_pad+0.03, yi, f"{cnt:,}".replace(",", " "), va="center", ha="left", fontsize=8, color="#444")

            ax.set_xlabel("SI (Specialization Index)")
            ax.set_yticks(y)
            ax.set_yticklabels(df_f["name"])
            ax.invert_yaxis()
            xmax = max(1.1, (df_f["si"].max() or 0) * 1.15)
            ax.set_xlim(left=left_pad, right=xmax)

            # light grid
            ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)

            st.pyplot(fig, use_container_width=True)

            # CSV
            out = df_f[["name","count","share","si"]].copy().sort_values(["si","count"], ascending=[False, False])
            st.download_button(
                "Download fields CSV",
                data=out.to_csv(index=False),
                file_name=f"fields_{title.replace(' ','_').lower()}_{chosen.openalex_id}.csv",
                mime="text/csv"
            )

    sc1, sc2 = st.columns(2)
    plot_fields(df_fields_all, "Fields (all)", sc1)
    plot_fields(df_fields_prim, "Fields (primary)", sc2)

    # Topics tables (yellow-ish gradient on ratio)
    def show_topics_table(blob, title, host):
        with host:
            st.markdown(f"**{title}**")
            df_topics = parse_topics_blob(blob)
            if df_topics.empty:
                st.info("No topic data"); return
            df_topics["ratio_%"] = (df_topics["share"].fillna(0.0) * 100.0)
            df_topics = df_topics.sort_values(["count","ratio_%"], ascending=False).reset_index(drop=True)
            df_topics.index = df_topics.index + 1
            styled = (
                df_topics[["name","count","ratio_%"]]
                .style.format({"ratio_%": "{:.2f}"})
                .background_gradient(cmap="YlOrBr", subset=["ratio_%"])
            )
            st.dataframe(styled, use_container_width=True)
            st.download_button(
                "Download topics CSV",
                data=df_topics.to_csv(index=True),
                file_name=f"topics_{title.replace(' ','_').lower()}_{chosen.openalex_id}.csv",
                mime="text/csv"
            )

    t1, t2 = st.columns(2)
    show_topics_table(chosen.topics_top100, "Topics (all, top 100)", t1)
    show_topics_table(chosen.primary_topics_top100, "Topics (primary, top 100)", t2)

    # -------- Benchmarking --------
    st.markdown("---")
    st.header("Benchmarking")

    # Defaults: ±300 around center (floor 100) & min shared topics = 15
    default_center = max(100.0, chosen.avg_pubs_per_year)
    min_default = max(100.0, default_center - 300.0)
    max_default = default_center + 300.0

    cA, cB, cC = st.columns(3)
    with cA:
        # One choice: applies to both topics and fields
        primary_mode = st.radio(
            "Benchmark using…",
            options=["All topics & fields", "Primary topics & fields"],
            index=0
        )
        use_primary = primary_mode.startswith("Primary")
    with cB:
        min_shared_topics = st.slider("Minimum shared topics (top 100)", 1, 50, 15)
        min_sim = st.slider("Minimum similarity (cosine on SI vectors)", 0.0, 1.0, 0.6, 0.01)
    with cC:
        min_pubs = st.number_input("Min average pubs/year", min_value=100.0, value=float(min_default), step=25.0)
        max_pubs = st.number_input("Max average pubs/year", min_value=100.0, value=float(max_default), step=25.0)

    allowed_types = sorted([t for t in df["type"].dropna().unique()])
    type_filter = st.multiselect("Institution type (optional)", allowed_types, default=[])

    # Prepare target topic set & SI vector
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
    target_vec = build_SI_vector(target_fields_df, ALL_KEYS)

    def row_passes_basic_filters(r):
        apy = r["avg_pubs_per_year"]
        if apy < min_pubs or apy > max_pubs: return False
        if type_filter and r.get("type") not in set(type_filter): return False
        return True

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

        # similarity
        df_fields_r = parse_fields_blob(r.primary_fields if use_primary else r.fields)
        vec_r = build_SI_vector(df_fields_r, ALL_KEYS)
        sim = cosine_sim(target_vec, vec_r)
        if sim < min_sim:
            continue

        # ---- DETAIL COLUMNS ----
        # shared topics details for the matched institution (name: count; ratio%)
        if not df_topics_r.empty:
            df_topics_r["ratio_%"] = (df_topics_r["share"].fillna(0.0) * 100.0)
        shared_df = df_topics_r[df_topics_r["name"].isin(shared_names)].copy()
        shared_df = shared_df.sort_values(["count","ratio_%"], ascending=False)
        shared_detail = "; ".join(
            f"{n} ({int(c)}; {r_:0.2f}%)"
            for n, c, r_ in shared_df[["name","count","ratio_%"]].itertuples(index=False, name=None)
        )

        # fields detail for matched institution (mode-dependent)
        fd = df_fields_r.copy()
        fd["ratio_%"] = fd["share"].fillna(0.0) * 100.0
        fields_detail = "; ".join(
            f"{n} ({int(c)}; {ratio:0.2f}% ; SI {si:0.2f})"
            for n, c, ratio, si in fd[["name","count","ratio_%","si"]].itertuples(index=False, name=None)
        )

        # doc type breakdown
        arts_r = to_int_safe(r.articles_2020_24) or 0
        chaps_r = to_int_safe(r.book_chapters_2020_24) or 0
        books_r = to_int_safe(r.books_2020_24) or 0
        docs_detail = f"Articles {arts_r:,} | Chapters {chaps_r:,} | Books {books_r:,}"

        rows.append({
            "openalex_id": r.openalex_id,
            "name": r.display_name,
            "type": r.type,
            "country": r.country,
            "city": r.city,
            "avg_pubs_per_year": r.avg_pubs_per_year,
            "shared_topics_count": len(shared_names),
            "similarity": sim,
            # hidden-by-default columns (toggle below)
            "shared_topics_detail": shared_detail,
            "fields_detail": fields_detail,
            "document_types": docs_detail,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        st.info("No matches with the current filters.")
    else:
        res = res.sort_values(
            ["similarity","shared_topics_count","avg_pubs_per_year"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        res["avg_pubs_per_year"] = res["avg_pubs_per_year"].round().astype(int)
        res["similarity"] = res["similarity"].round(3)

        show_details = st.checkbox("Show detailed columns (shared topics, fields, doc types)", value=False)

        base_cols = ["name","type","city","country","avg_pubs_per_year","shared_topics_count","similarity"]
        detail_cols = ["shared_topics_detail","fields_detail","document_types"]
        display_cols = base_cols + (detail_cols if show_details else [])

        st.write(f"**{len(res)}** matching institutions")
        st.dataframe(res[display_cols], use_container_width=True)

        # CSV always includes all columns (incl. hidden-by-default)
        st.download_button(
            "Download benchmark CSV",
            data=res[base_cols + detail_cols].to_csv(index=False),
            file_name=f"benchmarks_{chosen.openalex_id}.csv",
            mime="text/csv"
        )
