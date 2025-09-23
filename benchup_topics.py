# benchup_topics.py
# BenchUp topics (v2.2)
# - Legend only once (OpenAlex domains)
# - Updated domain colors
# - Clickable institution names (OpenAlex)
# - Rounded avg/min/max pubs
# - Fields sort mode (SI / Count / Fixed Domain order with Medicine first)
# - Topics static-threshold coloring
# - Two similarities: Field SI (cosine on SI) and Field SI % (cosine on shares)

import streamlit as st
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# -------- Page & light mode ----------
st.set_page_config(page_title="BenchUp topics", layout="wide")
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] { background: #ffffff !important; color: #111 !important; }
    h1, h2, h3, h4, h5, h6 { color: #111 !important; }
    .red-big { color:#d62728; font-size: 2.0rem; font-weight:700; }
    .note { font-size:0.92rem; color:#333; }
    table.bench { width: 100%; border-collapse: collapse; }
    table.bench th, table.bench td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; vertical-align: top; }
    table.bench th { background:#fafafa; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "institutions_master.parquet"

# -------- Domain coloring ----------
# Updated per your palette (kept colors for A&H and Eng/Tech)
DOMAIN_COLORS = {
    "Health Sciences": "#F855C3",   # user provided ("#F855C32" looked like a typo)
    "Physical Sciences": "#8190FF",
    "Life Sciences": "#0CA750",
    "Social Sciences": "#FFCB3A",
    "Arts & Humanities": "#9467bd",
    "Engineering & Tech": "#ff7f0e",
    "Other": "#7f7f7f",
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

DOMAIN_FIXED_ORDER = [
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Engineering & Tech",
    "Social Sciences",
    "Arts & Humanities",
    "Other",
]

# Preferred field order inside domains (only setting Medicine explicitly as first)
PREFERRED_FIRST = {"Health Sciences": ["Medicine"]}

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
    return DOMAIN_COLORS.get(estimate_domain(field_name), DOMAIN_COLORS["Other"])

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

def build_vector(df_fields, all_keys, kind="si"):
    # kind: "si" or "share"
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
    # Domain order (fixed), with Medicine first inside Health Sciences, others A–Z
    df["domain"] = df["name"].apply(estimate_domain)
    def field_rank(row):
        dom = row["domain"]
        pref = PREFERRED_FIRST.get(dom, [])
        if row["name"] in pref:
            return (DOMAIN_FIXED_ORDER.index(dom), 0, row["name"])
        # non-preferred: block 1, alpha
        return (DOMAIN_FIXED_ORDER.index(dom), 1, row["name"])
    df["_rank"] = df.apply(field_rank, axis=1)
    df = df.sort_values(["_rank"])
    return df.drop(columns=["_rank"])

# static-threshold coloring for topics ratio (%)
def color_topics_count(val):
    thresholds = [0, 10, 20, 30]
    colors = ["#FFFFFF", "#d9bc2b", "#695806", "#332a00"]
    if pd.isna(val): return "background-color: #FFFFFF; color: black"
    if val >= thresholds[-1]:
        return "background-color: #332a00; color: white"
    for i in range(len(thresholds)-1):
        if thresholds[i] <= val < thresholds[i+1]:
            ratio = (val - thresholds[i]) / (thresholds[i+1] - thresholds[i])
            c1 = np.array([int(colors[i][j:j+2], 16) for j in (1, 3, 5)])
            c2 = np.array([int(colors[i+1][j:j+2], 16) for j in (1, 3, 5)])
            c = c1 * (1-ratio) + c2 * ratio
            hex_color = '#%02x%02x%02x' % tuple(c.astype(int))
            text_color = "white" if val >= 20 else "black"
            return f"background-color: {hex_color}; color: {text_color}"
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
            return f"{name} ({city, country})".replace("('","(").replace("', '", ", ").replace("')",")")
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

        avg = int(round(chosen.avg_pubs_per_year))
        st.markdown(f"<div class='red-big'>{avg:,}</div>", unsafe_allow_html=True)
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

    sort_mode = st.radio(
        "Sort fields by",
        options=["SI (desc)", "Count (desc)", "Fixed domain order"],
        index=0,
        horizontal=True
    )

    df_fields_all = parse_fields_blob(chosen.fields)
    df_fields_prim = parse_fields_blob(chosen.primary_fields)

    # Fields scatter with counts on left margin + legend on first plot
    def plot_fields(df_f, title, host, show_legend=False):
        with host:
            st.markdown(f"**{title}**")
            if df_f.empty:
                st.info("No field data"); return
            df_f = sort_fields_df(df_f, "SI (desc)" if sort_mode=="SI (desc)" else ("Count (desc)" if sort_mode=="Count (desc)" else "Fixed"))
            if sort_mode == "Fixed domain order":
                # Our helper returns sorted df already
                pass
            y = np.arange(len(df_f))
            sizes = np.clip((df_f["count"].fillna(0).astype(float) / (df_f["count"].max() or 1)) * 3000, 50, 3000)
            colors = [domain_color(n) for n in df_f["name"]]
            domains_here = [estimate_domain(n) for n in df_f["name"]]

            left_pad = -0.55
            fig, ax = plt.subplots(figsize=(6.8, 0.48*len(df_f)+1))
            ax.scatter(df_f["si"], y, s=sizes, c=colors, alpha=0.85, edgecolors="none", zorder=3)
            ax.axvline(1.0, color="#444", linestyle="--", linewidth=1, zorder=1)

            # counts text at left
            for yi, (cnt, name) in enumerate(zip(df_f["count"].fillna(0).astype(int), df_f["name"])):
                ax.text(left_pad+0.03, yi, f"{cnt:,}".replace(",", " "), va="center", ha="left", fontsize=8, color="#444")

            ax.set_xlabel("SI (Specialization Index)")
            ax.set_yticks(y)
            ax.set_yticklabels(df_f["name"])
            ax.invert_yaxis()
            xmax = max(1.1, (df_f["si"].max() or 0) * 1.15)
            ax.set_xlim(left=left_pad, right=xmax)
            ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)

            # legend just under the title (only for first plot)
            if show_legend:
                uniq = []
                for d in domains_here:
                    if d not in uniq: uniq.append(d)
                handles = [Patch(color=DOMAIN_COLORS.get(d, DOMAIN_COLORS["Other"]), label=d) for d in uniq]
                # place legend above axes (slightly outside)
                ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.06),
                          ncol=min(4, len(handles)), frameon=False)

            st.pyplot(fig, use_container_width=True)

            out = df_f[["name","count","share","si"]].copy()
            st.download_button(
                "Download fields CSV",
                data=out.to_csv(index=False),
                file_name=f"fields_{title.replace(' ','_').lower()}_{chosen.openalex_id}.csv",
                mime="text/csv"
            )

    sc1, sc2 = st.columns(2)
    plot_fields(df_fields_all, "Fields (all)", sc1, show_legend=True)
    plot_fields(df_fields_prim, "Fields (primary)", sc2, show_legend=False)

    # Topics tables with static-threshold coloring
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
                .applymap(color_topics_count, subset=["ratio_%"])
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

    # Defaults: ±300 rounded, min shared topics 15
    default_center = int(round(max(100.0, chosen.avg_pubs_per_year)))
    min_default = int(round(max(100.0, default_center - 300)))
    max_default = int(round(default_center + 300))

    cA, cB, cC = st.columns(3)
    with cA:
        primary_mode = st.radio(
            "Benchmark using…",
            options=["All topics & fields", "Primary topics & fields"],
            index=0
        )
        use_primary = primary_mode.startswith("Primary")
    with cB:
        min_shared_topics = st.slider("Minimum shared topics (top 100)", 1, 50, 15)
        min_sim_si = st.slider("Min similarity: Field SI (cosine)", 0.0, 1.0, 0.5, 0.01)
        min_sim_share = st.slider("Min similarity: Field SI % (cosine)", 0.0, 1.0, 0.5, 0.01)
    with cC:
        min_pubs = st.number_input("Min average pubs/year", min_value=100, value=int(min_default), step=25)
        max_pubs = st.number_input("Max average pubs/year", min_value=100, value=int(max_default), step=25)

    allowed_types = sorted([t for t in df["type"].dropna().unique()])
    type_filter = st.multiselect("Institution type (optional)", allowed_types, default=[])

    # Prepare target topic set & vectors
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
    target_vec_si = build_vector(target_fields_df, ALL_KEYS, kind="si")
    target_vec_share = build_vector(target_fields_df, ALL_KEYS, kind="share")

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

        # field vectors
        df_fields_r = parse_fields_blob(r.primary_fields if use_primary else r.fields)
        vec_si_r = build_vector(df_fields_r, ALL_KEYS, kind="si")
        vec_share_r = build_vector(df_fields_r, ALL_KEYS, kind="share")

        sim_si = cosine_sim(target_vec_si, vec_si_r)
        sim_share = cosine_sim(target_vec_share, vec_share_r)
        if sim_si < min_sim_si or sim_share < min_sim_share:
            continue

        # details
        if not df_topics_r.empty:
            df_topics_r["ratio_%"] = (df_topics_r["share"].fillna(0.0) * 100.0)
        shared_df = df_topics_r[df_topics_r["name"].isin(shared_names)].copy()
        shared_df = shared_df.sort_values(["count","ratio_%"], ascending=False)
        shared_detail = "; ".join(
            f"{n} ({int(c)}; {r_:0.2f}%)"
            for n, c, r_ in shared_df[["name","count","ratio_%"]].itertuples(index=False, name=None)
        )

        fd = df_fields_r.copy()
        fd["ratio_%"] = fd["share"].fillna(0.0) * 100.0
        fields_detail = "; ".join(
            f"{n} ({int(c)}; {ratio:0.2f}% ; SI {si:0.2f})"
            for n, c, ratio, si in fd[["name","count","ratio_%","si"]].itertuples(index=False, name=None)
        )

        arts_r = to_int_safe(r.articles_2020_24) or 0
        chaps_r = to_int_safe(r.book_chapters_2020_24) or 0
        books_r = to_int_safe(r.books_2020_24) or 0
        docs_detail = f"Articles {arts_r:,} | Chapters {chaps_r:,} | Books {books_r:,}"

        url = f"https://openalex.org/institutions/{str(r.openalex_id).lower()}"
        rows.append({
            "openalex_id": r.openalex_id,
            "Institution": f'<a href="{url}" target="_blank">{r.display_name}</a>',
            "name": r.display_name,  # for CSV
            "type": r.type,
            "country": r.country,
            "city": r.city,
            "avg_pubs_per_year": int(round(r.avg_pubs_per_year)),
            "shared_topics_count": len(shared_names),
            "similarity_SI": round(sim_si, 3),
            "similarity_%": round(sim_share, 3),
            "shared_topics_detail": shared_detail,
            "fields_detail": fields_detail,
            "document_types": docs_detail,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        st.info("No matches with the current filters.")
    else:
        res = res.sort_values(
            ["similarity_SI","similarity_%","shared_topics_count","avg_pubs_per_year"],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)

        show_details = st.checkbox("Show detailed columns (shared topics, fields, doc types)", value=False)

        base_cols = ["Institution","type","city","country","avg_pubs_per_year","shared_topics_count","similarity_SI","similarity_%"]
        detail_cols = ["shared_topics_detail","fields_detail","document_types"]
        display_cols = base_cols + (detail_cols if show_details else [])

        # Render as HTML table so the Institution name is clickable
        # (st.dataframe does not render per-cell markdown links cleanly)
        def html_table(df_view: pd.DataFrame) -> str:
            headers = "".join(f"<th>{h}</th>" for h in df_view.columns)
            rows_html = []
            for _, row in df_view.iterrows():
                tds = []
                for col in df_view.columns:
                    v = row[col]
                    if col == "Institution":
                        tds.append(f"<td>{v}</td>")
                    else:
                        tds.append(f"<td>{v}</td>")
                rows_html.append("<tr>" + "".join(tds) + "</tr>")
            return f'<table class="bench"><thead><tr>{headers}</tr></thead><tbody>' + "".join(rows_html) + "</tbody></table>"

        st.write(f"**{len(res)}** matching institutions")
        st.markdown(html_table(res[display_cols]), unsafe_allow_html=True)

        # CSV: include all (with plain name, not HTML)
        csv_cols = ["name","type","city","country","avg_pubs_per_year","shared_topics_count","similarity_SI","similarity_%"] + detail_cols + ["openalex_id"]
        st.download_button(
            "Download benchmark CSV",
            data=res[csv_cols].to_csv(index=False),
            file_name=f"benchmarks_{chosen.openalex_id}.csv",
            mime="text/csv"
        )
