# benchup_topics.py
# Streamlit app: BenchUp topics (v2)
# - OpenAlex-native, no Scimago
# - Search by name/acronym/alternatives (accent & whitespace robust)
# - Institution profile: location, avg pubs/yr (2020–24), doc-type pie,
#   fields vs primary-fields (SI scatter), topics vs primary-topics (tables + CSV)
# - Benchmarking: pubs range, shared topics (primary or not), type filter,
#   similarity threshold on SI vectors (fields or primary_fields)

import streamlit as st
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
import matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path

st.set_page_config(page_title="BenchUp topics", layout="wide")

# ------------------
# Config
# ------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "institutions_master.parquet"  # <— robust path

# Optional: color by OpenAlex broad domain
# You can extend this map as needed. Unmatched fields will be grey.
DOMAIN_COLORS = {
    "Health Sciences": "#d62728",
    "Life Sciences": "#bcbd22",
    "Physical Sciences": "#1f77b4",
    "Social Sciences": "#17becf",
    "Arts & Humanities": "#9467bd",
    "Engineering & Tech": "#ff7f0e"
}
# A simple mapping from some field-name snippets to broad domain (tune later)
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

# ------------------
# Helpers
# ------------------
def norm(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unidecode(str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)  # collapse punctuation/accents
    return " ".join(s.split())

def parse_fields_blob(blob: str):
    """
    Parse strings like:
    "Engineering (30123 ; 0.1536 ; 1.46) | Physics ..."

    Returns DataFrame with columns: name, count, share, si
    share & si are floats if present; missing pieces set to NaN
    """
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["name", "count", "share", "si"])
    parts = [p.strip() for p in str(blob).split("|")]
    rows = []
    for p in parts:
        # Extract "Name (a ; b ; c)" – tolerate commas or semicolons and spaces
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", p)
        if not m:
            continue
        name = m.group(1).strip()
        inside = [x.strip() for x in re.split(r"[;,]", m.group(2))]
        # Expect [count, share, si] but be robust
        count = to_int_safe(inside[0]) if len(inside) > 0 else np.nan
        share = to_float_safe(inside[1]) if len(inside) > 1 else np.nan
        si    = to_float_safe(inside[2]) if len(inside) > 2 else np.nan
        rows.append((name, count, share, si))
    df = pd.DataFrame(rows, columns=["name", "count", "share", "si"])
    return df

def parse_topics_blob(blob: str):
    """
    Parse strings like:
    "Topic A (184 ; 0.012) | Topic B (203 ; 0.010)"
    Returns DataFrame with columns: name, count, share
    """
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
    try:
        return int(str(x).replace(",", "").strip())
    except:
        return np.nan

def to_float_safe(x):
    try:
        return float(str(x).replace("%", "").replace(",", ".").strip())
    except:
        return np.nan

def estimate_domain(field_name: str) -> str:
    for key, dom in FIELD_TO_DOMAIN_HINTS.items():
        if key.lower() in field_name.lower():
            return dom
    return "Other"

def domain_color(field_name: str) -> str:
    dom = estimate_domain(field_name)
    return DOMAIN_COLORS.get(dom, "#7f7f7f")

def build_search_index(df):
    # Create a normalized search string per institution
    def row_key(r):
        parts = [r.get("display_name", "")]
        acr = r.get("acronyms", "")
        alt = r.get("alternatives", "")
        if acr:
            parts.append(acr)
        if alt:
            parts.append(alt)
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
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def build_SI_vector(df_fields, all_keys):
    # df_fields has name, count, share, si
    d = {r["name"]: (r["si"] if not pd.isna(r["si"]) else 0.0) for _, r in df_fields.iterrows()}
    return [d.get(k, 0.0) for k in all_keys]

# ------------------
# Load data
# ------------------
@st.cache_data(show_spinner=True)
def load_master():
    df = pd.read_parquet(str(DATA_PATH)) # cast to str for pyarrow
    # Keep minimal types
    needed = [
        "openalex_id", "display_name", "acronyms", "alternatives",
        "type", "country", "city",
        "articles_2020_24", "book_chapters_2020_24", "books_2020_24",
        "primary_fields", "fields",
        "primary_topics_top100", "topics_top100"
    ]
    # only keep columns that exist
    cols = [c for c in needed if c in df.columns]
    df = df[cols].copy()
    # add avg pubs/yr
    df["avg_pubs_per_year"] = df.apply(avg_pubs_per_year, axis=1)
    # build search index
    df = build_search_index(df)
    return df

df = load_master()

# ------------------
# UI: Search
# ------------------
st.title("BenchUp topics")
st.caption("OpenAlex-native benchmarking by topics and thematic shape")

q = st.text_input("Find your institution (name, acronym, alternatives)", placeholder="e.g., CNRS, MIT, \"Université Paris\" ...")
if q:
    nq = norm(q)
    # priority: startswith > contains
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
    sel = st.selectbox(
        "Select institution",
        options=["—"] + [f"{r.display_name}  •  {r.city or ''}  •  {r.country or ''}" for _, r in results.iterrows()],
        index=0
    )
    chosen = None
    if sel != "—":
        # recover row
        idx = int(results.index[[f"{r.display_name}  •  {r.city or ''}  •  {r.country or ''}" for _, r in results.iterrows()].index(sel)])
        chosen = results.loc[idx]

# ------------------
# UI: Profile
# ------------------
if chosen is not None:
    st.markdown("---")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.subheader(chosen.display_name)
        st.write(f"**Type:** {chosen.type or '—'}")
        st.write(f"**Location:** {chosen.city or '—'}, {chosen.country or '—'}")
        avg = chosen.avg_pubs_per_year
        st.write(f"**Average publications per year (2020–2024):** {avg:,.0f}")

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

    # Parse fields & primary_fields
    df_fields_all = parse_fields_blob(chosen.fields)
    df_fields_prim = parse_fields_blob(chosen.primary_fields)

    # Fields scatter (SI vs field, size by count), ALL vs PRIMARY side-by-side
    sc1, sc2 = st.columns(2)
    for df_f, title, host in [
        (df_fields_all, "Fields (all)", sc1),
        (df_fields_prim, "Fields (primary)", sc2)
    ]:
        with host:
            st.markdown(f"**{title}**")
            if df_f.empty:
                st.info("No field data")
            else:
                df_f = df_f.sort_values("si", ascending=False)
                y = np.arange(len(df_f))
                sizes = np.clip((df_f["count"].fillna(0).astype(float) / (df_f["count"].max() or 1)) * 3000, 50, 3000)
                colors = [domain_color(n) for n in df_f["name"]]
                fig, ax = plt.subplots(figsize=(6.5, 0.45*len(df_f)+1))
                ax.scatter(df_f["si"], y, s=sizes, c=colors, alpha=0.8, edgecolors="none")
                ax.axvline(1.0, color="#444", linestyle="--", linewidth=1)
                ax.set_xlabel("SI (Specialization Index)")
                ax.set_yticks(y)
                ax.set_yticklabels(df_f["name"])
                ax.invert_yaxis()
                ax.set_xlim(left=0)
                st.pyplot(fig, use_container_width=True)

                # CSV download of the underlying data
                out = df_f[["name","count","share","si"]].copy()
                out = out.sort_values(["si","count"], ascending=[False, False])
                csv = out.to_csv(index=False)
                st.download_button(
                    "Download fields CSV",
                    data=csv,
                    file_name=f"fields_{title.replace(' ','_').lower()}_{chosen.openalex_id}.csv",
                    mime="text/csv"
                )

    # Topics: ALL vs PRIMARY tables + CSV
    t1, t2 = st.columns(2)
    for blob, title, host in [
        (chosen.topics_top100, "Topics (all, top 100)", t1),
        (chosen.primary_topics_top100, "Topics (primary, top 100)", t2)
    ]:
        with host:
            st.markdown(f"**{title}**")
            df_topics = parse_topics_blob(blob)
            if df_topics.empty:
                st.info("No topic data")
            else:
                # Ratio in %
                df_topics["ratio_%"] = (df_topics["share"].fillna(0.0) * 100.0).round(2)
                df_topics = df_topics.sort_values(["count","ratio_%"], ascending=False).reset_index(drop=True)
                df_topics.index = df_topics.index + 1
                st.dataframe(
                    df_topics[["name","count","ratio_%"]],
                    use_container_width=True,
                    hide_index=False
                )
                csv = df_topics.to_csv(index=True)
                st.download_button(
                    "Download topics CSV",
                    data=csv,
                    file_name=f"topics_{title.replace(' ','_').lower()}_{chosen.openalex_id}.csv",
                    mime="text/csv"
                )

    # ------------------
    # Benchmarking
    # ------------------
    st.markdown("---")
    st.header("Benchmarking")

    # Defaults for pubs per year window
    default_center = max(100.0, chosen.avg_pubs_per_year)
    min_default = max(100.0, default_center - 1000.0)
    max_default = default_center + 1000.0

    cA, cB, cC = st.columns(3)
    with cA:
        use_primary_for_topics = st.radio(
            "Shared topics from…",
            options=["Topics (all)", "Primary topics"],
            index=0,
            help="Which list should be used to count shared topics (top 100)?"
        )
        use_primary_for_shape = st.radio(
            "Similarity vector built from…",
            options=["Fields (all)", "Primary fields"],
            index=0,
            help="Use SI across all fields, or only primary fields."
        )
    with cB:
        min_shared_topics = st.slider("Minimum shared topics (top 100)", 1, 50, 5)
        min_sim = st.slider("Minimum similarity (cosine on SI vectors)", 0.0, 1.0, 0.6, 0.01)
    with cC:
        min_pubs = st.number_input("Min average pubs/year", min_value=100.0, value=float(min_default), step=50.0)
        max_pubs = st.number_input("Max average pubs/year", min_value=100.0, value=float(max_default), step=50.0)

    allowed_types = sorted([t for t in df["type"].dropna().unique()])
    type_filter = st.multiselect("Institution type (optional)", allowed_types, default=[])

    # Prepare target topic set & SI vector
    target_topics_df = parse_topics_blob(
        chosen.primary_topics_top100 if use_primary_for_topics.startswith("Primary") else chosen.topics_top100
    )
    target_topics_set = set(target_topics_df["name"].tolist())

    target_fields_df = parse_fields_blob(
        chosen.primary_fields if use_primary_for_shape.startswith("Primary") else chosen.fields
    )
    # Build a global field key list (union of names across dataset) for this mode
    @st.cache_data(show_spinner=False)
    def global_field_keys(df_all: pd.DataFrame, col: str):
        keys = set()
        # Light pass: collect unique field names that ever appear in this column
        for s in df_all[col].fillna("").tolist():
            df_tmp = parse_fields_blob(s)
            keys.update(df_tmp["name"].tolist())
        return sorted(keys)

    field_col = "primary_fields" if use_primary_for_shape.startswith("Primary") else "fields"
    ALL_KEYS = global_field_keys(df, field_col)
    target_vec = build_SI_vector(target_fields_df, ALL_KEYS)

    # Now filter + compute matches
    # (We’ll do simple passes and keep things readable; cache helps anyway.)
    def row_passes_basic_filters(r):
        apy = r["avg_pubs_per_year"]
        if apy < min_pubs or apy > max_pubs:
            return False
        if type_filter:
            if r.get("type") not in set(type_filter):
                return False
        return True

    # Iterate once; this is fine for ~10–50k rows
    rows = []
    for _, r in df.iterrows():
        if r.openalex_id == chosen.openalex_id:
            continue
        if not row_passes_basic_filters(r):
            continue

        # shared topics
        df_topics_r = parse_topics_blob(
            r.primary_topics_top100 if use_primary_for_topics.startswith("Primary") else r.topics_top100
        )
        shared = target_topics_set.intersection(set(df_topics_r["name"].tolist()))
        if len(shared) < min_shared_topics:
            continue

        # similarity
        df_fields_r = parse_fields_blob(
            r.primary_fields if use_primary_for_shape.startswith("Primary") else r.fields
        )
        vec_r = build_SI_vector(df_fields_r, ALL_KEYS)
        sim = cosine_sim(target_vec, vec_r)
        if sim < min_sim:
            continue

        rows.append({
            "openalex_id": r.openalex_id,
            "name": r.display_name,
            "type": r.type,
            "country": r.country,
            "city": r.city,
            "avg_pubs_per_year": r.avg_pubs_per_year,
            "shared_topics_count": len(shared),
            "similarity": sim,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        st.info("No matches with the current filters.")
    else:
        res = res.sort_values(["similarity","shared_topics_count","avg_pubs_per_year"], ascending=[False, False, True])
        res["avg_pubs_per_year"] = res["avg_pubs_per_year"].round().astype(int)
        res["similarity"] = res["similarity"].round(3)

        st.write(f"**{len(res)}** matching institutions")
        st.dataframe(
            res[["name","type","city","country","avg_pubs_per_year","shared_topics_count","similarity"]],
            use_container_width=True
        )

        # CSV download
        csv = res.to_csv(index=False)
        st.download_button(
            "Download benchmark CSV",
            data=csv,
            file_name=f"benchmarks_{chosen.openalex_id}.csv",
            mime="text/csv"
        )
