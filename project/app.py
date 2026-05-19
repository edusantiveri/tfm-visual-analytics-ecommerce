import io
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import re
from pathlib import Path

st.set_page_config(page_title="Visual Analytics · E-commerce", page_icon="📊", layout="wide")

def load_css(file_name):
    # 1. Obtenemos la ruta absoluta de la carpeta donde está este archivo app.py
    current_dir = Path(__file__).parent
    
    # 2. Construimos la ruta absoluta hacia el archivo CSS
    file_path = current_dir / file_name
    
    # 3. Comprobamos e inyectamos
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Esto te ayudará a ver en la pantalla si la ruta está fallando y cuál está buscando
        st.warning(f"⚠️ No se encontró el archivo de estilos en: {file_path}")

# Invocamos la función pasándole solo la ruta relativa interna
load_css("assets/styles.css")


# ══════════════════════════════════════════════════════════════════════════════
# ESQUEMA
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_COLUMNS = [
    ("order_id",    "texto",    True),
    ("date",        "fecha",    True),
    ("product",     "texto",    True),
    ("price",       "numérico", True),
    ("quantity",    "entero",   True),
    ("category",    "texto",    False),
    ("status",      "texto",    False),
    ("customer_id", "texto",    False),
]
REQUIRED_COLS = {c for c, _, r in EXPECTED_COLUMNS if r}


# ══════════════════════════════════════════════════════════════════════════════
# ESTANDARIZACIÓN DE COLUMNAS
# ══════════════════════════════════════════════════════════════════════════════

def looks_like_ids(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    id_pattern = re.compile(r'^[A-Z]{0,4}\d{3,}$|^\d+$')
    return sample.apply(lambda v: bool(id_pattern.match(v.strip()))).mean() > 0.7


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    keywords = {
        "order_id":    ["order_id", "id_pedido", "pedido_id"],
        "date":        ["date", "fecha", "timestamp", "day", "momento"],
        "product_id":  ["product_id", "id_producto", "producto_id", "sku"],
        "product":     ["product_name", "productname", "nombre_producto", "item_name", "articulo"],
        "price":       ["price", "precio", "monto", "amount", "cost", "valor", "unit_price", "item_price"],
        "quantity":    ["quantity", "cantidad", "qty", "unidades", "units", "cant"],
        "category":    ["category", "categoria", "tipo", "type", "clase"],
        "status":      ["status", "estado", "situacion"],
        "customer_id": ["customer_id", "id_cliente", "user_id", "client_id", "cliente"],
    }
    partial_keywords = {
        "order_id":    ["order", "pedido", "transaction"],
        "date":        ["date", "fecha", "time", "dia", "day"],
        "product_id":  ["sku"],
        "product":     ["name", "nombre", "articulo"],
        "price":       ["price", "precio", "cost", "valor", "amount"],
        "quantity":    ["quantity", "cantidad", "qty", "units", "cant"],
        "category":    ["category", "categoria", "tipo", "type"],
        "status":      ["status", "estado"],
        "customer_id": ["customer", "cliente", "user", "usuario", "client"],
    }
    already_present = {col for col in df.columns if col in keywords}
    scores: dict[str, dict[str, int]] = {col: {} for col in df.columns}
    col_order = {col: i for i, col in enumerate(df.columns)}
    for col in df.columns:
        col_snake = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', col)
        col_normalized = re.sub(r'[\s\-]+', '_', col_snake).lower().strip()
        tokens = set(col_normalized.split("_"))
        if col_normalized in ("product", "producto"):
            if "product_id" not in already_present and looks_like_ids(df[col]):
                scores[col] = {"product_id": 2}
            elif "product" not in already_present:
                scores[col] = {"product": 2}
            continue
        for standard_name in keywords:
            if standard_name in already_present:
                continue
            if col_normalized in keywords[standard_name]:
                scores[col][standard_name] = 2
            elif tokens & set(partial_keywords[standard_name]):
                scores[col][standard_name] = 1
    candidates = []
    for col, col_scores in scores.items():
        for standard_name, score in col_scores.items():
            candidates.append((score, -col_order[col], col, standard_name))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    assignments: dict[str, str] = {}
    assigned_standards: set[str] = set()
    for score, _, col, standard_name in candidates:
        if standard_name in assigned_standards or col in assignments:
            continue
        if standard_name in df.columns and assignments.get(col) != standard_name:
            continue
        assignments[col] = standard_name
        assigned_standards.add(standard_name)
    return df.rename(columns=assignments)


# ══════════════════════════════════════════════════════════════════════════════
# CAPA 1 — LIMPIEZA AUTOMÁTICA
# ══════════════════════════════════════════════════════════════════════════════

def apply_auto_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log: list[dict] = []
    df = df.copy()
    original_cols = list(df.columns)
    df.columns = (pd.Index(df.columns).str.strip().str.lower()
                  .str.replace(r"[\s\-]+", "_", regex=True)
                  .str.replace(r"[^\w]", "", regex=True))
    renamed = {o: n for o, n in zip(original_cols, df.columns) if o != n}
    if renamed:
        log.append({"tipo": "Normalización de nombres", "descripcion": "snake_case",
                    "impacto": str(renamed)})
    n_before = len(df)
    df.dropna(how="all", inplace=True)
    if len(df) < n_before:
        log.append({"tipo": "Filas vacías", "descripcion": "Filas 100% NaN eliminadas",
                    "impacto": f"{n_before - len(df)} fila(s)"})
    text_cols = df.select_dtypes(include="object").columns
    strip_counts = {}
    for col in text_cols:
        orig = df[col].copy()
        df[col] = df[col].str.strip()
        n = (orig != df[col]).sum()
        if n > 0:
            strip_counts[col] = int(n)
    if strip_counts:
        log.append({"tipo": "Strip espacios", "descripcion": "Leading/trailing eliminados",
                    "impacto": str(strip_counts)})
    type_changes = {}
    for col in [c for c in df.columns if any(kw in c for kw in
                ["price", "quantity", "qty", "amount", "cost", "revenue", "units", "cant"])]:
        if df[col].dtype == object:
            cleaned = (df[col].astype(str)
                       .str.replace(r"[€$£¥,]", "", regex=True)
                       .str.replace(r"\s", "", regex=True))
            conv = pd.to_numeric(cleaned, errors="coerce")
            if conv.notna().sum() / max(len(conv), 1) >= 0.8:
                df[col] = conv
                type_changes[col] = "→float64"
    if "date" in df.columns and df["date"].dtype == object:
        conv = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
        if conv.notna().sum() / max(len(conv), 1) >= 0.8:
            df["date"] = conv
            type_changes["date"] = "→datetime64"
    if type_changes:
        log.append({"tipo": "Inferencia de tipos", "descripcion": "Tipos corregidos",
                    "impacto": str(type_changes)})
    ghost = [c for c in df.columns if re.match(r"^unnamed[_:\s]\d+$", c.lower())]
    if ghost:
        df.drop(columns=ghost, inplace=True)
        log.append({"tipo": "Columnas fantasma", "descripcion": "Unnamed eliminadas",
                    "impacto": str(ghost)})
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c in df.columns and df[c].nunique() <= 50
                and c not in ("order_id", "product_id", "customer_id")]
    norm = {}
    for col in cat_cols:
        orig_n = df[col].nunique()
        n_val = df[col].str.lower().str.strip()
        if n_val.nunique() < orig_n:
            df[col] = n_val
            norm[col] = f"{orig_n}→{n_val.nunique()}"
    if norm:
        log.append({"tipo": "Normalización categórica", "descripcion": "Lowercase unificado",
                    "impacto": str(norm)})
    return df, log


def render_auto_cleaning_log(log: list[dict]):
    if not log:
        st.markdown('<div class="quality-ok">✅ Sin cambios automáticos necesarios</div>',
                    unsafe_allow_html=True)
        return
    st.markdown(f'<div class="quality-warn">🔧 {len(log)} transformación(es) automática(s) aplicada(s)</div>',
                unsafe_allow_html=True)
    with st.expander("📋 Ver log de transformaciones automáticas", expanded=False):
        for i, e in enumerate(log, 1):
            st.markdown(f"**{i}. {e['tipo']}** — _{e['descripcion']}_ — `{e['impacto']}`")
        log_df = pd.DataFrame(log)
        st.download_button("⬇️ Descargar log Capa 1 (CSV)",
                           data=log_df.to_csv(index=False).encode("utf-8"),
                           file_name="log_capa1.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# EDA — FUNCIONES ORIGINALES
# ══════════════════════════════════════════════════════════════════════════════

def profile_dataframe(df: pd.DataFrame) -> dict:
    n = len(df)
    report = {
        "filas":       n,
        "columnas":    df.shape[1],
        "nulos":       int(df.isnull().sum().sum()),
        "duplicados":  int(df.duplicated().sum()),
        "pct_nulos":   round(df.isnull().sum().sum() / max(n * df.shape[1], 1) * 100, 2),
        "col_nulos":   df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "missing_req": [c for c in REQUIRED_COLS if c not in df.columns],
        "outliers":    {},
    }
    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            report["outliers"][col] = n_out
    return report


def plot_missing_values(df: pd.DataFrame):
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "columna":   null_counts.index,
        "n_nulos":   null_counts.values,
        "pct_nulos": null_pct.values
    }).query("n_nulos > 0").sort_values("pct_nulos", ascending=True)
    if missing_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=missing_df["columna"], x=100 - missing_df["pct_nulos"],
        orientation="h", name="Presentes",
        marker_color="rgba(74, 144, 226, 0.15)", showlegend=True,
    ))
    fig.add_trace(go.Bar(
        y=missing_df["columna"], x=missing_df["pct_nulos"],
        orientation="h", name="Nulos", marker_color="#E05252",
        text=[f"{p}%  ({n})" for p, n in zip(missing_df["pct_nulos"], missing_df["n_nulos"])],
        textposition="outside", showlegend=True,
    ))
    fig.update_layout(
        barmode="stack", title="Distribución de valores nulos por columna",
        xaxis=dict(title="Porcentaje (%)", range=[0, 115]),
        yaxis=dict(title=""),
        height=max(250, len(missing_df) * 45 + 80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def summarize_duplicates(df: pd.DataFrame) -> pd.DataFrame | None:
    dup_mask = df.duplicated(keep=False)
    if not dup_mask.any():
        return None
    dup_df = df[dup_mask].copy()
    try:
        hashable_cols = [c for c in df.columns if df[c].apply(
            lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()]
        return (dup_df[hashable_cols]
                .groupby(hashable_cols, dropna=False)
                .size().reset_index(name="⚠️ n_repeticiones")
                .sort_values("⚠️ n_repeticiones", ascending=False))
    except Exception:
        return dup_df.drop_duplicates()


MAX_BOXPLOT_POINTS = 5_000

def plot_outlier_boxplots(df: pd.DataFrame, outlier_cols: dict) -> go.Figure | None:
    if not outlier_cols:
        return None
    cols_to_plot = list(outlier_cols.keys())
    ncols = min(len(cols_to_plot), 3)
    nrows = (len(cols_to_plot) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
        subplot_titles=[f"{c}  ({outlier_cols[c]:,} outliers)" for c in cols_to_plot])
    palette_hex = ["#4A90E2", "#E2904A", "#50C878", "#E25252", "#9B59B6", "#1ABC9C"]
    def hex_to_rgba(hex_color, alpha=0.3):
        h = hex_color.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"
    def smart_sample(series):
        s = series.dropna()
        if len(s) <= MAX_BOXPLOT_POINTS:
            return s, None
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask_out = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        outliers = s[mask_out]
        inliers = s[~mask_out]
        budget = max(MAX_BOXPLOT_POINTS - len(outliers), 500)
        if len(inliers) > budget:
            inliers = inliers.sample(budget, random_state=42)
        return pd.concat([outliers, inliers]), f"muestra de {len(outliers)+len(inliers):,}/{len(s):,}"
    sample_notes = []
    for idx, col in enumerate(cols_to_plot):
        row = idx // ncols + 1
        col_pos = idx % ncols + 1
        color = palette_hex[idx % len(palette_hex)]
        data_col, note = smart_sample(df[col])
        if note:
            sample_notes.append(f"`{col}`: {note}")
        fig.add_trace(go.Box(
            y=data_col, name=col, marker_color=color, boxmean="sd",
            line_width=1.5, fillcolor=hex_to_rgba(color), showlegend=False,
        ), row=row, col=col_pos)
    title = "Boxplots · Detección de outliers (método IQR)"
    if sample_notes:
        title += "<br><sup>⚡ Dataset grande — " + "  |  ".join(sample_notes) + "</sup>"
    fig.update_layout(title=title, height=300 * nrows,
        margin=dict(l=10, r=10, t=70, b=30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def check_format_consistency(df: pd.DataFrame) -> list[dict]:
    issues = []
    for col in df.columns:
        col_issues = []
        series = df[col].dropna()
        if series.empty:
            continue
        if df[col].dtype == object:
            str_series = series.astype(str)
            n_lt = (str_series != str_series.str.strip()).sum()
            if n_lt > 0:
                col_issues.append(f"✂️ {n_lt} valor(es) con espacios sobrantes (leading/trailing)")
            unique_vals = str_series.str.strip().unique()
            if len(unique_vals) <= 100:
                lower_map: dict[str, list[str]] = {}
                for v in unique_vals:
                    lower_map.setdefault(v.lower(), []).append(v)
                mixed_case = {k: v for k, v in lower_map.items() if len(v) > 1}
                if mixed_case:
                    examples = list(mixed_case.values())[:2]
                    examples_str = "; ".join([f"'{a}' / '{b}'" for a, b in [ex[:2] for ex in examples]])
                    col_issues.append(f"🔡 Mezcla de capitalización en {len(mixed_case)} valor(es): {examples_str}")
            n_numeric_strings = str_series.str.match(r'^\s*-?\d+([.,]\d+)?\s*$').sum()
            if 0 < n_numeric_strings < len(str_series):
                col_issues.append(f"🔢 {n_numeric_strings} valor(es) aparentemente numéricos en columna de texto")
            n_special = str_series.str.contains(r'[\x00\t\n\r]', regex=True).sum()
            if n_special > 0:
                col_issues.append(f"⛔ {n_special} valor(es) con caracteres especiales (tabs, saltos de línea…)")
        if df[col].dtype == object and any(kw in col.lower() for kw in
                ["date", "fecha", "time", "dia", "day", "timestamp"]):
            str_series = series.astype(str).str.strip()
            date_patterns = {
                "YYYY-MM-DD":        r'^\d{4}-\d{2}-\d{2}$',
                "DD/MM/YYYY":        r'^\d{2}/\d{2}/\d{4}$',
                "MM/DD/YYYY":        r'^\d{2}/\d{2}/\d{4}$',
                "DD-MM-YYYY":        r'^\d{2}-\d{2}-\d{4}$',
                "YYYY/MM/DD":        r'^\d{4}/\d{2}/\d{2}$',
                "ISO 8601 completo": r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}',
                "DD.MM.YYYY":        r'^\d{2}\.\d{2}\.\d{4}$',
            }
            matched_formats = {k for k, p in date_patterns.items()
                               if str_series.str.match(p).any()}
            if len(matched_formats) > 1:
                col_issues.append(f"📅 Múltiples formatos de fecha detectados: {', '.join(matched_formats)}")
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in ("price", "quantity", "revenue") and (series < 0).any():
                col_issues.append(f"➖ {int((series < 0).sum())} valor(es) negativos en columna '{col}'")
        if col_issues:
            issues.append({"columna": col, "problemas": col_issues})
    return issues


def render_format_consistency(issues: list[dict]):
    if not issues:
        st.markdown('<div class="quality-ok">✅ Sin problemas de consistencia de formato detectados</div>',
                    unsafe_allow_html=True)
        return
    st.markdown(f'<div class="quality-warn">⚠️ Se detectaron problemas de formato en <b>{len(issues)}</b> columna(s)</div>',
                unsafe_allow_html=True)
    st.markdown("")
    for item in issues:
        with st.expander(f"🔍 `{item['columna']}` — {len(item['problemas'])} problema(s)", expanded=False):
            for prob in item["problemas"]:
                st.markdown(f"- {prob}")


def render_eda(df: pd.DataFrame, title: str):
    rep = profile_dataframe(df)
    st.markdown(f"### {title}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filas", f"{rep['filas']:,}")
    m2.metric("Columnas", rep["columnas"])
    m3.metric("Nulos", f"{rep['nulos']:,}",
              delta=f"{rep['pct_nulos']}%",
              delta_color="inverse" if rep["nulos"] > 0 else "off")
    m4.metric("Duplicados", f"{rep['duplicados']:,}",
              delta_color="inverse" if rep["duplicados"] > 0 else "off")
    st.markdown("")
    st.markdown("**1 · Columnas requeridas**")
    if rep["missing_req"]:
        st.markdown(f'<div class="quality-err">❌ Columnas requeridas ausentes: <b>{", ".join(rep["missing_req"])}</b></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="quality-ok">✅ Todas las columnas requeridas presentes</div>',
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**2 · Duplicados**")
    if rep["duplicados"] > 0:
        st.markdown(f'<div class="quality-warn">⚠️ {rep["duplicados"]} filas duplicadas detectadas</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        st.caption("ℹ️ Se considera duplicada una fila cuyo valor en **todas** las columnas coincide exactamente.")
        dup_summary = summarize_duplicates(df)
        if dup_summary is not None and not dup_summary.empty:
            st.markdown("Combinaciones idénticas detectadas:")
            st.dataframe(
                dup_summary.style
                    .background_gradient(subset=["⚠️ n_repeticiones"], cmap="Reds", vmin=1)
                    .format({"⚠️ n_repeticiones": "{:.0f}"}),
                use_container_width=True,
                height=min(300, 35 * len(dup_summary) + 40),
            )
    else:
        st.markdown('<div class="quality-ok">✅ Sin duplicados</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**3 · Valores nulos**")
    if rep["pct_nulos"] > 0:
        st.markdown(f'<div class="quality-warn">⚠️ {rep["pct_nulos"]}% de valores nulos detectados</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        fig_missing = plot_missing_values(df)
        if fig_missing:
            st.plotly_chart(fig_missing, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="quality-ok">✅ Sin valores nulos</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**4 · Outliers**")
    try:
        if rep["outliers"]:
            outlier_summary = "  |  ".join([f"`{c}`: {n}" for c, n in rep["outliers"].items()])
            st.markdown(f'<div class="quality-warn">⚠️ Outliers detectados (IQR): {outlier_summary}</div>',
                        unsafe_allow_html=True)
            st.markdown("")
            fig_box = plot_outlier_boxplots(df, rep["outliers"])
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="quality-ok">✅ Sin outliers detectados</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al renderizar boxplots: {e}")
    st.markdown("---")
    st.markdown("**5 · Consistencia de formato**")
    try:
        render_format_consistency(check_format_consistency(df))
    except Exception as e:
        st.error(f"Error al analizar consistencia de formato: {e}")
    st.markdown("---")
    st.markdown("**Vista previa (primeras 5 filas)**")
    st.dataframe(df.head(5), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CAPA 2 — LIMPIEZA ASISTIDA
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    has_price    = "price"    in df.columns and pd.api.types.is_numeric_dtype(df["price"])
    has_quantity = "quantity" in df.columns and pd.api.types.is_numeric_dtype(df["quantity"])
    has_order    = "order_id" in df.columns
    if has_price and has_quantity:
        df = df.copy()
        df["_revenue"] = df["price"] * df["quantity"]
        kpis["facturacion"] = round(df["_revenue"].sum(), 2)
        kpis["unidades"]    = int(df["quantity"].sum())
    else:
        kpis["facturacion"] = None
        kpis["unidades"]    = None
    kpis["pedidos"] = int(df["order_id"].nunique()) if has_order else None
    if kpis.get("facturacion") is not None and kpis.get("pedidos"):
        kpis["ticket_medio"] = round(kpis["facturacion"] / kpis["pedidos"], 2)
    else:
        kpis["ticket_medio"] = None
    return kpis


def kpi_impact_preview(df_before: pd.DataFrame, df_after: pd.DataFrame):
    k_before = compute_kpis(df_before)
    k_after  = compute_kpis(df_after)
    st.markdown("**Impacto en KPIs si confirmas esta acción:**")
    changed = False
    for key, label, fmt in [
        ("facturacion",  "Facturación total", "€{:,.2f}"),
        ("ticket_medio", "Ticket medio",      "€{:,.2f}"),
        ("unidades",     "Unidades vendidas", "{:,.0f}"),
        ("pedidos",      "Pedidos únicos",    "{:,.0f}"),
    ]:
        b, a = k_before.get(key), k_after.get(key)
        if b is not None and a is not None and round(b, 2) != round(a, 2):
            delta = a - b
            pct   = (delta / b * 100) if b != 0 else 0
            c1, c2, _ = st.columns(3)
            c1.metric(f"{label} (antes)", fmt.format(b))
            c2.metric(f"{label} (después)", fmt.format(a),
                      delta=f"{'+' if delta >= 0 else ''}{fmt.format(delta)} ({pct:+.1f}%)",
                      delta_color="normal" if delta >= 0 else "inverse")
            changed = True
    if not changed:
        st.caption("ℹ️ Esta acción no modifica los KPIs principales.")


def _compute_outlier_mask(df: pd.DataFrame, col: str) -> pd.Series:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    return (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)


def render_capa2(fname: str):
    df = st.session_state["datasets_c2"][fname].copy()
    if "c2_log" not in st.session_state:
        st.session_state["c2_log"] = {}
    if fname not in st.session_state["c2_log"]:
        st.session_state["c2_log"][fname] = []

    st.markdown("### 🧹 Capa 2 — Limpieza asistida")
    st.caption("Cada acción muestra su impacto en KPIs antes de confirmar.")
    any_issue = False

    # C2.1 Duplicados
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        any_issue = True
        with st.expander(f"⚠️ {n_dup} filas duplicadas", expanded=True):
            st.caption("Duplicado = todas las columnas coinciden exactamente.")
            df_sin_dup = df.drop_duplicates()
            kpi_impact_preview(df, df_sin_dup)
            try:
                hashable = [c for c in df.columns if df[c].apply(
                    lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()]
                sample = df[df.duplicated(keep=False)][hashable].head(6)
                st.markdown(f"**Ejemplos** ({len(sample)} filas):")
                st.dataframe(sample, use_container_width=True, height=min(220, 35 * len(sample) + 40))
            except Exception:
                pass
            if st.button("✓ Eliminar duplicados", key=f"c2_dup_{fname}", type="primary"):
                st.session_state["datasets_c2"][fname] = df_sin_dup.reset_index(drop=True)
                st.session_state["c2_log"][fname].append({
                    "accion": "Eliminar duplicados",
                    "impacto": f"{n_dup} fila(s) eliminada(s)",
                })
                st.rerun()

    # C2.2 Valores nulos
    null_cols = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    if null_cols:
        any_issue = True
        with st.expander(f"⚠️ Nulos en {len(null_cols)} columna(s)", expanded=True):
            for col, n_null in null_cols.items():
                st.markdown(f"**`{col}`** — {n_null} nulo(s) ({n_null / len(df) * 100:.1f}%)")
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                if is_numeric:
                    opciones = {
                        "Imputar con la media":   ("media",   df[col].mean()),
                        "Imputar con la mediana": ("mediana", df[col].median()),
                        "Eliminar filas nulas":   ("eliminar", None),
                    }
                else:
                    moda_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
                    opciones = {
                        f"Imputar con la moda ('{moda_val}')": ("moda", moda_val),
                        "Eliminar filas nulas": ("eliminar", None),
                    }
                eleccion = st.radio(
                    f"¿Qué hacer con los nulos de `{col}`?",
                    list(opciones.keys()),
                    key=f"c2_null_opt_{fname}_{col}",
                    horizontal=True,
                )
                tipo, valor = opciones[eleccion]
                df_preview = df.dropna(subset=[col]) if tipo == "eliminar" else df.copy()
                if tipo != "eliminar":
                    df_preview[col] = df_preview[col].fillna(valor)
                kpi_impact_preview(df, df_preview)
                if st.button(f"✓ Aplicar a `{col}`", key=f"c2_null_apply_{fname}_{col}", type="primary"):
                    if tipo == "eliminar":
                        st.session_state["datasets_c2"][fname] = (
                            st.session_state["datasets_c2"][fname].dropna(subset=[col]).reset_index(drop=True))
                        msg = f"Nulos eliminados en `{col}`: {n_null} fila(s)"
                    else:
                        st.session_state["datasets_c2"][fname][col] = (
                            st.session_state["datasets_c2"][fname][col].fillna(valor))
                        msg = f"Nulos imputados en `{col}` con {tipo} ({valor})"
                    st.session_state["c2_log"][fname].append({"accion": f"Nulos — {col}", "impacto": msg})
                    st.rerun()
                st.markdown("---")

    # C2.3 Outliers
    outlier_cols = {}
    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        n_out = int(mask.sum())
        if n_out > 0:
            outlier_cols[col] = {"n": n_out, "mask": mask, "q1": q1, "q3": q3, "iqr": iqr}

    if outlier_cols:
        any_issue = True
        with st.expander(f"⚠️ Outliers en {len(outlier_cols)} columna(s)", expanded=True):
            st.caption("Detección por método IQR. Outlier = valor fuera de Q1−1.5·IQR o Q3+1.5·IQR.")
            for col, info in outlier_cols.items():
                st.markdown(f"**`{col}`** — {info['n']} outlier(s)")
                fig = go.Figure(go.Box(
                    y=df[col].dropna(), name=col, marker_color="#4A90E2", boxmean="sd",
                    line_width=1.5, fillcolor="rgba(74,144,226,0.2)", showlegend=False,
                ))
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                out_vals = df[col][info["mask"]]
                st.caption(f"Rango outliers: {out_vals.min():.2f} – {out_vals.max():.2f}  |  "
                           f"Q1={info['q1']:.2f}  Q3={info['q3']:.2f}  IQR={info['iqr']:.2f}")
                accion = st.radio(
                    f"¿Qué hacer con outliers de `{col}`?",
                    ["Conservar (no hacer nada)", "Excluir filas con outliers"],
                    key=f"c2_out_opt_{fname}_{col}", horizontal=True,
                )
                if accion == "Excluir filas con outliers":
                    df_preview_out = df[~info["mask"]].reset_index(drop=True)
                    kpi_impact_preview(df, df_preview_out)
                    if st.button(f"✓ Excluir outliers de `{col}`",
                                 key=f"c2_out_apply_{fname}_{col}", type="primary"):
                        mask_cur = _compute_outlier_mask(st.session_state["datasets_c2"][fname], col)
                        st.session_state["datasets_c2"][fname] = (
                            st.session_state["datasets_c2"][fname][~mask_cur].reset_index(drop=True))
                        st.session_state["c2_log"][fname].append({
                            "accion": f"Outliers — {col}",
                            "impacto": f"{info['n']} fila(s) excluida(s)",
                        })
                        st.rerun()
                st.markdown("---")

    if not any_issue:
        st.markdown('<div class="quality-ok">✅ Capa 2 completada — sin problemas que requieran decisión</div>',
                    unsafe_allow_html=True)

    log_c2 = st.session_state["c2_log"].get(fname, [])
    if log_c2:
        with st.expander(f"📋 Log Capa 2 — {len(log_c2)} acción(es)", expanded=False):
            for i, e in enumerate(log_c2, 1):
                st.markdown(f"**{i}. {e['accion']}** — `{e['impacto']}`")
            st.download_button("⬇️ Descargar log Capa 2 (CSV)",
                               data=pd.DataFrame(log_c2).to_csv(index=False).encode("utf-8"),
                               file_name=f"log_capa2_{fname}.csv", mime="text/csv",
                               key=f"dl_c2_log_{fname}")


# ══════════════════════════════════════════════════════════════════════════════
# CAPA 3 — DASHBOARD DE KPIs + EXPORTACIÓN
# Justificación académica:
#   Heer & Shneiderman (2012): especificación de datos, manipulación de vistas
#   y gestión del proceso analítico como pilares del VA interactivo.
#   Bačić & Fadlalla (2016): reducción de carga cognitiva en dashboards.
# ══════════════════════════════════════════════════════════════════════════════

_PLOT_CFG = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=48, b=32),
    font=dict(size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
_GRID = "rgba(200,200,200,0.3)"


def _add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ("price" in df.columns and "quantity" in df.columns
            and pd.api.types.is_numeric_dtype(df["price"])
            and pd.api.types.is_numeric_dtype(df["quantity"])):
        df["_revenue"] = df["price"] * df["quantity"]
    return df


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _render_kpi_cards(df: pd.DataFrame):
    """Nivel 1 — Tarjetas de métricas instantáneas."""
    df = _add_revenue(df)
    has_rev   = "_revenue"  in df.columns
    has_order = "order_id"  in df.columns
    has_qty   = ("quantity" in df.columns
                 and pd.api.types.is_numeric_dtype(df["quantity"]))

    facturacion  = df["_revenue"].sum()     if has_rev   else None
    n_pedidos    = df["order_id"].nunique() if has_order else None
    ticket_medio = (round(facturacion / n_pedidos, 2)
                    if facturacion and n_pedidos else None)
    unidades     = int(df["quantity"].sum()) if has_qty  else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Facturación total",
              f"€{facturacion:,.2f}" if facturacion is not None else "—")
    c2.metric("🧾 Ticket medio",
              f"€{ticket_medio:,.2f}" if ticket_medio is not None else "—")
    c3.metric("📦 Pedidos únicos",
              f"{n_pedidos:,}" if n_pedidos is not None else "—")
    c4.metric("📊 Unidades vendidas",
              f"{unidades:,}" if unidades is not None else "—")


def _render_timeline(df: pd.DataFrame, fname: str):
    """Nivel 2 — Evolución temporal con selector de granularidad."""
    df = _ensure_date(_add_revenue(df))
    if "_revenue" not in df.columns or "date" not in df.columns:
        st.info("ℹ️ Se necesitan `date`, `price` y `quantity` para la evolución temporal.")
        return

    df_ok = df.dropna(subset=["date", "_revenue"])
    if df_ok.empty:
        st.warning("No hay datos de fecha y revenue válidos.")
        return

    st.markdown("**Evolución temporal de ventas**")
    gran = st.radio("Granularidad", ["Diaria", "Semanal", "Mensual"],
                    horizontal=True, key=f"c3_gran_{fname}")
    freq = {"Diaria": "D", "Semanal": "W", "Mensual": "ME"}[gran]

    ts = (df_ok.set_index("date")["_revenue"]
          .resample(freq).sum()
          .reset_index()
          .rename(columns={"date": "periodo", "_revenue": "facturacion"}))
    ts["tendencia"] = ts["facturacion"].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ts["periodo"], y=ts["facturacion"],
        name="Facturación", marker_color="#4A90E2", opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=ts["periodo"], y=ts["tendencia"],
        name="Tendencia (media 3 periodos)",
        line=dict(color="#E2904A", width=2, dash="dot"), mode="lines",
    ))
    fig.update_layout(
        **_PLOT_CFG,
        title=f"Facturación {gran.lower()}",
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="€", showgrid=True, gridcolor=_GRID),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_category(df: pd.DataFrame):
    """Nivel 3 — Desglose por categoría: barras + tabla."""
    df = _add_revenue(df)
    if "_revenue" not in df.columns or "category" not in df.columns:
        st.info("ℹ️ Se necesitan `category`, `price` y `quantity` para el desglose.")
        return

    agg = {"facturacion": ("_revenue", "sum")}
    if "order_id" in df.columns:
        agg["pedidos"] = ("order_id", "nunique")
    if "quantity" in df.columns:
        agg["unidades"] = ("quantity", "sum")

    cat_df = (df.groupby("category")
              .agg(**agg)
              .reset_index()
              .sort_values("facturacion", ascending=True))
    cat_df["pct"] = (cat_df["facturacion"] / cat_df["facturacion"].sum() * 100).round(1)

    col_chart, col_table = st.columns([3, 2], gap="medium")

    with col_chart:
        st.markdown("**Facturación por categoría**")
        fig = go.Figure(go.Bar(
            x=cat_df["facturacion"], y=cat_df["category"],
            orientation="h",
            marker=dict(
                color=cat_df["facturacion"],
                colorscale=[[0, "rgba(74,144,226,0.3)"], [1, "#4A90E2"]],
                showscale=False,
            ),
            text=[f"€{v:,.0f}  ({p}%)"
                  for v, p in zip(cat_df["facturacion"], cat_df["pct"])],
            textposition="outside",
        ))
        fig.update_layout(
            **_PLOT_CFG,
            xaxis=dict(title="€", showgrid=True, gridcolor=_GRID),
            yaxis=dict(title=""),
            height=max(240, len(cat_df) * 44 + 80),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_table:
        st.markdown("**Tabla resumen**")
        rename = {"category": "Categoría", "facturacion": "Facturación (€)",
                  "pct": "% Total", "pedidos": "Pedidos", "unidades": "Unidades"}
        display_cols = [c for c in ["category", "facturacion", "pct", "pedidos", "unidades"]
                        if c in cat_df.columns]
        display_df = (cat_df[display_cols]
                      .sort_values("facturacion", ascending=False)
                      .rename(columns=rename))
        fmt = {"Facturación (€)": "€{:,.2f}", "% Total": "{:.1f}%"}
        if "Pedidos"   in display_df.columns: fmt["Pedidos"]   = "{:,.0f}"
        if "Unidades"  in display_df.columns: fmt["Unidades"]  = "{:,.0f}"
        st.dataframe(
            display_df.style
                .format(fmt)
                .background_gradient(subset=["Facturación (€)"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )


def _render_status(df: pd.DataFrame):
    """Nivel 4 — Distribución de status (donut). Solo si existe la columna."""
    if "status" not in df.columns:
        return
    sc = df["status"].value_counts().reset_index()
    sc.columns = ["status", "n"]
    if sc.empty:
        return
    st.markdown("**Estado de pedidos**")
    colors = ["#4A90E2", "#50C878", "#E2904A", "#E25252", "#9B59B6"]
    fig = go.Figure(go.Pie(
        labels=sc["status"], values=sc["n"], hole=0.55,
        marker=dict(colors=colors[:len(sc)]),
        textinfo="percent+label",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**_PLOT_CFG, height=280, showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=-0.1))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_top_products(df: pd.DataFrame, fname: str):
    """Nivel 5 — Top N productos por facturación."""
    df = _add_revenue(df)
    prod_col = next((c for c in ["product", "product_id"] if c in df.columns), None)
    if prod_col is None or "_revenue" not in df.columns:
        return
    n_unique = df[prod_col].nunique()
    if n_unique <= 1:
        return

    n_top = st.slider("Número de productos a mostrar", 5, min(25, n_unique), 10,
                      key=f"c3_top_{fname}")
    top = (df.groupby(prod_col)["_revenue"]
           .sum().nlargest(n_top).reset_index()
           .rename(columns={prod_col: "producto", "_revenue": "facturacion"}))

    st.markdown(f"**Top {n_top} productos por facturación**")
    fig = go.Figure(go.Bar(
        x=top["facturacion"], y=top["producto"],
        orientation="h", marker_color="#9B59B6", opacity=0.8,
        text=[f"€{v:,.0f}" for v in top["facturacion"]],
        textposition="outside",
    ))
    fig.update_layout(
        **_PLOT_CFG,
        xaxis=dict(title="€", showgrid=True, gridcolor=_GRID),
        yaxis=dict(title="", autorange="reversed"),
        height=max(240, n_top * 36 + 80),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_export(fname: str, df_clean: pd.DataFrame):
    """Exportación del dataset limpio + log de auditoría completo."""
    st.markdown("### 📥 Exportar")
    col_csv, col_log = st.columns(2)

    with col_csv:
        st.download_button(
            label="⬇️ Dataset limpio (CSV)",
            data=df_clean.to_csv(index=False).encode("utf-8"),
            file_name=f"{fname.replace('.', '_')}_limpio.csv",
            mime="text/csv",
            help="Dataset tras Capa 1 + Capa 2, listo para análisis externo",
            use_container_width=True,
        )

    with col_log:
        log_c1 = st.session_state.get("c1_logs", {}).get(fname, [])
        log_c2 = st.session_state.get("c2_log",  {}).get(fname, [])
        rows_c1 = [{"capa": "1 — Automática",
                    "accion": e.get("tipo", ""),
                    "impacto": e.get("impacto", "")} for e in log_c1]
        rows_c2 = [{"capa": "2 — Asistida",
                    "accion": e.get("accion", ""),
                    "impacto": e.get("impacto", "")} for e in log_c2]
        all_rows = rows_c1 + rows_c2
        if all_rows:
            st.download_button(
                label="⬇️ Log de auditoría completo (CSV)",
                data=pd.DataFrame(all_rows).to_csv(index=False).encode("utf-8"),
                file_name=f"{fname.replace('.', '_')}_log_auditoria.csv",
                mime="text/csv",
                help="Registro de todas las transformaciones aplicadas (Capa 1 + Capa 2)",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Log de auditoría (vacío)", disabled=True,
                      use_container_width=True)


def render_capa3(fname: str, df: pd.DataFrame):
    """
    Punto de entrada de la Capa 3.
    df = DataFrame post Capa 2 (ya limpio).
    """
    st.markdown("### 📊 Capa 3 — Dashboard de KPIs")
    st.caption(
        "Visualizaciones generadas automáticamente sobre los datos limpios. "
        "Responde a RQ1 (reducción de tiempo) y RQ3 (efectividad visual)."
    )

    has_price    = ("price"    in df.columns
                    and pd.api.types.is_numeric_dtype(df["price"]))
    has_quantity = ("quantity" in df.columns
                    and pd.api.types.is_numeric_dtype(df["quantity"]))

    if not has_price or not has_quantity:
        st.warning(
            "⚠️ El dashboard requiere `price` y `quantity` numéricos. "
            "Comprueba que la Capa 1 ha inferido los tipos correctamente."
        )
        return

    # Nivel 1: KPI cards
    _render_kpi_cards(df)
    st.markdown("---")

    # Nivel 2: Evolución temporal
    _render_timeline(df, fname)
    st.markdown("---")

    # Nivel 3 + 4: Categorías y status en columnas si ambos existen
    has_cat    = "category" in df.columns
    has_status = "status"   in df.columns

    if has_cat and has_status:
        col_cat, col_st = st.columns([2, 1], gap="medium")
        with col_cat:
            _render_category(df)
        with col_st:
            _render_status(df)
    elif has_cat:
        _render_category(df)
    elif has_status:
        _render_status(df)

    # Nivel 5: Top productos
    prod_col = next((c for c in ["product", "product_id"] if c in df.columns), None)
    if prod_col and df[prod_col].nunique() > 1:
        st.markdown("---")
        _render_top_products(df, fname)

    # Exportación
    st.markdown("---")
    _render_export(fname, df)


# ══════════════════════════════════════════════════════════════════════════════
# PARSE FILE
# ══════════════════════════════════════════════════════════════════════════════

def parse_file(uf):
    name = uf.name.lower()
    try:
        if name.endswith(".csv"):      df = pd.read_csv(uf)
        elif name.endswith(".xlsx"):   df = pd.read_excel(uf)
        elif name.endswith(".json"):
            data = json.load(uf)
            df = pd.json_normalize(data) if isinstance(data, (list, dict)) else None
            if df is None:
                return None, None, None, [], "JSON sin estructura tabular."
        elif name.endswith(".xml"):    df = pd.read_xml(uf)
        else:
            return None, None, None, [], "Formato no soportado."
        if df is None or df.empty or df.shape[1] == 0:
            return None, None, None, [], "Archivo sin estructura tabular válida."
        df = standardize_columns(df)
        df_raw = df.copy()
        df_c1, log_c1 = apply_auto_cleaning(df)
        df_c2 = df_c1.copy()
        return df_raw, df_c1, df_c2, log_c1, None
    except Exception as e:
        return None, None, None, [], f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key in ("datasets_raw", "datasets_c1", "datasets_c2", "c1_logs", "c2_log"):
    if key not in st.session_state:
        st.session_state[key] = {}
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def reset_uploader():
    st.session_state["uploader_key"] += 1


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <h1>📊 Visual Analytics · E-commerce</h1>
  <p>Carga tu fichero de ventas — el sistema analiza, limpia y genera KPIs automáticamente.</p>
</div>
""", unsafe_allow_html=True)

col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.subheader("Subir fichero")
    uploaded_files = st.file_uploader(
        "Arrastra tu fichero o selecciónalo",
        type=["csv", "xlsx", "json", "xml"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
        help="Máximo 150 MB. JSON y XML deben tener estructura tabular.",
    )
    if st.button("🗑 Limpiar lista de subida"):
        reset_uploader()
        st.rerun()

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state["datasets_raw"]:
                df_raw, df_c1, df_c2, log_c1, err = parse_file(uf)
                if err:
                    st.error(f"**{uf.name}** — {err}")
                else:
                    st.session_state["datasets_raw"][uf.name] = df_raw
                    st.session_state["datasets_c1"][uf.name]  = df_c1
                    st.session_state["datasets_c2"][uf.name]  = df_c2
                    st.session_state["c1_logs"][uf.name]      = log_c1
                    st.success(f"✅ **{uf.name}** cargado correctamente.")

    if st.session_state["datasets_raw"]:
        st.markdown(f"#### {len(st.session_state['datasets_raw'])} archivo(s) en memoria")

        for fname in list(st.session_state["datasets_raw"].keys()):
            df_raw = st.session_state["datasets_raw"][fname]
            df_c1  = st.session_state["datasets_c1"][fname]
            df_c2  = st.session_state["datasets_c2"][fname]
            log_c1 = st.session_state["c1_logs"][fname]

            with st.expander(
                f"📄 {fname}  —  {df_raw.shape[0]} filas · {df_raw.shape[1]} columnas",
                expanded=True
            ):
                btn_col, _ = st.columns([1, 4])
                with btn_col:
                    if st.button("🗑 Eliminar", key=f"del_{fname}"):
                        for k in ("datasets_raw", "datasets_c1", "datasets_c2",
                                  "c1_logs", "c2_log"):
                            st.session_state[k].pop(fname, None)
                        st.rerun()

                # PASO 1 — EDA en bruto
                render_eda(df_raw, "📊 EDA")
                st.markdown("---")

                # PASO 2 — Capa 1
                st.markdown("### 🔧 Capa 1 — Limpieza automática")
                render_auto_cleaning_log(log_c1)
                if log_c1:
                    st.markdown("")
                    
                else:
                    st.info("ℹ️ Sin cambios en Capa 1 — datos idénticos al estado inicial.")
                st.markdown("---")

                # PASO 3 — Capa 2
                render_capa2(fname)
                st.markdown("---")

                # PASO 4 — EDA post Capa 2 (solo si hubo cambios)
                df_c2_actual = st.session_state["datasets_c2"][fname]
                if len(df_c2_actual) != len(df_c1) or df_c2_actual.shape != df_c1.shape:
                    st.markdown("---")

                # PASO 5 — Capa 3
                render_capa3(fname, st.session_state["datasets_c2"][fname])

    else:
        st.info("No hay archivos cargados. Sube un fichero para comenzar.")


# ══════════════════════════════════════════════════════════════════════════════
# AUTO MERGE
# ══════════════════════════════════════════════════════════════════════════════

def auto_merge_datasets(datasets: dict) -> pd.DataFrame | None:
    dfs = list(datasets.values())
    if not dfs:
        return None
    dfs = sorted(dfs, key=lambda x: x.shape[1], reverse=True)
    base_df = dfs[0].copy()
    used = [False] * len(dfs)
    used[0] = True
    merged = True
    while merged:
        merged = False
        for i, df in enumerate(dfs):
            if used[i]:
                continue
            common_cols = list(set(base_df.columns) & set(df.columns))
            join_keys = [col for col in common_cols if "id" in col.lower()]
            if join_keys:
                try:
                    base_df = base_df.merge(df, on=join_keys, how="left",
                                            suffixes=("", "_dup"))
                    base_df = base_df[[c for c in base_df.columns
                                       if not c.endswith("_dup")]]
                    used[i] = True
                    merged = True
                except Exception as e:
                    st.warning(f"No se pudo unir un dataset: {e}")
    return base_df


if len(st.session_state["datasets_c2"]) > 1:
    st.markdown("---")
    st.markdown("## 🔗 Dataset unificado")
    st.caption("Une automáticamente los ficheros usando columnas ID como clave de join.")

    if st.button("Generar dataset combinado"):
        merged_df = auto_merge_datasets(st.session_state["datasets_c2"])

        if merged_df is not None and not merged_df.empty:
            st.success("✅ Datasets combinados correctamente")
            rep = profile_dataframe(merged_df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filas",      f"{merged_df.shape[0]:,}")
            col2.metric("Columnas",   merged_df.shape[1])
            col3.metric("Nulos",      rep["nulos"])
            col4.metric("Duplicados", rep["duplicados"])

            if {"price", "quantity"}.issubset(merged_df.columns):
                merged_df["revenue"] = merged_df["price"] * merged_df["quantity"]
                st.markdown("### 📊 KPIs del dataset combinado")
                k1, k2, k3 = st.columns(3)
                k1.metric("💰 Revenue total",  f"€{merged_df['revenue'].sum():,.2f}")
                if "order_id" in merged_df.columns:
                    k2.metric("🧾 Pedidos únicos", f"{merged_df['order_id'].nunique():,}")
                k3.metric("📦 Unidades", f"{int(merged_df['quantity'].sum()):,}")

            st.markdown("### Vista previa")
            st.dataframe(merged_df.head(), use_container_width=True)
            st.download_button(
                "⬇️ Descargar dataset combinado (CSV)",
                data=merged_df.to_csv(index=False).encode("utf-8"),
                file_name="dataset_combinado.csv",
                mime="text/csv",
            )
        else:
            st.warning("⚠️ No se pudieron combinar los datasets.")


# ══════════════════════════════════════════════════════════════════════════════
# COLUMNA INFO (derecha)
# ══════════════════════════════════════════════════════════════════════════════

with col_info:
    st.subheader("Columnas esperadas")
    cols_html = '<div class="col-grid">'
    for name, tipo, req in EXPECTED_COLUMNS:
        badge = ('<span class="badge-req">requerido</span>' if req
                 else '<span class="badge-opt">opcional</span>')
        cols_html += (f'<div class="col-card"><div class="col-name">{name}</div>'
                      f'<div class="col-type">{tipo}</div>{badge}</div>')
    cols_html += "</div>"
    st.markdown(cols_html, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Flujo de procesamiento")
    for i, step in enumerate([
        "EDA inicial — datos en bruto tal como llegan.",
        "Capa 1 — limpieza automática: tipos, espacios, columnas fantasma.",
        "Capa 2 — tú decides: duplicados, nulos, outliers con impacto en KPIs.",
        "Capa 3 — dashboard de KPIs interactivo + exportación.",
    ], 1):
        st.markdown(
            f'<div class="step"><div class="step-num">{i}</div>'
            f'<div class="step-text">{step}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.subheader("Formatos soportados")
    fc1, fc2, fc3, fc4 = st.columns(4)
    for c, fmt in zip([fc1, fc2, fc3, fc4], ["CSV", "Excel", "JSON", "XML"]):
        c.markdown(f"**{fmt}**")
    st.caption("⚠️ JSON y XML deben tener estructura tabular plana")