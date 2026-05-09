import streamlit as st
import pandas as pd
import plotly.express as px
import json
import io

st.set_page_config(
    page_title="DataAI Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f9fafb; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
    .block-container { padding: 2rem 2.5rem; }

    .page-title {
        font-size: 1.75rem; font-weight: 700; color: #111827;
        letter-spacing: -0.02em; margin-bottom: 0.1rem;
    }
    .page-sub { font-size: 0.9rem; color: #6b7280; }

    .kpi-card {
        background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 1rem 1.25rem;
    }
    .kpi-label {
        font-size: 0.7rem; color: #9ca3af; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;
    }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #111827; line-height: 1; }
    .kpi-sub   { font-size: 0.75rem; color: #6b7280; margin-top: 3px; }

    .ai-card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 1.25rem 1.5rem; border-left: 4px solid #6366f1; margin: 0.5rem 0;
    }
    .ai-label {
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #6366f1; margin-bottom: 8px;
    }
    .ai-text { font-size: 0.95rem; color: #374151; line-height: 1.75; }

    .section-title {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.07em; color: #9ca3af; margin: 1.25rem 0 0.6rem;
    }
    .pill {
        display: inline-block; padding: 3px 10px; border-radius: 99px;
        font-size: 0.7rem; font-weight: 600; background: #f0fdf4;
        color: #16a34a; border: 1px solid #bbf7d0;
    }
    .preview-badge {
        font-size: 0.7rem; font-weight: 600; color: #d97706;
        background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 6px; padding: 2px 8px; display: inline-block; margin-bottom: 6px;
    }
    .stButton > button { border-radius: 8px; font-size: 0.85rem; font-weight: 500; }
    div[data-testid="stTabs"] button { font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

CHART_TYPES = {
    "📊 Barras":        "bar",
    "📈 Linhas":        "line",
    "📉 Área":          "area",
    "🔥 Barras Empil.": "bar_stacked",
    "🔵 Dispersão":     "scatter",
    "🥧 Pizza":         "pie",
    "🍩 Donut":         "donut",
    "🗺️ Treemap":       "treemap",
    "📦 Box Plot":      "box",
    "🌡️ Histograma":    "histogram",
}

AGG_FUNCS = {
    "Nenhuma": None, "Soma": "sum", "Média": "mean",
    "Contagem": "count", "Máximo": "max", "Mínimo": "min", "Mediana": "median",
}


# ── AI ────────────────────────────────────────────────────────────────────────

def get_provider():
    if st.secrets.get("GROQ_API_KEY"):      return "groq"
    if st.secrets.get("OPENAI_API_KEY"):    return "openai"
    if st.secrets.get("ANTHROPIC_API_KEY"): return "anthropic"
    return None

def provider_label():
    return {"groq": "Groq · LLaMA 3.3", "openai": "OpenAI · GPT-4o Mini",
            "anthropic": "Anthropic · Claude Haiku"}.get(get_provider(), "Sem API configurada")

def call_ai(prompt: str) -> str:
    p = get_provider()
    if not p:
        return "⚠️ Nenhuma chave API configurada nos Secrets do Streamlit."
    try:
        if p == "groq":
            from groq import Groq
            r = Groq(api_key=st.secrets["GROQ_API_KEY"]).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}], max_tokens=900)
            return r.choices[0].message.content
        if p == "openai":
            from openai import OpenAI
            r = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]).chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}], max_tokens=900)
            return r.choices[0].message.content
        if p == "anthropic":
            import anthropic
            r = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"]).messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=900,
                messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
    except Exception as e:
        return f"❌ {e}"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_file(f) -> pd.DataFrame:
    n = f.name.lower()
    if n.endswith(".csv"):            return pd.read_csv(f)
    if n.endswith((".xls", ".xlsx")): return pd.read_excel(f)
    if n.endswith(".tsv"):            return pd.read_csv(f, sep="\t")
    if n.endswith(".json"):
        c = json.load(f)
        if isinstance(c, list): return pd.DataFrame(c)
        for k in ("data", "items", "records", "results"):
            if k in c and isinstance(c[k], list): return pd.DataFrame(c[k])
        return pd.DataFrame([c])
    return pd.read_csv(f)

def infer_col_type(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s): return "numérica"
    try:
        pd.to_datetime(s, infer_datetime_format=True)
        return "data"
    except Exception:
        return "texto"

def ai_context(df: pd.DataFrame) -> str:
    num = df.select_dtypes(include="number").columns.tolist()
    stats = "\n".join(
        f"  {c}: total={df[c].sum():.0f} média={df[c].mean():.1f} "
        f"min={df[c].min():.1f} max={df[c].max():.1f}" for c in num)
    return (f"Colunas: {', '.join(df.columns)} | Registos: {len(df)}\n"
            f"Estatísticas:\n{stats}\n\nAmostra:\n{df.head(15).to_string(index=False)}")


# ── Chart builder ─────────────────────────────────────────────────────────────
# Normaliza tipos, agrega e plota. Separado da UI para facilitar testes.

def smart_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str,
                color_col: str | None, agg_func: str, x_as_date: bool):

    cols = [c for c in [x_col, y_col, color_col] if c and c in df.columns]
    plot = df[cols].copy()

    # Converte X para datetime se o utilizador activou o toggle
    if x_as_date:
        plot[x_col] = pd.to_datetime(plot[x_col], errors="coerce")
        plot = plot.dropna(subset=[x_col])

    # Garante que Y é numérico antes de agregar
    if y_col in plot.columns:
        plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")

    # Agrega por X (e por cor se definida)
    if agg_func and agg_func != "Nenhuma" and y_col in plot.columns:
        group = [x_col] + ([color_col] if color_col and color_col in plot.columns else [])
        plot  = plot.groupby(group, sort=False)[y_col].agg(AGG_FUNCS[agg_func]).reset_index()

    if x_as_date:
        plot = plot.sort_values(x_col)

    ca  = {"color": color_col} if color_col and color_col in plot.columns else {}
    pal = dict(color_discrete_sequence=px.colors.qualitative.Vivid)
    tpl = "plotly_white"

    match chart_type:
        case "bar":
            fig = px.bar(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "bar_stacked":
            fig = px.bar(plot, x=x_col, y=y_col, **ca, **pal, template=tpl, barmode="stack")
        case "line":
            fig = px.line(plot, x=x_col, y=y_col, **ca, markers=True, **pal, template=tpl)
        case "area":
            fig = px.area(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "scatter":
            fig = px.scatter(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "pie":
            fig = px.pie(plot, names=x_col, values=y_col, **pal, template=tpl)
        case "donut":
            fig = px.pie(plot, names=x_col, values=y_col, hole=0.45, **pal, template=tpl)
        case "treemap":
            path = [x_col] + ([color_col] if color_col and color_col in plot.columns else [])
            fig  = px.treemap(plot, path=path, values=y_col, **pal)
        case "box":
            fig = px.box(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "histogram":
            fig = px.histogram(plot, x=x_col, **ca, **pal, template=tpl)
        case _:
            fig = px.bar(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)

    fig.update_layout(
        margin=dict(t=40, b=30, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 DataAI Analyzer")
    st.markdown(f"<span class='pill'>⚡ {provider_label()}</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-title'>Fonte de dados</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv","xlsx","xls","json","tsv"],
                                label_visibility="collapsed")

    st.divider()
    st.markdown("<div class='section-title'>Tipo de gráfico</div>", unsafe_allow_html=True)
    chart_label = st.selectbox("", list(CHART_TYPES.keys()), label_visibility="collapsed")
    chart_type  = CHART_TYPES[chart_label]

    st.divider()
    st.caption("Secrets necessários:\n`GROQ_API_KEY` · `OPENAI_API_KEY` · `ANTHROPIC_API_KEY`")


# ── Load / reset on new upload ────────────────────────────────────────────────

if uploaded:
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("file_id") != file_id:
        st.session_state["df"]       = load_file(uploaded)
        st.session_state["source"]   = uploaded.name
        st.session_state["file_id"]  = file_id
        st.session_state.pop("working_df", None)
        st.session_state.pop("summary", None)

df: pd.DataFrame | None = st.session_state.get("df")

# ── Landing page ──────────────────────────────────────────────────────────────

st.markdown("<div class='page-title'>📊 DataAI Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Carregue qualquer base de dados · Prepare · Visualize · Questione com IA</div>",
            unsafe_allow_html=True)

if df is None:
    st.divider()
    st.info("👈  Carregue um ficheiro CSV, Excel, JSON ou TSV na barra lateral para começar.")
    st.stop()

# ── KPI bar ───────────────────────────────────────────────────────────────────

source       = st.session_state.get("source", "")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
text_cols    = df.select_dtypes(exclude="number").columns.tolist()
null_pct     = round(df.isnull().mean().mean() * 100, 1)

st.caption(f"📄 **{source}** · {provider_label()}")
st.divider()

for col, label, val, sub in zip(
    st.columns(4),
    ["Registos", "Colunas numéricas", "Valores nulos", "Tipos de dados"],
    [f"{len(df):,}", len(numeric_cols), f"{null_pct}%", df.dtypes.nunique()],
    [f"{len(df.columns)} colunas", f"{len(text_cols)} texto", "do total de células", "tipos distintos"],
):
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{val}</div><div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_prep, tab_chart, tab_summary, tab_data, tab_ask = st.tabs([
    "🧹 Preparar Dados", "📈 Gráficos", "✨ Resumo IA", "🗃️ Dados", "💬 Perguntar à IA"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREPARAR DADOS
# working_df persiste em session_state e é usado em todos os outros tabs.
# Cada operação actualiza working_df e redesenha o preview em tempo real.
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prep:

    working_df: pd.DataFrame = st.session_state.get("working_df", df.copy())

    # ── Diagnóstico ───────────────────────────────────────────────────────────
    st.markdown("#### Diagnóstico das colunas")
    col_info = pd.DataFrame({
        "Coluna":        working_df.columns,
        "Tipo actual":   [str(working_df[c].dtype) for c in working_df.columns],
        "Tipo inferido": [infer_col_type(working_df[c]) for c in working_df.columns],
        "Nulos":         [int(working_df[c].isnull().sum()) for c in working_df.columns],
        "% Nulos":       [f"{working_df[c].isnull().mean()*100:.1f}%" for c in working_df.columns],
        "Únicos":        [int(working_df[c].nunique()) for c in working_df.columns],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Operações de limpeza")

    # ── Remover colunas ───────────────────────────────────────────────────────
    with st.expander("🗑️ Remover colunas"):
        drop_cols = st.multiselect(
            "Selecciona uma ou mais colunas a remover",
            working_df.columns.tolist(), key="drop_cols"
        )
        if st.button("Remover seleccionadas", key="btn_drop", disabled=not drop_cols):
            working_df = working_df.drop(columns=drop_cols)
            st.session_state["working_df"] = working_df
            st.success(f"Removidas: {drop_cols}")
            st.rerun()

    # ── Converter tipos ───────────────────────────────────────────────────────
    with st.expander("🔄 Converter tipo de coluna"):
        conv_cols = st.multiselect(
            "Selecciona uma ou mais colunas a converter",
            working_df.columns.tolist(), key="conv_cols"
        )
        conv_type = st.selectbox(
            "Converter para",
            ["numérico", "data/hora", "texto", "categórico"], key="conv_type"
        )
        if st.button("Converter", key="btn_conv", disabled=not conv_cols):
            errs = []
            for c in conv_cols:
                try:
                    if conv_type == "numérico":
                        working_df[c] = pd.to_numeric(working_df[c], errors="coerce")
                    elif conv_type == "data/hora":
                        working_df[c] = pd.to_datetime(working_df[c], errors="coerce")
                    elif conv_type == "texto":
                        working_df[c] = working_df[c].astype(str)
                    elif conv_type == "categórico":
                        working_df[c] = working_df[c].astype("category")
                except Exception as e:
                    errs.append(f"{c}: {e}")
            st.session_state["working_df"] = working_df
            if errs:
                st.warning("Alguns erros: " + "; ".join(errs))
            else:
                st.success(f"{len(conv_cols)} coluna(s) convertidas para {conv_type}.")
            st.rerun()

    # ── Tratar nulos ──────────────────────────────────────────────────────────
    with st.expander("🩹 Tratar valores nulos"):
        null_cols_avail = [c for c in working_df.columns if working_df[c].isnull().any()]
        if not null_cols_avail:
            st.success("Sem valores nulos detectados.")
        else:
            sel_null_cols = st.multiselect(
                "Colunas com nulos a tratar", null_cols_avail, key="sel_null_cols"
            )
            null_strat = st.selectbox(
                "Estratégia",
                ["Remover linhas", "Preencher com média", "Preencher com mediana",
                 "Preencher com zero", "Preencher com valor"], key="null_strat"
            )
            fill_val = st.text_input("Valor personalizado", key="fill_val") \
                       if null_strat == "Preencher com valor" else ""

            if st.button("Aplicar", key="btn_null", disabled=not sel_null_cols):
                for c in sel_null_cols:
                    s = working_df[c]
                    if null_strat == "Remover linhas":
                        working_df = working_df.dropna(subset=[c])
                    elif null_strat == "Preencher com média" and pd.api.types.is_numeric_dtype(s):
                        working_df[c] = s.fillna(s.mean())
                    elif null_strat == "Preencher com mediana" and pd.api.types.is_numeric_dtype(s):
                        working_df[c] = s.fillna(s.median())
                    elif null_strat == "Preencher com zero":
                        working_df[c] = s.fillna(0)
                    elif null_strat == "Preencher com valor":
                        working_df[c] = s.fillna(fill_val)
                st.session_state["working_df"] = working_df
                st.success("Nulos tratados.")
                st.rerun()

    # ── Filtrar linhas ────────────────────────────────────────────────────────
    with st.expander("🔍 Filtrar linhas"):
        c1, c2, c3 = st.columns(3)
        f_col = c1.selectbox("Coluna", working_df.columns.tolist(), key="f_col")
        f_op  = c2.selectbox("Operador",
                             ["contém", "igual a", ">", ">=", "<", "<=", "não contém"],
                             key="f_op")
        f_val = c3.text_input("Valor", key="f_val")
        if st.button("Filtrar", key="btn_filter", disabled=not f_val):
            try:
                s = working_df[f_col]
                if f_op == "contém":
                    mask = s.astype(str).str.contains(f_val, case=False, na=False)
                elif f_op == "não contém":
                    mask = ~s.astype(str).str.contains(f_val, case=False, na=False)
                elif f_op == "igual a":
                    mask = s.astype(str) == f_val
                else:
                    v    = float(f_val)
                    mask = {">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[f_op]
                working_df = working_df[mask]
                st.session_state["working_df"] = working_df
                st.success(f"{len(working_df):,} registos após filtro.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # ── Coluna calculada ──────────────────────────────────────────────────────
    with st.expander("➕ Criar coluna calculada"):
        c1, c2 = st.columns([1, 2])
        new_name = c1.text_input("Nome da nova coluna", key="new_name")
        new_expr = c2.text_input(
            "Expressão (usa nomes de colunas como variáveis)",
            placeholder="ex: Casos_Pos / Testados * 100",
            key="new_expr",
        )
        if st.button("Criar coluna", key="btn_calc", disabled=not (new_name and new_expr)):
            try:
                working_df[new_name] = working_df.eval(new_expr)
                st.session_state["working_df"] = working_df
                st.success(f"Coluna '{new_name}' criada com sucesso.")
                st.rerun()
            except Exception as e:
                st.error(f"Expressão inválida: {e}")

    st.divider()

    # ── Preview dinâmico ──────────────────────────────────────────────────────
    # Mostra o estado actual dos dados após cada operação, em tempo real.
    st.markdown(
        f"<div class='preview-badge'>👁 Preview — {working_df.shape[0]:,} linhas × "
        f"{working_df.shape[1]} colunas</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(working_df.head(50), use_container_width=True, height=300)

    st.divider()

    # ── Acções globais ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↩️ Repor dados originais", use_container_width=True):
            st.session_state.pop("working_df", None)
            st.session_state.pop("summary", None)
            st.rerun()
    with c2:
        st.download_button(
            "⬇️ Exportar dados tratados",
            working_df.to_csv(index=False).encode(),
            "dados_tratados.csv", "text/csv",
            use_container_width=True,
        )
    with c3:
        if st.button("✅ Confirmar e usar nos gráficos", use_container_width=True, type="primary"):
            st.session_state["working_df"] = working_df
            st.session_state.pop("summary", None)
            st.success("Dados confirmados. Vai ao separador Gráficos.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    plot_df    = st.session_state.get("working_df", df.copy())
    all_cols   = plot_df.columns.tolist()
    num_cols_p = plot_df.select_dtypes(include="number").columns.tolist()
    txt_cols_p = plot_df.select_dtypes(exclude="number").columns.tolist()

    st.markdown("#### Configuração do gráfico")
    c1, c2, c3 = st.columns(3)
    x_col     = c1.selectbox("Eixo X / Labels", all_cols,
                              index=all_cols.index(txt_cols_p[0] if txt_cols_p else all_cols[0]))
    y_col     = c2.selectbox("Eixo Y / Valores", all_cols,
                              index=all_cols.index(num_cols_p[0] if num_cols_p else all_cols[0]))
    color_raw = c3.selectbox("Separar por (cor)", ["— Nenhuma —"] + all_cols)
    color_col = color_raw if color_raw != "— Nenhuma —" else None

    c4, c5, c6 = st.columns(3)
    agg_func    = c4.selectbox("Agregação", list(AGG_FUNCS.keys()))
    x_as_date   = c5.toggle("Tratar X como data",
                             value=(infer_col_type(plot_df[x_col]) == "data"))
    chart_title = c6.text_input("Título (opcional)")

    try:
        fig = smart_chart(plot_df, chart_type, x_col, y_col, color_col, agg_func, x_as_date)
        if chart_title:
            fig.update_layout(title=dict(text=chart_title, font=dict(size=15)))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Não foi possível gerar o gráfico: {e}")

    with st.expander("⬇️ Exportar gráfico"):
        fmt = st.radio("Formato", ["HTML", "PNG", "SVG"], horizontal=True)
        if st.button("Exportar"):
            if fmt == "HTML":
                buf = io.StringIO()
                fig.write_html(buf)
                st.download_button("Descarregar HTML", buf.getvalue(), "grafico.html", "text/html")
            else:
                try:
                    img = fig.to_image(format=fmt.lower())
                    st.download_button(f"Descarregar {fmt}", img, f"grafico.{fmt.lower()}")
                except Exception:
                    st.warning("Instala kaleido para exportar imagens: `pip install kaleido`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESUMO IA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    active_df = st.session_state.get("working_df", df)
    st.markdown("#### Análise executiva automática")

    if "summary" not in st.session_state:
        with st.spinner("A analisar os dados…"):
            prompt = (
                "Analisa estes dados e escreve um resumo executivo em português europeu, "
                "conciso e perspicaz (máximo 5 frases). Destaca padrões, tendências e anomalias. "
                "Responde apenas com o texto do resumo, sem títulos nem markdown.\n\n"
                + ai_context(active_df)
            )
            st.session_state["summary"] = call_ai(prompt)

    st.markdown(
        f"<div class='ai-card'><div class='ai-label'>🤖 Análise IA · {provider_label()}</div>"
        f"<div class='ai-text'>{st.session_state['summary']}</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Regenerar análise"):
        st.session_state.pop("summary", None)
        st.rerun()

    num_df = active_df.select_dtypes(include="number")
    if not num_df.empty:
        st.markdown("#### Estatísticas descritivas")
        st.dataframe(num_df.describe().T.style.format("{:.2f}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DADOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    active_df = st.session_state.get("working_df", df)
    c1, c2 = st.columns([3, 1])
    search = c1.text_input("", placeholder="🔍 Pesquisar em qualquer coluna…",
                           label_visibility="collapsed")
    n_rows = c2.selectbox("", [50, 100, 500, "Todas"], label_visibility="collapsed")

    disp = active_df.copy()
    if search:
        mask = disp.astype(str).apply(
            lambda s: s.str.contains(search, case=False, na=False)).any(axis=1)
        disp = disp[mask]
    if n_rows != "Todas":
        disp = disp.head(int(n_rows))

    st.dataframe(disp, use_container_width=True, height=430)
    st.caption(f"A mostrar **{len(disp):,}** de **{len(active_df):,}** registos")
    st.download_button("⬇️ Exportar CSV", active_df.to_csv(index=False).encode(),
                       "dados.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PERGUNTAR À IA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    active_df = st.session_state.get("working_df", df)
    st.markdown("#### Interrogue os seus dados em linguagem natural")

    question = st.text_input("", placeholder="Ex: Qual o período com mais casos positivos?",
                             label_visibility="collapsed")
    if st.button("Perguntar →", type="primary") and question:
        with st.spinner("A consultar a IA…"):
            prompt = (
                f"Dados:\n{ai_context(active_df)}\n\nPergunta: {question}\n\n"
                "Responde em português europeu, de forma directa e fundamentada nos dados."
            )
            st.markdown(
                f"<div class='ai-card'><div class='ai-label'>💬 Resposta · {provider_label()}</div>"
                f"<div class='ai-text'>{call_ai(prompt)}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Sugestões rápidas")
    suggestions = [
        "Qual o período com maior crescimento?",
        "Existem valores anómalos ou outliers?",
        "Quais as 5 maiores entradas?",
        "Que correlações existem entre as variáveis?",
        "Resume as tendências principais.",
    ]
    for row in [suggestions[:3], suggestions[3:]]:
        for col, s in zip(st.columns(len(row)), row):
            with col:
                if st.button(s, use_container_width=True, key=f"sug_{s[:12]}"):
                    with st.spinner("A consultar a IA…"):
                        prompt = (f"Dados:\n{ai_context(active_df)}\n\n"
                                  f"Pergunta: {s}\nResponde em português europeu.")
                        st.markdown(
                            f"<div class='ai-card'><div class='ai-label'>💬 Resposta</div>"
                            f"<div class='ai-text'>{call_ai(prompt)}</div></div>",
                            unsafe_allow_html=True,
                        )
