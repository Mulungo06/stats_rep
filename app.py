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
    .stButton > button { border-radius: 8px; font-size: 0.85rem; font-weight: 500; }
    div[data-testid="stTabs"] button { font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

SAMPLE_DATA = {
    "Data":      ["2024-01-15","2024-01-22","2024-02-03","2024-02-18","2024-03-05",
                  "2024-03-19","2024-04-02","2024-04-17","2024-05-01","2024-05-20",
                  "2024-06-04","2024-06-21","2024-07-08","2024-07-25","2024-08-10"],
    "Região":    ["Norte","Sul","Centro","Este","Oeste","Norte","Sul","Centro","Este","Oeste",
                  "Norte","Sul","Centro","Este","Oeste"],
    "Casos_Pos": [120,85,210,45,310,145,92,245,61,380,178,110,290,73,420],
    "Casos_Neg": [880,415,590,355,690,755,408,655,439,620,722,490,710,527,580],
    "Testados":  [1000,500,800,400,1000,900,500,900,500,1000,900,600,1000,600,1000],
    "Óbitos":    [3,1,6,1,9,4,2,7,1,11,5,3,8,2,13],
    "Trimestre": ["Q1","Q1","Q1","Q1","Q1","Q1","Q2","Q2","Q2","Q2","Q2","Q3","Q3","Q3","Q3"],
}

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
    if n.endswith(".csv"):             return pd.read_csv(f)
    if n.endswith((".xls", ".xlsx")):  return pd.read_excel(f)
    if n.endswith(".tsv"):             return pd.read_csv(f, sep="\t")
    if n.endswith(".json"):
        c = json.load(f)
        if isinstance(c, list): return pd.DataFrame(c)
        for k in ("data","items","records","results"):
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
# Converte X para data se pedido, agrega pelo método escolhido, depois plota.

def smart_chart(df, chart_type, x_col, y_col, color_col, agg_func, x_as_date):
    cols = [c for c in [x_col, y_col, color_col] if c]
    plot = df[cols].copy()

    if x_as_date:
        plot[x_col] = pd.to_datetime(plot[x_col], errors="coerce")
        plot = plot.dropna(subset=[x_col])

    if agg_func and agg_func != "Nenhuma" and y_col in plot.columns:
        group = [x_col] + ([color_col] if color_col else [])
        plot  = plot.groupby(group, sort=False)[y_col].agg(AGG_FUNCS[agg_func]).reset_index()

    if x_as_date:
        plot = plot.sort_values(x_col)

    ca  = {"color": color_col} if color_col else {}
    pal = dict(color_discrete_sequence=px.colors.qualitative.Vivid)
    tpl = "plotly_white"

    match chart_type:
        case "bar":         fig = px.bar(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "bar_stacked": fig = px.bar(plot, x=x_col, y=y_col, **ca, **pal, template=tpl, barmode="stack")
        case "line":        fig = px.line(plot, x=x_col, y=y_col, **ca, markers=True, **pal, template=tpl)
        case "area":        fig = px.area(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "scatter":     fig = px.scatter(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "pie":         fig = px.pie(plot, names=x_col, values=y_col, **pal, template=tpl)
        case "donut":       fig = px.pie(plot, names=x_col, values=y_col, hole=0.45, **pal, template=tpl)
        case "treemap":
            path = [x_col] + ([color_col] if color_col else [])
            fig  = px.treemap(plot, path=path, values=y_col, **pal)
        case "box":         fig = px.box(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)
        case "histogram":   fig = px.histogram(plot, x=x_col, **ca, **pal, template=tpl)
        case _:             fig = px.bar(plot, x=x_col, y=y_col, **ca, **pal, template=tpl)

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
    if st.button("Carregar dados de exemplo", use_container_width=True):
        st.session_state.update(df=pd.DataFrame(SAMPLE_DATA),
                                source="Exemplo — Vigilância Epidemiológica")
        st.session_state.pop("summary", None)
        st.session_state.pop("working_df", None)

    st.divider()
    st.markdown("<div class='section-title'>Tipo de gráfico</div>", unsafe_allow_html=True)
    chart_label = st.selectbox("", list(CHART_TYPES.keys()), label_visibility="collapsed")
    chart_type  = CHART_TYPES[chart_label]

    st.divider()
    st.caption("Configura a chave nos Secrets:\n`GROQ_API_KEY` · `OPENAI_API_KEY` · `ANTHROPIC_API_KEY`")


# ── Load file ─────────────────────────────────────────────────────────────────

if uploaded:
    try:
        st.session_state.update(df=load_file(uploaded), source=uploaded.name)
        st.session_state.pop("summary", None)
        st.session_state.pop("working_df", None)
    except Exception as e:
        st.error(f"Erro ao ler o ficheiro: {e}")

df: pd.DataFrame | None = st.session_state.get("df")

st.markdown("<div class='page-title'>📊 DataAI Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Carregue qualquer base de dados · Prepare · Visualize · Questione com IA</div>",
            unsafe_allow_html=True)

if df is None:
    st.divider()
    st.info("👈  Carregue um ficheiro na barra lateral ou use os dados de exemplo para começar.")
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
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prep:
    st.markdown("#### Limpeza e transformação")
    working_df = st.session_state.get("working_df", df.copy())

    col_info = pd.DataFrame({
        "Coluna":        working_df.columns,
        "Tipo actual":   [str(working_df[c].dtype) for c in working_df.columns],
        "Tipo inferido": [infer_col_type(working_df[c]) for c in working_df.columns],
        "Nulos":         [working_df[c].isnull().sum() for c in working_df.columns],
        "Únicos":        [working_df[c].nunique() for c in working_df.columns],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)
    st.divider()

    with st.expander("🗑️ Remover colunas"):
        drop_cols = st.multiselect("Colunas a remover", working_df.columns.tolist())
        if st.button("Aplicar remoção", key="drop") and drop_cols:
            working_df = working_df.drop(columns=drop_cols)
            st.session_state["working_df"] = working_df
            st.success(f"Removidas: {drop_cols}")
            st.rerun()

    with st.expander("🔄 Converter tipo de coluna"):
        c1, c2 = st.columns(2)
        conv_col  = c1.selectbox("Coluna", working_df.columns.tolist(), key="conv_col")
        conv_type = c2.selectbox("Converter para",
                                 ["numérico","data/hora","texto","categórico"], key="conv_type")
        if st.button("Converter", key="do_conv"):
            try:
                if conv_type == "numérico":
                    working_df[conv_col] = pd.to_numeric(working_df[conv_col], errors="coerce")
                elif conv_type == "data/hora":
                    working_df[conv_col] = pd.to_datetime(working_df[conv_col], errors="coerce")
                elif conv_type == "texto":
                    working_df[conv_col] = working_df[conv_col].astype(str)
                elif conv_type == "categórico":
                    working_df[conv_col] = working_df[conv_col].astype("category")
                st.session_state["working_df"] = working_df
                st.success(f"'{conv_col}' convertida para {conv_type}.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with st.expander("🩹 Tratar valores nulos"):
        null_cols = [c for c in working_df.columns if working_df[c].isnull().any()]
        if not null_cols:
            st.success("Sem valores nulos detectados.")
        else:
            c1, c2 = st.columns(2)
            null_col   = c1.selectbox("Coluna com nulos", null_cols, key="null_col")
            null_strat = c2.selectbox("Estratégia",
                ["Remover linhas","Preencher com média","Preencher com mediana",
                 "Preencher com zero","Preencher com valor"], key="null_strat")
            fill_val = st.text_input("Valor", key="fill_val") if null_strat == "Preencher com valor" else ""
            if st.button("Aplicar", key="do_null"):
                s = working_df[null_col]
                if null_strat == "Remover linhas":
                    working_df = working_df.dropna(subset=[null_col])
                elif null_strat == "Preencher com média" and pd.api.types.is_numeric_dtype(s):
                    working_df[null_col] = s.fillna(s.mean())
                elif null_strat == "Preencher com mediana" and pd.api.types.is_numeric_dtype(s):
                    working_df[null_col] = s.fillna(s.median())
                elif null_strat == "Preencher com zero":
                    working_df[null_col] = s.fillna(0)
                elif null_strat == "Preencher com valor":
                    working_df[null_col] = s.fillna(fill_val)
                st.session_state["working_df"] = working_df
                st.success("Nulos tratados.")
                st.rerun()

    with st.expander("🔍 Filtrar linhas"):
        c1, c2, c3 = st.columns(3)
        f_col = c1.selectbox("Coluna", working_df.columns.tolist(), key="f_col")
        f_op  = c2.selectbox("Operador",
                             ["contém","igual a",">",">=","<","<=","não contém"], key="f_op")
        f_val = c3.text_input("Valor", key="f_val")
        if st.button("Filtrar", key="do_filter") and f_val:
            try:
                s = working_df[f_col]
                if f_op == "contém":
                    mask = s.astype(str).str.contains(f_val, case=False, na=False)
                elif f_op == "não contém":
                    mask = ~s.astype(str).str.contains(f_val, case=False, na=False)
                elif f_op == "igual a":
                    mask = s.astype(str) == f_val
                else:
                    v = float(f_val)
                    mask = {">": s>v, ">=": s>=v, "<": s<v, "<=": s<=v}[f_op]
                working_df = working_df[mask]
                st.session_state["working_df"] = working_df
                st.success(f"{len(working_df):,} registos após filtro.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with st.expander("➕ Criar coluna calculada"):
        c1, c2 = st.columns([1, 2])
        new_name = c1.text_input("Nome da nova coluna", key="new_name")
        new_expr = c2.text_input("Expressão", placeholder="ex: Casos_Pos / Testados * 100", key="new_expr")
        if st.button("Criar coluna", key="do_calc") and new_name and new_expr:
            try:
                working_df[new_name] = working_df.eval(new_expr)
                st.session_state["working_df"] = working_df
                st.success(f"Coluna '{new_name}' criada.")
                st.rerun()
            except Exception as e:
                st.error(f"Expressão inválida: {e}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↩️ Repor dados originais", use_container_width=True):
            st.session_state.pop("working_df", None)
            st.rerun()
    with c2:
        st.download_button("⬇️ Exportar dados tratados",
                           working_df.to_csv(index=False).encode(),
                           "dados_tratados.csv", "text/csv", use_container_width=True)
    with c3:
        if st.button("✅ Usar estes dados nos gráficos", use_container_width=True, type="primary"):
            st.session_state["working_df"] = working_df
            st.success("Dados actualizados. Vai ao separador Gráficos.")

    st.caption(f"Dimensão actual: **{working_df.shape[0]:,}** linhas × **{working_df.shape[1]}** colunas")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    plot_df   = st.session_state.get("working_df", df.copy())
    all_cols  = plot_df.columns.tolist()
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
    agg_func      = c4.selectbox("Agregação", list(AGG_FUNCS.keys()))
    x_as_date     = c5.toggle("Tratar X como data",
                               value=(infer_col_type(plot_df[x_col]) == "data"))
    chart_title   = c6.text_input("Título (opcional)")

    try:
        fig = smart_chart(plot_df, chart_type, x_col, y_col, color_col, agg_func, x_as_date)
        if chart_title:
            fig.update_layout(title=dict(text=chart_title, font=dict(size=15)))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Não foi possível gerar o gráfico: {e}")

    with st.expander("⬇️ Exportar gráfico"):
        fmt = st.radio("Formato", ["HTML","PNG","SVG"], horizontal=True)
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
    n_rows = c2.selectbox("", [50,100,500,"Todas"], label_visibility="collapsed")

    disp = active_df.copy()
    if search:
        mask = disp.astype(str).apply(lambda s: s.str.contains(search, case=False, na=False)).any(axis=1)
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

    question = st.text_input("", placeholder="Ex: Qual a semana com mais casos positivos?",
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
