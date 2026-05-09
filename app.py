import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataAI Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 600; margin-bottom: 0.25rem; }
    .subtitle { color: #6b7280; margin-bottom: 1.5rem; font-size: 1rem; }
    .ai-box {
        background: linear-gradient(135deg, #eff6ff, #f0fdf4);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .ai-answer {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #e2e8f0;
        margin-top: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .provider-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sample data ───────────────────────────────────────────────────────────────
SAMPLE_DATA = {
    "Região":    ["Norte","Sul","Centro","Este","Oeste","Norte","Sul","Centro","Este","Oeste","Norte","Sul","Centro","Este","Oeste"],
    "Vendas":    [45200,38700,52100,29400,61300,49800,41200,58700,33100,67900,53400,44500,63200,37800,72400],
    "Clientes":  [312,267,398,189,441,345,290,432,214,489,371,308,467,241,521],
    "Lucro":     [12400,9800,15600,7200,19800,14100,10500,18200,8400,22100,16300,11700,20100,9800,24600],
    "Trimestre": ["Q1","Q1","Q1","Q1","Q1","Q2","Q2","Q2","Q2","Q2","Q3","Q3","Q3","Q3","Q3"],
}

CHART_OPTIONS = {
    "📊 Barras":      "bar",
    "📈 Linhas":      "line",
    "🥧 Pizza":       "pie",
    "🍩 Donut":       "donut",
    "🔵 Dispersão":   "scatter",
    "🗺️ Treemap":     "treemap",
    "📦 Box Plot":    "box",
    "🌡️ Histograma":  "histogram",
}

# ── AI provider helpers ───────────────────────────────────────────────────────

def get_provider():
    """Detect which AI provider is configured in secrets."""
    if st.secrets.get("GOOGLE_API_KEY"):
        return "google"
    elif st.secrets.get("OPENAI_API_KEY"):
        return "openai"
    elif st.secrets.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def call_ai(prompt: str) -> str:
    provider = get_provider()

    if provider is None:
        return (
            "⚠️ Nenhuma chave API configurada.\n\n"
            "Adiciona uma das seguintes chaves nos Secrets do Streamlit:\n"
            "- GOOGLE_API_KEY (Google Gemini — grátis)\n"
            "- OPENAI_API_KEY (OpenAI)\n"
            "- ANTHROPIC_API_KEY (Anthropic Claude)"
        )

    try:
        # ── Google Gemini ──────────────────────────────────────────────────
        if provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            model = model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text

        # ── OpenAI ────────────────────────────────────────────────────────
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            return response.choices[0].message.content

        # ── Anthropic Claude ───────────────────────────────────────────────
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

    except Exception as e:
        return f"❌ Erro ao chamar a IA: {str(e)}"


def provider_badge() -> str:
    p = get_provider()
    if p == "google":
        return "🟢 Google Gemini (grátis)"
    elif p == "openai":
        return "⚫ OpenAI GPT-4o Mini"
    elif p == "anthropic":
        return "🟠 Anthropic Claude Haiku"
    return "⚠️ Sem API configurada"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        content = json.load(uploaded_file)
        if isinstance(content, list):
            return pd.DataFrame(content)
        for key in ("data", "items", "records", "results"):
            if key in content and isinstance(content[key], list):
                return pd.DataFrame(content[key])
        return pd.DataFrame([content])
    elif name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    else:
        return pd.read_csv(uploaded_file)


def build_data_context(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    stats_lines = []
    for col in numeric_cols:
        s = df[col].describe()
        stats_lines.append(
            f"  {col}: total={df[col].sum():.0f}, média={s['mean']:.1f}, "
            f"min={s['min']:.1f}, max={s['max']:.1f}"
        )
    sample = df.head(20).to_string(index=False)
    return (
        f"Colunas: {', '.join(df.columns)}\n"
        f"Total de registos: {len(df)}\n"
        f"Estatísticas:\n" + "\n".join(stats_lines) +
        f"\n\nAmostra:\n{sample}"
    )

# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart(df, chart_type, x_col, y_col, color_col=None):
    kwargs = dict(color_discrete_sequence=px.colors.qualitative.Set2)
    ca = {"color": color_col} if color_col and color_col != "— Nenhuma —" else {}

    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, **ca, **kwargs)
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, **ca, markers=True, **kwargs)
    elif chart_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, **kwargs)
    elif chart_type == "donut":
        fig = px.pie(df, names=x_col, values=y_col, hole=0.45, **kwargs)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, **ca, **kwargs)
    elif chart_type == "treemap":
        path = [x_col] + ([color_col] if color_col and color_col != "— Nenhuma —" else [])
        fig = px.treemap(df, path=path, values=y_col, **kwargs)
    elif chart_type == "box":
        fig = px.box(df, x=x_col, y=y_col, **ca, **kwargs)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_col, **ca, **kwargs)
    else:
        fig = px.bar(df, x=x_col, y=y_col, **kwargs)

    fig.update_layout(
        margin=dict(t=30, b=20, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 DataAI Analyzer")
    st.caption(provider_badge())
    st.markdown("---")

    st.markdown("### 📁 Carregar dados")
    uploaded = st.file_uploader(
        "Selecione um ficheiro",
        type=["csv", "xlsx", "xls", "json", "tsv"],
    )
    if st.button("▶ Usar dados de exemplo", use_container_width=True):
        st.session_state["df"] = pd.DataFrame(SAMPLE_DATA)
        st.session_state["source"] = "Exemplo — Vendas por Região"
        st.session_state.pop("summary", None)

    st.markdown("---")
    st.markdown("### ⚙️ Tipo de gráfico")
    chart_label = st.selectbox("", list(CHART_OPTIONS.keys()), label_visibility="collapsed")
    chart_type = CHART_OPTIONS[chart_label]

    st.markdown("---")
    st.markdown("### 🔑 Como configurar a API")
    st.markdown("""
**Grátis — Google Gemini:**
1. [aistudio.google.com](https://aistudio.google.com) → Get API Key
2. Adiciona nos Secrets: `GOOGLE_API_KEY = "..."`

**Pago — OpenAI:**
1. [platform.openai.com](https://platform.openai.com) → API Keys
2. Adiciona: `OPENAI_API_KEY = "sk-..."`

**Pago — Anthropic:**
1. [console.anthropic.com](https://console.anthropic.com) → API Keys
2. Adiciona: `ANTHROPIC_API_KEY = "sk-ant-..."`
""")

# ── Load uploaded file ────────────────────────────────────────────────────────

if uploaded:
    try:
        st.session_state["df"] = load_data(uploaded)
        st.session_state["source"] = uploaded.name
        st.session_state.pop("summary", None)
    except Exception as e:
        st.error(f"Erro ao ler o ficheiro: {e}")

df = st.session_state.get("df")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">📊 DataAI Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Carregue qualquer base de dados e deixe a IA fazer a análise</div>', unsafe_allow_html=True)

if df is None:
    st.info("👈 Carregue um ficheiro na barra lateral ou clique em **Usar dados de exemplo**.")

    st.markdown("---")
    st.markdown("### 🚀 Como começar grátis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🟢 Google Gemini (GRÁTIS)**
1. Vai a [aistudio.google.com](https://aistudio.google.com)
2. Cria conta Google
3. Clica em **"Get API Key"**
4. Copia a chave
5. Cola nos Secrets do Streamlit: `GOOGLE_API_KEY = "..."`
        """)
    with col2:
        st.markdown("""
**⚫ OpenAI (a partir de $5)**
1. Vai a [platform.openai.com](https://platform.openai.com)
2. Cria conta
3. Vai a **API Keys** → Create
4. Adiciona $5 de crédito
5. Cola: `OPENAI_API_KEY = "sk-..."`
        """)
    with col3:
        st.markdown("""
**🟠 Anthropic (a partir de $5)**
1. Vai a [console.anthropic.com](https://console.anthropic.com)
2. Cria conta
3. Vai a **API Keys** → Create
4. Adiciona crédito
5. Cola: `ANTHROPIC_API_KEY = "sk-ant-..."`
        """)
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────

source = st.session_state.get("source", "")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
text_cols    = df.select_dtypes(exclude="number").columns.tolist()

st.caption(f"📄 **{source}**  •  {len(df):,} registos  •  {len(df.columns)} colunas  •  {provider_badge()}")

cols_m = st.columns(4)
cols_m[0].metric("Registos", f"{len(df):,}")
cols_m[1].metric("Colunas", len(df.columns))
if numeric_cols:
    cols_m[2].metric(f"Total: {numeric_cols[0]}", f"{df[numeric_cols[0]].sum():,.0f}")
    cols_m[3].metric(f"Média: {numeric_cols[0]}", f"{df[numeric_cols[0]].mean():,.1f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_summary, tab_chart, tab_data, tab_ask = st.tabs(
    ["✨ Resumo IA", "📈 Gráficos", "🗃️ Dados", "💬 Perguntar à IA"]
)

# ── Resumo ────────────────────────────────────────────────────────────────────
with tab_summary:
    st.markdown("#### Análise automática com IA")

    if "summary" not in st.session_state:
        with st.spinner("A gerar análise..."):
            ctx = build_data_context(df)
            prompt = (
                "Analisa estes dados e fornece um resumo executivo em português de Portugal, "
                "conciso e perspicaz (máximo 5 frases). Destaca padrões importantes, valores extremos "
                "ou tendências. Responde apenas com o resumo, sem títulos nem markdown.\n\n" + ctx
            )
            st.session_state["summary"] = call_ai(prompt)

    st.markdown(
        f'<div class="ai-box">🤖 <strong>Análise IA</strong><br><br>{st.session_state["summary"]}</div>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Regenerar análise"):
        st.session_state.pop("summary", None)
        st.rerun()

    if numeric_cols:
        st.markdown("#### Estatísticas descritivas")
        st.dataframe(df[numeric_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

# ── Gráficos ──────────────────────────────────────────────────────────────────
with tab_chart:
    all_cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        default_x = text_cols[0] if text_cols else all_cols[0]
        x_col = st.selectbox("Eixo X / Labels", all_cols, index=all_cols.index(default_x))
    with c2:
        default_y = numeric_cols[0] if numeric_cols else all_cols[0]
        y_col = st.selectbox("Eixo Y / Valores", all_cols, index=all_cols.index(default_y))
    with c3:
        color_col = st.selectbox("Cor (opcional)", ["— Nenhuma —"] + all_cols)

    try:
        fig = build_chart(df, chart_type, x_col, y_col, color_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")

    with st.expander("📥 Exportar gráfico"):
        fmt = st.radio("Formato", ["HTML", "PNG", "SVG"], horizontal=True)
        if st.button("Exportar"):
            if fmt == "HTML":
                buf = io.StringIO()
                fig.write_html(buf)
                st.download_button("⬇ Descarregar HTML", buf.getvalue(), "grafico.html", "text/html")
            else:
                try:
                    img = fig.to_image(format=fmt.lower())
                    st.download_button(f"⬇ Descarregar {fmt}", img, f"grafico.{fmt.lower()}")
                except Exception:
                    st.warning("Para exportar PNG/SVG instala: `pip install kaleido`")

# ── Dados ─────────────────────────────────────────────────────────────────────
with tab_data:
    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.text_input("🔍 Filtrar", placeholder="Escreve para filtrar...")
    with c2:
        n_rows = st.selectbox("Mostrar", [50, 100, 500, "Tudo"])

    display_df = df.copy()
    if search:
        mask = display_df.astype(str).apply(lambda c: c.str.contains(search, case=False)).any(axis=1)
        display_df = display_df[mask]
    if n_rows != "Tudo":
        display_df = display_df.head(int(n_rows))

    st.dataframe(display_df, use_container_width=True, height=400)
    st.caption(f"A mostrar {len(display_df):,} de {len(df):,} registos")
    st.download_button("⬇ Exportar CSV", df.to_csv(index=False).encode(), "dados.csv", "text/csv")

# ── Perguntar à IA ────────────────────────────────────────────────────────────
with tab_ask:
    st.markdown("#### Faça qualquer pergunta sobre os seus dados")

    question = st.text_input(
        "Pergunta",
        placeholder="Ex: Qual é a região com mais vendas? Qual o trimestre mais lucrativo?",
        label_visibility="collapsed",
    )
    if st.button("🤖 Perguntar", type="primary") and question:
        with st.spinner("A consultar a IA..."):
            ctx = build_data_context(df)
            prompt = (
                f"Tens acesso a esta base de dados:\n{ctx}\n\n"
                f"Pergunta: {question}\n\n"
                "Responde em português de Portugal, de forma clara e direta."
            )
            answer = call_ai(prompt)
            st.markdown(
                f'<div class="ai-answer">💬 <strong>Resposta:</strong><br><br>{answer}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### 💡 Sugestões")
    suggestions = [
        "Qual o valor máximo e mínimo?",
        "Quais as 3 maiores entradas?",
        "Existem valores nulos ou em falta?",
        "Que tendências consegues identificar?",
        "Faz um resumo estatístico completo.",
    ]
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                with st.spinner("A consultar a IA..."):
                    ctx = build_data_context(df)
                    prompt = f"Dados:\n{ctx}\n\nPergunta: {s}\n\nResponde em português de Portugal."
                    answer = call_ai(prompt)
                    st.markdown(
                        f'<div class="ai-answer">💬 <strong>Resposta:</strong><br><br>{answer}</div>',
                        unsafe_allow_html=True,
                    )
