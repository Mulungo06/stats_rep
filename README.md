# 📊 DataAI Analyzer

Analisa qualquer base de dados com IA. Suporta **Google Gemini (grátis)**, OpenAI e Anthropic Claude.

## 🚀 Deploy rápido

### 1. GitHub
```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/SEU_USER/dataai-analyzer.git
git push -u origin main
```

### 2. Streamlit Cloud
1. Vai a [share.streamlit.io](https://share.streamlit.io)
2. **New app** → seleciona o repositório → `app.py` → Deploy
3. **Settings → Secrets** → cola a tua chave API

### 3. Chaves API (escolhe uma)

| Provider | Custo | Onde obter |
|----------|-------|------------|
| **Google Gemini** | **Grátis** | [aistudio.google.com](https://aistudio.google.com) |
| OpenAI | A partir de $5 | [platform.openai.com](https://platform.openai.com) |
| Anthropic | A partir de $5 | [console.anthropic.com](https://console.anthropic.com) |

Cola nos Secrets do Streamlit Cloud:
```toml
# Gemini (grátis)
GOOGLE_API_KEY = "AIza..."

# OU OpenAI
OPENAI_API_KEY = "sk-..."

# OU Anthropic
ANTHROPIC_API_KEY = "sk-ant-..."
```

## 💻 Correr localmente

```bash
pip install -r requirements.txt

# Edita .streamlit/secrets.toml com a tua chave
streamlit run app.py
```

## ✨ Funcionalidades

- Upload CSV, Excel, JSON, TSV
- Resumo executivo automático com IA
- 8 tipos de gráficos interativos (Plotly)
- Filtro e pesquisa na tabela
- Perguntas em linguagem natural
- Exportação de gráficos
