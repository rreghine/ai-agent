import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import time
import json
import os

st.set_page_config(
    page_title="E-Commerce AI Agent | GenAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


A1 = '#0a1628'
A2 = '#1e3a5f'
A3 = '#2e6fad'
A4 = '#4a90c4'
A5 = '#6baed6'
A6 = '#c6dbef'
BR = '#e8f1fa'
WH = '#ffffff'
BD = '#dce8f5'
TX = '#2c3e50'
TL = '#7f8c8d'
VR = '#c0392b'
AM = '#e67e22'


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: {WH};
        color: {TX};
    }}
    .main {{ background-color: {WH}; }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}

    /* ── Header ── */
    .header-wrap {{
        background: linear-gradient(135deg, {A1} 0%, {A2} 60%, {A3} 100%);
        padding: 16px 22px 14px 22px;
        border-radius: 6px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        box-sizing: border-box;
    }}
    .header-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {WH};
        margin: 0;
    }}
    .header-badge {{
        font-size: 0.62rem;
        font-weight: 700;
        background: rgba(255,255,255,0.2);
        color: {WH};
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 3px;
        padding: 2px 7px;
        letter-spacing: 1px;
        margin-left: 8px;
        vertical-align: middle;
    }}
    .header-sub {{
        font-size: 0.71rem;
        color: rgba(255,255,255,0.55);
        margin-top: 4px;
    }}
    .header-right {{
        font-size: 0.7rem;
        color: rgba(255,255,255,0.45);
        text-align: right;
    }}

    /* ── KPI Cards ── */
    .kpi-card {{
        background: {WH};
        border: 1px solid {BD};
        border-top: 3px solid var(--c);
        border-radius: 6px;
        padding: 14px 16px 12px;
        box-shadow: 0 1px 4px rgba(10,22,40,0.05);
    }}
    .kpi-label {{
        font-size: 0.58rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: {TL};
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1;
        color: var(--c);
        letter-spacing: -0.5px;
    }}
    .kpi-sub {{
        font-size: 0.63rem;
        color: {TL};
        margin-top: 4px;
    }}

    /* ── Seção ── */
    .sec-label {{
        font-size: 0.6rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {A4};
        border-bottom: 2px solid {A6};
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }}

    /* ── Resposta ── */
    .resp-box {{
        background: {WH};
        border: 1px solid {BD};
        border-left: 4px solid {A3};
        border-radius: 0 6px 6px 0;
        padding: 16px 20px;
        font-size: 0.88rem;
        color: {TX};
        line-height: 1.75;
    }}
    .guard-box {{
        background: #fff8f0;
        border: 1px solid #fde8cc;
        border-left: 4px solid {AM};
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: #7d4e1a;
    }}

    /* ── Eval row ── */
    .eval-row {{
        display: flex;
        border: 1px solid {BD};
        border-radius: 6px;
        overflow: hidden;
        margin-top: 10px;
    }}
    .eval-cell {{
        flex: 1;
        padding: 9px 13px;
        border-right: 1px solid {BD};
        background: {BR};
    }}
    .eval-cell:last-child {{ border-right: none; }}
    .eval-lbl {{
        font-size: 0.57rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: {TL};
        margin-bottom: 3px;
    }}
    .eval-val {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {TX};
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 2px solid {BD};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.82rem;
        font-weight: 600;
        padding: 10px 22px;
        color: {TL};
    }}
    .stTabs [aria-selected="true"] {{
        color: {A3} !important;
        border-bottom: 2px solid {A3} !important;
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


def get_api_key():
    key = os.environ.get('GEMINI_API_KEY')
    if key:
        return key
    try:
        key = st.secrets.get('GEMINI_API_KEY')
        if key:
            return key
    except Exception:
        pass
    try:
        from google.colab import userdata
        key = userdata.get('GEMINI_API_KEY')
        if key:
            return key
    except Exception:
        pass
    return None

try:
    from google import genai as _genai
    _key = get_api_key()
    client = _genai.Client(api_key=_key) if _key else None
except Exception:
    client = None

MODEL_NAME = 'gemma-3-27b-it'

# ── Base de conhecimento ───────────────────────────────────────────────────
@st.cache_data
def load_knowledge_base():
    if os.path.exists('knowledge_base.json'):
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['documents'], data['metricas']

    paths = ['./data/', '/content/drive/MyDrive/Agente de IA/dataset/']
    df_path = next((p for p in paths if os.path.exists(p)), None)
    if df_path is None:
        return None, None

    try:
        orders      = pd.read_csv(df_path + 'olist_orders_dataset.csv')
        customers   = pd.read_csv(df_path + 'olist_customers_dataset.csv')
        items       = pd.read_csv(df_path + 'olist_order_items_dataset.csv')
        payments    = pd.read_csv(df_path + 'olist_order_payments_dataset.csv')
        reviews     = pd.read_csv(df_path + 'olist_order_reviews_dataset.csv')
        products    = pd.read_csv(df_path + 'olist_products_dataset.csv')
        sellers     = pd.read_csv(df_path + 'olist_sellers_dataset.csv')
        translation = pd.read_csv(df_path + 'product_category_name_translation.csv')
        products    = products.merge(translation, on='product_category_name', how='left')

        docs        = []
        df_merged   = orders.merge(customers, on='customer_id')
        top_estados = df_merged['customer_state'].value_counts().head(10)
        payment_types = payments['payment_type'].value_counts()
        avg_score   = reviews['review_score'].mean()
        df_ip       = items.merge(products, on='product_id')

        docs.append(f"RESUMO:\n- Pedidos: {orders.shape[0]:,}\n- Clientes: {customers['customer_unique_id'].nunique():,}\n- Ticket médio: R$ {items['price'].mean():.2f}\n- Período: 2016–2018")
        docs.append(f"STATUS:\n{orders['order_status'].value_counts().to_string()}")
        docs.append(f"TOP ESTADOS:\n{top_estados.to_string()}")
        docs.append(f"PAGAMENTOS:\n{payment_types.to_string()}\nMédia parcelas: {payments['payment_installments'].mean():.1f}")
        docs.append(f"AVALIAÇÕES:\nNota média: {avg_score:.2f}/5.0\n{reviews['review_score'].value_counts().sort_index().to_string()}")
        docs.append(f"TOP CATEGORIAS:\n{df_ip.groupby('product_category_name_english')['price'].agg(['sum','count']).sort_values('sum',ascending=False).head(10).to_string()}")

        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
        ent = orders[orders['order_status'] == 'delivered'].copy()
        ent['atrasado'] = ent['order_delivered_customer_date'] > ent['order_estimated_delivery_date']
        taxa = ent['atrasado'].mean() * 100
        docs.append(f"ATRASOS:\nTaxa: {taxa:.1f}%\n{ent.merge(customers,on='customer_id').groupby('customer_state')['atrasado'].mean().sort_values(ascending=False).head(10).mul(100).round(1).to_string()}")

        top_sell = items.merge(sellers, on='seller_id').groupby('seller_state')['price'].sum().sort_values(ascending=False).head(10)
        docs.append(f"VENDEDORES POR ESTADO:\n{top_sell.to_string()}")

        rec = df_ip.groupby('product_category_name_english')['price'].sum()
        met = {
            'total_pedidos':         int(orders.shape[0]),
            'ticket_medio':          round(float(items['price'].mean()), 2),
            'nota_media':            round(float(avg_score), 2),
            'taxa_atraso':           round(float(taxa), 1),
            'top_categoria':         str(rec.idxmax()),
            'categoria_menos':       str(rec.idxmin()),
            'top_estado':            str(top_estados.index[0]),
            'forma_pagamento':       str(payment_types.index[0]),
            'forma_pagamento_menos': str(payment_types.index[-1]),
        }
        return docs, met
    except Exception as e:
        st.error(f'Erro: {e}')
        return None, None


@st.cache_resource
def init_rag(documents):
    if not documents:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        m  = SentenceTransformer('all-MiniLM-L6-v2')
        em = m.encode(documents)
        return m, em
    except:
        return None

# ── Guardrails ─────────────────────────────────────────────────────────────
TOPICOS = ['pedido','cliente','produto','venda','pagamento','entrega','frete',
           'avalia','nota','categoria','estado','cidade','vendedor','receita',
           'valor','preco','preço','atraso','status','ticket','compra','review',
           'ecommerce','marketplace','total','média','media','quantos','quanto']
BLOQ = ['cpf','rg','senha','password','dados pessoais','endereço completo','nome completo']

def validar(p):
    pl = p.lower()
    if len(p.strip()) < 5:
        return {'ok': False, 'msg': '🛡️ Pergunta muito curta.'}
    if len(p) > 500:
        return {'ok': False, 'msg': '🛡️ Pergunta muito longa.'}
    for b in BLOQ:
        if b in pl:
            return {'ok': False, 'msg': '🛡️ Não forneço dados pessoais.'}
    if not any(t in pl for t in TOPICOS):
        return {'ok': False, 'msg': '🛡️ Só respondo sobre e-commerce (pedidos, produtos, pagamentos, entregas).'}
    return {'ok': True, 'msg': ''}

# ── Ground Truth ───────────────────────────────────────────────────────────
CAT_PT = {
    'health_beauty':'Saúde e Beleza','bed_bath_table':'Cama, Mesa e Banho',
    'sports_leisure':'Esportes e Lazer','computers_accessories':'Computadores e Acessórios',
    'furniture_decor':'Móveis e Decoração','housewares':'Utilidades Domésticas',
    'watches_gifts':'Relógios e Presentes','telephony':'Telefonia',
    'toys':'Brinquedos','electronics':'Eletrônicos','auto':'Automóveis',
}
PAG_PT = {
    'credit_card':'Cartão de crédito','boleto':'Boleto bancário',
    'voucher':'Voucher','debit_card':'Cartão de débito','not_defined':'Não definido',
}

def ground_truth(p, m):
    if not m: return None
    pl = p.lower()
    if any(x in pl for x in ['mais pedidos','mais compras']):
        return 'SP (São Paulo) com maior volume de pedidos'
    if any(x in pl for x in ['total de pedidos','quantos pedidos']):
        return f"{m['total_pedidos']:,} pedidos"
    if any(x in pl for x in ['ticket médio','valor médio','preço médio']):
        return f"R$ {m['ticket_medio']}"
    if any(x in pl for x in ['pagamento','forma de pagamento']):
        if any(x in pl for x in ['menos','menor','mais raro']):
            return f"{PAG_PT.get(m['forma_pagamento_menos'], m['forma_pagamento_menos'])} é a menos usada"
        return f"{PAG_PT.get(m['forma_pagamento'], m['forma_pagamento'])} é a mais usada"
    if any(x in pl for x in ['nota média','avaliação média']):
        return f"{m['nota_media']} de 5.0"
    if any(x in pl for x in ['atraso','taxa de atraso']):
        return f"{m['taxa_atraso']}% dos pedidos"
    if any(x in pl for x in ['categoria','mais vendida']):
        if any(x in pl for x in ['menos','menor']):
            return f"{CAT_PT.get(m['categoria_menos'], m['categoria_menos'])} menor receita"
        return f"{CAT_PT.get(m['top_categoria'], m['top_categoria'])} maior receita"
    return None

def avaliar(resp, gt):
    if not gt: return {'status': 'sem_ground_truth', 'score': None}
    pt = set(gt.lower().split())
    pr = set(resp.lower().split())
    ov = len(pt & pr) / len(pt) if pt else 0
    if ov >= 0.5:  return {'status': 'correta',    'score': 1.0}
    if ov >= 0.25: return {'status': 'parcial',    'score': 0.5}
    return             {'status': 'alucinacao', 'score': 0.0}

def custo(ti, to):
    g = (ti / 1000 * 0.03) + (to / 1000 * 0.06)
    return {'total_tokens': ti + to, 'custo_gpt4': round(g, 6)}

# ── Responder ──────────────────────────────────────────────────────────────
def responder(pergunta, docs, met, rag_m, rag_e):
    t0  = time.time()
    val = validar(pergunta)
    if not val['ok']:
        return {'pergunta': pergunta, 'resposta': val['msg'],
                'guardrail': True, 'latencia': int((time.time()-t0)*1000),
                'custo': None, 'aval': None, 'gt': None}

    if rag_m and rag_e is not None:
        qe  = rag_m.encode([pergunta])
        sc  = np.dot(rag_e, qe.T).flatten()
        idx = np.argsort(sc)[::-1][:3]
        ctx = '\n---\n'.join([docs[i] for i in idx])
    else:
        ctx = '\n---\n'.join(docs[:3]) if docs else ''

    prompt = f"""Analista de dados de e-commerce brasileiro.
REGRAS: responda em português, traduza credit_card=cartão de crédito,
health_beauty=saúde e beleza, use números do contexto, máximo 3 linhas.

CONTEXTO: {ctx}
PERGUNTA: {pergunta}
RESPOSTA:"""

    try:
        r   = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        txt = r.text
        ti  = int(len(prompt.split()) * 1.3)
        to  = int(len(txt.split()) * 1.3)
    except Exception as e:
        txt = f'Erro: {e}'
        ti = to = 0

    gt = ground_truth(pergunta, met)
    return {
        'pergunta': pergunta, 'resposta': txt, 'guardrail': False,
        'gt': gt, 'aval': avaliar(txt, gt), 'custo': custo(ti, to),
        'latencia': int((time.time() - t0) * 1000),
    }

# ══════════════════════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════════════════════

# ── Sidebar — nativa, sem CSS customizado ──────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuração")
    st.code(MODEL_NAME, language=None)

    if client:
        st.success("✅ Google Gemma conectado")
    else:
        st.error("❌ API Key não encontrada")
        st.info("Defina `GEMINI_API_KEY` nos Secrets.")

    st.divider()
    st.markdown("**Exemplos de perguntas:**")

    exemplos = [
        "Qual estado tem mais pedidos?",
        "Qual é o ticket médio?",
        "Qual a forma de pagamento mais usada?",
        "Qual a forma de pagamento menos usada?",
        "Qual categoria gera mais receita?",
        "Qual a taxa de atraso nas entregas?",
        "Qual a nota média de avaliação?",
        "Quantos pedidos existem no total?",
    ]
    for ex in exemplos:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state['q'] = ex

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#888;text-align:center'>"
        "Rafael Reghine Munhoz<br>Data Analyst | MBA USP<br>"
        "<a href='https://github.com/rreghine' style='color:#2e6fad'>"
        "github.com/rreghine</a></div>",
        unsafe_allow_html=True
    )

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-wrap">
    <div>
        <div class="header-title">
            E-Commerce AI Agent
            <span class="header-badge">GenAI</span>
        </div>
        <div class="header-sub">
            Google Gemma &nbsp;·&nbsp; RAG &nbsp;·&nbsp; Guardrails &nbsp;·&nbsp;
            Hallucination Evaluation &nbsp;·&nbsp; Cost Analysis &nbsp;·&nbsp; MLflow
        </div>
    </div>
    <div class="header-right">
        github.com/rreghine<br>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Carregar dados ─────────────────────────────────────────────────────────
with st.spinner('Carregando base de conhecimento...'):
    docs, met = load_knowledge_base()

if docs is None:
    st.warning("⚠️ Dataset não encontrado. Coloque os CSVs na pasta `data/`.")
    st.stop()

rag = init_rag(docs)
rag_m, rag_e = (rag if rag else (None, None))

if 'hist' not in st.session_state:
    st.session_state['hist'] = []

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  💬  Agente  ", "  📊  Dashboard  "])

# ══════════════════════════════════════════════════════════════════════════
# ABA 1 — AGENTE
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    default = st.session_state.pop('q', '')
    c1, c2  = st.columns([5, 1])
    with c1:
        perg = st.text_input("p", value=default,
                             placeholder="Ex: Qual estado tem mais pedidos?",
                             label_visibility="collapsed")
    with c2:
        send = st.button("Enviar →", type="primary", use_container_width=True)

    if send and perg.strip():
        if not client:
            st.error("Configure a API Key.")
        else:
            with st.spinner('Consultando...'):
                res = responder(perg, docs, met, rag_m, rag_e)
                st.session_state['hist'].insert(0, res)

    if st.session_state['hist']:
        st.markdown('<div class="sec-label">Histórico de Conversas</div>',
                    unsafe_allow_html=True)

        for i, r in enumerate(st.session_state['hist']):
            with st.expander(f"**{r['pergunta']}**", expanded=(i == 0)):

                if r.get('guardrail'):
                    st.markdown(f'<div class="guard-box">{r["resposta"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="resp-box">{r["resposta"]}</div>',
                                unsafe_allow_html=True)

                    av = r.get('aval', {})
                    st_map = {
                        'correta':          f'<span style="color:{A3}">✔ Correta</span>',
                        'parcial':          f'<span style="color:{AM}">◑ Parcial</span>',
                        'alucinacao':       f'<span style="color:{VR}">✘ Alucinação</span>',
                        'sem_ground_truth': '— Sem GT',
                    }
                    c  = r.get('custo') or {}
                    st.markdown(f"""
                    <div class="eval-row">
                        <div class="eval-cell">
                            <div class="eval-lbl">Ground Truth</div>
                            <div class="eval-val">{r.get('gt') or '—'}</div>
                        </div>
                        <div class="eval-cell">
                            <div class="eval-lbl">Avaliação</div>
                            <div class="eval-val">{st_map.get(av.get('status',''), '—')}</div>
                        </div>
                        <div class="eval-cell">
                            <div class="eval-lbl">Tokens</div>
                            <div class="eval-val">{c.get('total_tokens',0):,}</div>
                        </div>
                        <div class="eval-cell">
                            <div class="eval-lbl">Latência</div>
                            <div class="eval-val">{r.get('latencia',0)}ms</div>
                        </div>
                        <div class="eval-cell">
                            <div class="eval-lbl">GPT-4 equiv.</div>
                            <div class="eval-val">${c.get('custo_gpt4',0):.5f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Limpar histórico"):
            st.session_state['hist'] = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# ABA 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state['hist']:
        st.info("Faça perguntas na aba Agente para ver as métricas.")
    else:
        validas = [r for r in st.session_state['hist'] if not r.get('guardrail')]
        todos   = st.session_state['hist']

        total   = len(todos)
        corr    = sum(1 for r in validas if r.get('aval') and r['aval']['status'] == 'correta')
        guards  = sum(1 for r in todos if r.get('guardrail'))
        tok_t   = sum(r['custo']['total_tokens'] for r in validas if r.get('custo'))
        lat_m   = np.mean([r['latencia'] for r in todos]) if todos else 0
        cg4     = sum(r['custo']['custo_gpt4'] for r in validas if r.get('custo'))
        com_gt  = [r for r in validas if r.get('aval') and r['aval']['status'] in ['correta','parcial','alucinacao']]
        t_hal   = (sum(1 for r in com_gt if r['aval']['status'] == 'alucinacao') / len(com_gt) * 100) if com_gt else 0

        # KPI Cards
        kpis = [
            ("Total Queries",    str(total),        A3),
            ("Corretas",         str(corr),         A2),
            ("Taxa Alucinação",  f"{t_hal:.1f}%",   VR),
            ("Guardrails",       str(guards),       A4),
            ("Tokens",           f"{tok_t:,}",      A3),
            ("Latência Média",   f"{lat_m:.0f}ms",  AM),
        ]
        cols = st.columns(6)
        for col, (lbl, val, cor) in zip(cols, kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="--c:{cor}">
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-value">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if len(validas) >= 2:
            cmap  = LinearSegmentedColormap.from_list('b', [A6, A5, A4, A3, A2, A1])
            BORDA = BD

            plt.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.facecolor': WH,
                'figure.facecolor': WH,
                'axes.grid': True,
                'grid.color': '#edf2f7',
                'grid.linewidth': 0.7,
                'axes.axisbelow': True,
            })

            aval_map = {
                'correta': 'Correta', 'parcial': 'Parcial',
                'alucinacao': 'Alucinação', 'sem_ground_truth': 'Sem GT'
            }
            aval_cor = {
                'correta': A3, 'parcial': A4,
                'alucinacao': VR, 'sem_ground_truth': A6
            }

            fig = plt.figure(figsize=(16, 9), facecolor=WH)
            gs  = gridspec.GridSpec(2, 3, figure=fig,
                                    hspace=0.55, wspace=0.35,
                                    top=0.88, bottom=0.10,
                                    left=0.06, right=0.97)

            fig.text(0.5, 0.94, 'Performance Dashboard',
                     ha='center', fontsize=13, fontweight='bold', color=A1)

            # G1 — Avaliação
            ax1   = fig.add_subplot(gs[0, 0])
            avsts = [r['aval']['status'] for r in validas if r.get('aval')]
            avcnt = pd.Series(avsts).value_counts()
            lbls  = [aval_map.get(k, k) for k in avcnt.index]
            cors  = [aval_cor.get(k, A3) for k in avcnt.index]
            bars  = ax1.barh(lbls, avcnt.values, color=cors, edgecolor=WH, height=0.5)
            for b, v in zip(bars, avcnt.values):
                ax1.text(b.get_width()+0.05, b.get_y()+b.get_height()/2,
                         str(v), va='center', fontsize=10, fontweight='700', color=A1)
            ax1.set_title('Avaliação das Respostas', fontweight='bold', color=A1, fontsize=11, pad=10)
            ax1.set_xlim(0, avcnt.max() * 1.3)
            for sp in ['top','right']: ax1.spines[sp].set_visible(False)
            ax1.spines['left'].set_color(BORDA)
            ax1.spines['bottom'].set_color(BORDA)
            ax1.tick_params(colors=TL, labelsize=9)

            # G2 — Tokens
            ax2  = fig.add_subplot(gs[0, 1])
            toks = [r['custo']['total_tokens'] for r in validas if r.get('custo')]
            nrm  = np.array(toks) / max(toks)
            bars2 = ax2.bar(range(len(toks)), toks,
                            color=[cmap(v) for v in nrm], edgecolor=WH, width=0.7)
            for b, v in zip(bars2, toks):
                ax2.text(b.get_x()+b.get_width()/2, b.get_height()+max(toks)*0.015,
                         f'{int(v):,}', ha='center', va='bottom',
                         fontsize=8, fontweight='600', color=A1, rotation=35)
            med = np.mean(toks)
            ax2.axhline(med, color=VR, lw=1.5, ls='--', label=f'Média: {med:,.0f}')
            ax2.legend(fontsize=8)
            ax2.set_title('Tokens por Query', fontweight='bold', color=A1, fontsize=11, pad=10)
            ax2.set_xlabel('Query #', fontsize=9, color=TL)
            ax2.set_ylabel('Tokens', fontsize=9, color=TL)
            for sp in ['top','right']: ax2.spines[sp].set_visible(False)
            ax2.spines['left'].set_color(BORDA)
            ax2.spines['bottom'].set_color(BORDA)
            ax2.tick_params(colors=TL, labelsize=8)

            # G3 — Latência
            ax3  = fig.add_subplot(gs[0, 2])
            lats = [r['latencia'] for r in todos]
            ax3.fill_between(range(len(lats)), lats, alpha=0.12, color=A4)
            ax3.plot(range(len(lats)), lats, color=A3, lw=2.5,
                     marker='o', ms=6, mfc=WH, mec=A3, mew=2)
            for x, y in enumerate(lats):
                ax3.text(x, y+max(lats)*0.04, f'{int(y)}ms',
                         ha='center', va='bottom', fontsize=8, fontweight='600', color=A1)
            ax3.axhline(np.mean(lats), color=VR, lw=1.5, ls='--',
                        label=f'Média: {np.mean(lats):.0f}ms')
            ax3.legend(fontsize=8)
            ax3.set_title('Latência por Query (ms)', fontweight='bold', color=A1, fontsize=11, pad=10)
            ax3.set_xlabel('Query #', fontsize=9, color=TL)
            for sp in ['top','right']: ax3.spines[sp].set_visible(False)
            ax3.spines['left'].set_color(BORDA)
            ax3.spines['bottom'].set_color(BORDA)
            ax3.tick_params(colors=TL, labelsize=8)

            # G4 — Custo
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.bar(['Gemma\n(free tier)', 'GPT-4\n(estimado)'],
                    [0.0, cg4], color=[A3, VR], edgecolor=WH, width=0.45)
            ax4.text(0, max(cg4*0.05, 0.000001), '$0.00\nGratuito',
                     ha='center', va='bottom', fontsize=10, fontweight='700', color=A3)
            ax4.text(1, cg4+cg4*0.05, f'${cg4:.5f}',
                     ha='center', va='bottom', fontsize=10, fontweight='700', color=VR)
            ax4.text(0.5, max(cg4*0.5, 0.000002), '100%\nEconomia',
                     ha='center', va='center', fontsize=11, fontweight='700', color=A2,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=BR, edgecolor=A4, lw=1))
            ax4.set_title('Custo — Gemma vs GPT-4', fontweight='bold', color=A1, fontsize=11, pad=10)
            ax4.set_ylabel('USD', fontsize=9, color=TL)
            for sp in ['top','right']: ax4.spines[sp].set_visible(False)
            ax4.spines['left'].set_color(BORDA)
            ax4.spines['bottom'].set_color(BORDA)
            ax4.tick_params(colors=TL, labelsize=9)

            # G5 — Donut
            ax5 = fig.add_subplot(gs[1, 1])
            sc   = pd.Series(avsts).value_counts()
            w, t, at = ax5.pie(
                sc.values,
                labels=[f'{aval_map.get(k,k)} ({v})' for k,v in sc.items()],
                colors=[aval_cor.get(k, A3) for k in sc.index],
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.55, edgecolor=WH, linewidth=2),
                textprops=dict(fontsize=8, color=A1)
            )
            for a in at:
                a.set_fontsize(8); a.set_fontweight('bold'); a.set_color(WH)
            ax5.set_title('Distribuição de Status', fontweight='bold', color=A1, fontsize=11, pad=10)

            # G6 — Custo acumulado
            ax6  = fig.add_subplot(gs[1, 2])
            cacum = np.cumsum([r['custo']['custo_gpt4'] for r in validas if r.get('custo')])
            mc   = max(cacum.max(), 0.000001)
            bars6 = ax6.bar(range(len(cacum)), cacum,
                            color=[cmap(v/mc) for v in cacum], edgecolor=WH, width=0.7)
            for b, v in zip(bars6, cacum):
                ax6.text(b.get_x()+b.get_width()/2, b.get_height()+mc*0.02,
                         f'${v:.5f}', ha='center', va='bottom',
                         fontsize=7, fontweight='600', color=A1, rotation=30)
            ax6.set_title('Custo Acumulado (GPT-4 equiv.)', fontweight='bold', color=A1, fontsize=11, pad=10)
            ax6.set_xlabel('Query #', fontsize=9, color=TL)
            ax6.set_ylabel('USD', fontsize=9, color=TL)
            for sp in ['top','right']: ax6.spines[sp].set_visible(False)
            ax6.spines['left'].set_color(BORDA)
            ax6.spines['bottom'].set_color(BORDA)
            ax6.tick_params(colors=TL, labelsize=8)

            # Rodapé
            fig.add_artist(plt.Line2D([0.06,0.97],[0.065,0.065],
                           transform=fig.transFigure, color=BORDA, lw=1))
            fig.text(0.5, 0.038,
                     'Rafael Reghine Munhoz  ·  Data Analyst | MBA USP',
                     ha='center', fontsize=8, fontweight='600', color=A2)
            fig.text(0.5, 0.018,
                     'github.com/rreghine  ·  linkedin.com/in/rafaelreghine',
                     ha='center', fontsize=8, color=A4)

            st.pyplot(fig)
        else:
            st.info("Faça pelo menos 2 perguntas para ver os gráficos.")

        # Tabela
        st.markdown('<div class="sec-label">Histórico Detalhado</div>',
                    unsafe_allow_html=True)
        df_h = pd.DataFrame([{
            'Pergunta':    r['pergunta'][:55]+'...' if len(r['pergunta'])>55 else r['pergunta'],
            'Avaliação':   r['aval']['status'] if r.get('aval') else 'guardrail',
            'Tokens':      r['custo']['total_tokens'] if r.get('custo') else 0,
            'Latência':    f"{r['latencia']}ms",
            'GPT-4 equiv': f"${r['custo']['custo_gpt4']:.5f}" if r.get('custo') else '—',
        } for r in st.session_state['hist']])
        st.dataframe(df_h, use_container_width=True, hide_index=True)
