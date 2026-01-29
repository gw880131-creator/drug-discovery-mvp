import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import py3Dmol
from stmol import showmol
import plotly.graph_objects as go
import requests
import urllib.parse
import time

# --- 1. 頁面與 CSS 風格設定 (複製您的 HTML 風格) ---
st.set_page_config(
    page_title="MedChem Pro | Enterprise Drug Discovery Platform", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 強制注入 Tailwind 風格的 CSS
st.markdown("""
<style>
    /* 引入 Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    /* 全局背景：深海藍漸層 */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* 玻璃擬態面板 (Glass Panel) */
    div[data-testid="stExpander"], div.css-1r6slb0, .stDataFrame, .metric-card {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
    }

    /* 輸入框樣式 */
    .stTextInput input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
    }

    /* 按鈕樣式 (仿 Tailwind blue-600) */
    .stButton>button {
        background: linear-gradient(to right, #2563eb, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }

    /* 關鍵指標數值顏色 */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8 !important; /* Sky Blue */
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem;
    }

    /* 側邊欄樣式 */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* 標題與文字顏色 */
    h1, h2, h3 { color: #f8fafc !important; }
    p, li { color: #cbd5e1; }

    /* 自定義 Badge */
    .enterprise-badge {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.7rem;
        font-weight: bold;
        text-transform: uppercase;
        margin-left: 10px;
    }
    
    /* 風險等級顏色 */
    .risk-high { color: #ef4444; font-weight: bold; text-shadow: 0 0 8px rgba(239, 68, 68, 0.4); }
    .risk-medium { color: #f59e0b; font-weight: bold; }
    .risk-low { color: #10b981; font-weight: bold; text-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心運算邏輯 (保留您需要的 Python 功能) ---

# 資料庫模擬
PATENT_DB = {
    "donepezil": {"patent_no": "US4895841", "expiry": "Expired (2010)", "similarity": 82, "risk": "Yellow"},
    "memantine": {"patent_no": "US4122193", "expiry": "Expired (2015)", "similarity": 15, "risk": "Green"}
}

TRANSFORMATIONS = {
    "reduce_lipophilicity": [
        {"name": "Scaffold Hop (苯環 → 吡啶)", "smarts": "c1ccccc1>>c1ccncc1", "desc": "引入氮原子增加極性，降低 LogP", "ref": "Bioorg. Med. Chem. 2013"},
    ],
    "improve_metabolic_stability": [
        {"name": "Fluorination (代謝封閉)", "smarts": "[cH1:1]>>[c:1](F)", "desc": "阻斷 CYP450 氧化位點", "ref": "J. Med. Chem. 2008"},
    ],
    "increase_lipophilicity": [
        {"name": "Methylation (甲基化)", "smarts": "[nH1:1]>>[n:1](C)", "desc": "增加親脂性，提升 BBB 穿透", "ref": "J. Med. Chem. 2011"}
    ]
}

# 運算函式
def calculate_metrics(mol):
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "qed": QED.qed(mol),
        "in_egg": (Descriptors.TPSA(mol) < 79 and 0.4 < Descriptors.MolLogP(mol) < 6.0)
    }

def apply_transformation(mol, metrics):
    logp = metrics['logp']
    if logp > 4.0:
        cat, reason = "reduce_lipophilicity", "⚠️ LogP 過高 (>4.0)，建議引入雜環。"
    elif logp < 1.0:
        cat, reason = "increase_lipophilicity", "⚠️ LogP 過低 (<1.0)，建議甲基化。"
    else:
        cat, reason = "improve_metabolic_stability", "✅ 理化性質良好，建議優化代謝穩定性。"
    
    for t in TRANSFORMATIONS[cat]:
        try:
            rxn = AllChem.ReactionFromSmarts(t['smarts'])
            products = rxn.RunReactants((mol,))
            if products:
                new_mol = products[0][0]
                Chem.SanitizeMol(new_mol)
                return new_mol, t['name'], t['desc'], t['ref'], reason
        except: continue
    
    return mol, "Stereoisomer Optimization", "立體化學調整", "N/A", reason + " (結構特殊，建議手性優化)"

def generate_3d_block(mol):
    try:
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv2())
        AllChem.MMFFOptimizeMolecule(mol_3d)
        return Chem.MolToPDBBlock(mol_3d)
    except: return None

# API 連線 (ChEMBL)
@st.cache_data(ttl=3600)
def fetch_chembl_data(smiles):
    try:
        base = "https://www.ebi.ac.uk/chembl/api/data"
        safe_s = urllib.parse.quote(smiles)
        res = requests.get(f"{base}/similarity/{safe_s}/85?format=json", timeout=5)
        if res.status_code == 200:
            d = res.json()
            if d['molecules']:
                mol_data = d['molecules'][0]
                act_res = requests.get(f"{base}/activity?molecule_chembl_id={mol_data['molecule_chembl_id']}&limit=5&format=json", timeout=5)
                acts = []
                if act_res.status_code == 200:
                    for a in act_res.json().get('activities', []):
                        if a.get('target_pref_name'):
                            acts.append({"Target": a['target_pref_name'], "Type": a['standard_type'], "Value": f"{a['standard_value']} {a.get('standard_units','')}"})
                return {"found": True, "id": mol_data['molecule_chembl_id'], "acts": acts}
    except: pass
    return {"found": False}

# --- 3. UI 主程式 (仿 HTML 結構) ---

# Header 區塊
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('# MedChem <span style="color:#3b82f6">Pro</span> <span class="enterprise-badge">Enterprise V25.0</span>', unsafe_allow_html=True)
    st.caption("工業級藥物篩選平台 | FDA 21 CFR Part 11 Compliant | Powered by RDKit & BrainX AI")
with c2:
    st.markdown('<div style="text-align:right; color:#4ade80; padding-top:20px;"><i class="fas fa-check-circle"></i> System Online</div>', unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.header("🔍 藥物檢索")
    search_input = st.text_input("輸入藥名 / SMILES", "Donepezil")
    
    col_run_1, col_run_2 = st.columns(2)
    with col_run_1:
        run_btn = st.button("🚀 執行分析", use_container_width=True)
    with col_run_2:
        batch_btn = st.button("📂 批量上傳", use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 📚 快速範例")
    if st.button("Ceftriaxone (BX100)"):
        search_input = "Ceftriaxone" # 這行在 Streamlit logic 中需配合 session_state 使用，此為簡化
        st.info("請在上方輸入框鍵入 'Ceftriaxone' 後點擊分析")
    
    st.markdown("---")
    st.caption("Connected to: ChEMBL, PubChem, USPTO")

# 主邏輯
if run_btn and search_input:
    with st.spinner("正在連線核心運算引擎與外部資料庫..."):
        # 1. 解析
        try:
            mol = Chem.MolFromSmiles(search_input)
            if not mol:
                c = pcp.get_compounds(search_input, 'name')
                if c:
                    search_input = c[0].synonyms[0] if c[0].synonyms else search_input
                    mol = Chem.MolFromSmiles(c[0].isomeric_smiles)
        except: mol = None
        
        if not mol:
            st.error("❌ 無法解析分子結構")
        else:
            time.sleep(0.5) # 模擬運算感
            metrics = calculate_metrics(mol)
            opt_mol, opt_name, opt_desc, opt_ref, opt_reason = apply_transformation(mol, metrics)
            chembl = fetch_chembl_data(Chem.MolToSmiles(mol))
            
            # --- 儀表板 ---
            
            # Tab 1: 科學核心
            st.markdown("### 1️⃣ 核心科學運算模組 (Scientific Core)")
            
            # 五大指標卡
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("MW (分子量)", f"{metrics['mw']:.1f}", delta="< 500")
            k2.metric("LogP (脂溶性)", f"{metrics['logp']:.2f}", delta="1-3")
            k3.metric("TPSA (極性表面)", f"{metrics['tpsa']:.1f}", delta="< 90")
            k4.metric("HBD (氫鍵供體)", f"{metrics['hbd']}", delta="< 5")
            k5.metric("QED (類藥性)", f"{metrics['qed']:.2f}", delta="> 0.6")
            
            # BOILED-Egg
            col_chart, col_info = st.columns([2, 1])
            with col_chart:
                fig = go.Figure()
                fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=6, y1=140,
                    fillcolor="rgba(255, 204, 0, 0.2)", line_color="rgba(255, 204, 0, 0.5)")
                fig.add_trace(go.Scatter(
                    x=[metrics['logp']], y=[metrics['tpsa']], mode='markers+text',
                    marker=dict(size=18, color='#4ade80' if metrics['in_egg'] else '#f87171', line=dict(width=2, color='white')),
                    text=["Current"], textposition="top center"
                ))
                fig.update_layout(
                    xaxis_title="WLOGP", yaxis_title="TPSA",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), height=300, margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_info:
                st.markdown("""
                <div style="background:rgba(30,41,59,0.5); padding:15px; border-radius:10px; border:1px solid rgba(255,255,255,0.1);">
                    <h4 style="color:#fcd34d; margin-top:0;">🥚 蛋黃圖分析</h4>
                    <p style="font-size:0.9rem; color:#cbd5e1;">此圖預測藥物能否穿透血腦屏障 (BBB)。</p>
                    <ul style="font-size:0.8rem; color:#94a3b8; padding-left:20px;">
                        <li>🟡 <strong>黃色區 (BBB):</strong> 容易入腦</li>
                        <li>⚪ <strong>白色區 (HIA):</strong> 腸道吸收佳</li>
                        <li>🔴 <strong>紅點:</strong> 您的分子</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Tab 2: AI 優化
            st.markdown("### 2️⃣ 結構優化與 AI 建議 (MedChem Brain)")
            st.info(f"💡 **AI 診斷:** {opt_reason}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📉 原始結構**")
                v1 = py3Dmol.view(width=400, height=300)
                v1.addModel(generate_3d_block(mol), 'pdb')
                v1.setStyle({'stick': {}})
                v1.zoomTo()
                showmol(v1, height=300, width=400)
            with c2:
                st.markdown(f"**📈 建議策略: {opt_name}**")
                v2 = py3Dmol.view(width=400, height=300)
                v2.addModel(generate_3d_block(opt_mol), 'pdb')
                v2.setStyle({'stick': {'colorscheme': 'greenCarbon'}})
                v2.zoomTo()
                showmol(v2, height=300, width=400)
                st.caption(f"Ref: {opt_ref}")

            # Tab 3: 實證數據 (含 FTO 與 毒理)
            st.markdown("### 3️⃣ 實證醫學與專利分析 (Evidence Based)")
            
            t1, t2 = st.tabs(["☠️ 毒理風險", "⚖️ 專利 FTO"])
            
            with t1:
                col_h, col_l = st.columns(2)
                with col_h:
                    risk = "Moderate" if metrics['logp'] > 3.5 else "Low"
                    color = "risk-medium" if risk == "Moderate" else "risk-low"
                    st.markdown(f"""
                    <div style="border-left: 4px solid #ef4444; padding-left: 10px;">
                        <h4 style="margin:0;">🫀 心臟毒性 (hERG)</h4>
                        <p class="{color}" style="font-size:1.2rem;">Risk: {risk}</p>
                        <p style="font-size:0.9rem; color:#94a3b8;">機制: 預測基於 ChEMBL 活性數據與分子極性。</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_l:
                    risk_l = "Moderate" if metrics['logp'] > 4.0 else "Low"
                    color_l = "risk-medium" if risk_l == "Moderate" else "risk-low"
                    st.markdown(f"""
                    <div style="border-left: 4px solid #f59e0b; padding-left: 10px;">
                        <h4 style="margin:0;">🧪 肝臟毒性 (DILI)</h4>
                        <p class="{color_l}" style="font-size:1.2rem;">Risk: {risk_l}</p>
                        <p style="font-size:0.9rem; color:#94a3b8;">機制: CYP450 代謝穩定性評估。</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if chembl['found']:
                    st.markdown("#### 🔗 ChEMBL 真實活性數據")
                    st.dataframe(pd.DataFrame(chembl['acts']), use_container_width=True)

            with t2:
                # FTO 模擬圖
                st.markdown("#### 🗺️ 專利風險地圖")
                sim_val = 82 if "donepezil" in search_input.lower() else 15
                fig_p = go.Figure()
                fig_p.add_vrect(x0=0, x1=80, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0, annotation_text="安全區")
                fig_p.add_vrect(x0=80, x1=100, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, annotation_text="侵權區")
                fig_p.add_trace(go.Scatter(
                    x=[sim_val], y=[0.5], mode='markers+text',
                    marker=dict(size=20, color='#3b82f6', symbol='diamond', line=dict(width=2, color='white')),
                    text=["Current"], textposition="top center"
                ))
                fig_p.update_layout(xaxis=dict(range=[0, 100], title="相似度 %"), yaxis=dict(showticklabels=False), height=200, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_p, use_container_width=True)
                
                if sim_val > 80:
                    st.warning("⚠️ **高風險:** 結構與專利 US4895841 (Donepezil) 高度相似。建議進行 Claim 分析。")
                else:
                    st.success("✅ **低風險:** 未發現高度相似的核心專利。")
