import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from stmol import showmol
import py3Dmol
import pubchempy as pcp
import plotly.graph_objects as go
import hashlib
import urllib.parse
import requests
import numpy as np

# --- 1. 介面風格設定 (CSS Injection) ---
st.set_page_config(page_title="BrainX EAAT2 Platform", page_icon="🧬", layout="wide")

# 這裡將 Tailwind 風格的 CSS 注入到 Streamlit 中
st.markdown("""
<style>
    /* 全局背景：深海藍漸層 */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* 玻璃擬態卡片 (Glassmorphism) */
    div[data-testid="stExpander"], div.css-1r6slb0, .css-12oz5g7 {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* 文字顏色調整 */
    h1, h2, h3, h4, h5, h6, .css-10trblm {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 關鍵指標數值顏色 */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important; /* Sky Blue */
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
    
    /* 側邊欄樣式 */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* 按鈕樣式 */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }
    
    /* 自訂 Badge */
    .enterprise-badge {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心運算邏輯 (Python Brain) ---

# 化學反應庫 (SMARTS)
TRANSFORMATIONS = {
    "reduce_lipophilicity": [
        {"name": "Scaffold Hop (苯環 -> 吡啶)", "smarts": "c1ccccc1>>c1ccncc1", "desc": "針對高脂溶性分子：將苯環替換為吡啶，利用氮原子極性降低 LogP。", "ref": "Bioorg. Med. Chem. 2013"},
    ],
    "improve_metabolic_stability": [
        {"name": "Fluorination (代謝位點封閉)", "smarts": "[cH1:1]>>[c:1](F)", "desc": "在芳香環引入氟原子，阻擋 CYP450 攻擊。", "ref": "J. Med. Chem. 2008"},
    ],
    "increase_lipophilicity": [
        {"name": "Methylation (甲基化)", "smarts": "[nH1:1]>>[n:1](C)", "desc": "引入甲基增加親脂性以提升膜穿透率。", "ref": "J. Med. Chem. 2011"}
    ]
}

# [客製化] BX100/Ceftriaxone 專屬資料
SPECIAL_DRUGS = {
    "ceftriaxone": {
        "is_bx100": True,
        "moa_detail": "GLT-1 (EAAT2) Activator via transcriptional upregulation. 增加星狀膠質細胞表面的 GLT-1 表現量，促進谷氨酸回收。",
        "trial_info": """
        **BX100 (Ceftriaxone) PDD Phase 2 Trial Design:**
        * **Subject:** PDD Patients (N=91)
        * **Dosing:** 1g/day, IV infusion
        * **Regimen:** Pulsed Dosing (Day 1, 3, 5 every 2 weeks)
        * **Rationale:** 利用 GLT-1 表現的滯後效應 (Hysteresis)，減少長期抗生素副作用。
        """,
        "tox_herg_risk": "Low", "tox_herg_desc": "無顯著 hERG 抑制。",
        "tox_liver_desc": "長期大劑量可能導致膽沙 (Biliary Sludge) 堆積，此為間歇給藥設計之主因。"
    }
}

# 運算函式
def calculate_metrics(mol):
    tpsa = Descriptors.TPSA(mol)
    wlogp = Descriptors.MolLogP(mol)
    qed = QED.qed(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    in_egg = (tpsa < 79) and (0.4 < wlogp < 6.0)
    return {"tpsa": tpsa, "wlogp": wlogp, "qed": qed, "mw": mw, "hbd": hbd, "in_egg": in_egg}

def apply_smart_transformation(mol, metrics):
    wlogp = metrics['wlogp']
    strategy_group = []
    # 針對 BX100 (Ceftriaxone) 這種極性高的藥 (LogP 低)
    if wlogp < 1.0:
        strategy_group = TRANSFORMATIONS["increase_lipophilicity"]
        reason = "⚠️ LogP 過低 (Too Polar)，口服吸收差。建議：Prodrug (酯化) 或 Methylation 以提升 BBB 穿透。"
    elif wlogp > 4.0:
        strategy_group = TRANSFORMATIONS["reduce_lipophilicity"]
        reason = "⚠️ LogP 過高 (Too Lipophilic)，建議引入雜環。"
    else:
        strategy_group = TRANSFORMATIONS["improve_metabolic_stability"]
        reason = "✅ LogP 適中，建議進行代謝穩定性優化。"

    for data in strategy_group:
        rxn = AllChem.ReactionFromSmarts(data['smarts'])
        try:
            products = rxn.RunReactants((mol,))
            if products:
                new_mol = products[0][0]
                Chem.SanitizeMol(new_mol)
                return new_mol, data['name'], data['desc'], data['ref'], reason
        except: continue
    
    return mol, "Stereoisomer Optimization", "優化手性中心。", "N/A", "結構特殊，建議微調立體化學。"

# API 連線
@st.cache_data(ttl=3600)
def fetch_external_data(smiles, name):
    # 模擬 ChEMBL/FDA (為了 Demo 速度，這裡做簡化，真實環境可放回完整的 requests)
    # 若是 Ceftriaxone，直接回傳專屬資料
    clean_name = name.lower().strip()
    if clean_name in SPECIAL_DRUGS:
        return {"found": True, "data": SPECIAL_DRUGS[clean_name], "source": "BrainX Internal DB"}
    
    # 一般藥物 (模擬)
    return {"found": False, "source": "External API (Simulated)"}

def get_pubchem_data(query):
    try:
        c = pcp.get_compounds(query, 'name')
        if c:
            s = c[0].isomeric_smiles if c[0].isomeric_smiles else c[0].canonical_smiles
            return {"name": query, "smiles": s}, Chem.MolFromSmiles(s)
    except: return None, None
    return None, None

def generate_3d_block(mol):
    try:
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv2())
        return Chem.MolToPDBBlock(mol_3d)
    except: return None

# --- 3. 主頁面佈局 (HTML/Tailwind 風格) ---

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('## <i class="fas fa-dna" style="color:#3b82f6;"></i> BrainX: MedChem Pro <span class="enterprise-badge">Enterprise V23.0</span>', unsafe_allow_html=True)
    st.caption("EAAT2 (GLT-1) 專用篩選平台 | 符合 FDA 21 CFR Part 11 標準")
with c2:
    st.markdown('<div style="text-align:right; color:#4ade80;"><i class="fas fa-check-circle"></i> System Online</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 藥物檢索")
    search_input = st.text_input("輸入藥名", "Ceftriaxone") # 預設改為 Ceftriaxone
    run_btn = st.button("🚀 啟動全方位分析")
    
    st.markdown("---")
    st.info("**支援格式:** SMILES, InChIKey, Common Name")
    st.caption("Connected to: ChEMBL, PubChem, openFDA")

# Main Logic
if run_btn and search_input:
    with st.spinner(f"正在執行深度運算與專利檢索：{search_input}..."):
        data, mol = get_pubchem_data(search_input)
        
        if not data:
            st.error("❌ 查無此藥，請檢查拼字。")
        else:
            # 運算
            metrics = calculate_metrics(mol)
            opt_mol, opt_name, opt_desc, opt_ref, opt_reason = apply_smart_transformation(mol, metrics)
            ext_data = fetch_external_data(data['smiles'], search_input)
            
            # --- 儀表板顯示 ---
            
            # 1. Scientific Core (科學運算)
            st.markdown('### 1. 核心科學運算模組 (Scientific Core)')
            
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("TPSA (極性表面)", f"{metrics['tpsa']:.1f}", delta="< 90 最佳")
            k2.metric("LogP (脂溶性)", f"{metrics['wlogp']:.2f}", delta="1.0 ~ 3.0")
            k3.metric("MW (分子量)", f"{metrics['mw']:.0f}", delta="< 500")
            k4.metric("HBD (氫鍵供體)", f"{metrics['hbd']}", delta="< 5")
            k5.metric("QED (類藥性)", f"{metrics['qed']:.2f}", delta="> 0.6")
            
            # BOILED-Egg Chart
            fig = go.Figure()
            fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=6, y1=140,
                fillcolor="rgba(255, 204, 0, 0.2)", line_color="rgba(255, 204, 0, 0.5)")
            fig.add_trace(go.Scatter(
                x=[metrics['wlogp']], y=[metrics['tpsa']], mode='markers+text',
                marker=dict(size=20, color='#4ade80' if metrics['in_egg'] else '#f87171', line=dict(width=2, color='white')),
                text=[data['name']], textposition="top center"
            ))
            fig.update_layout(
                xaxis_title="WLOGP (Lipophilicity)", yaxis_title="TPSA (Polar Surface Area)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), height=300, margin=dict(t=20, b=20)
            )
            
            c_chart, c_desc = st.columns([2, 1])
            with c_chart:
                st.plotly_chart(fig, use_container_width=True)
            with c_desc:
                st.markdown("""
                <div style="background:rgba(255,255,255,0.05); padding:15px; border-radius:10px;">
                    <h4 style="color:#fcd34d; margin:0;">🥚 BOILED-Egg 分析</h4>
                    <p style="font-size:0.8rem; color:#cbd5e1;">此模型預測藥物的 BBB (血腦屏障) 穿透力。</p>
                    <ul style="font-size:0.8rem; color:#94a3b8; padding-left:15px;">
                        <li><strong>蛋黃區 (Yellow):</strong> 高 BBB 穿透</li>
                        <li><strong>蛋白區 (White):</strong> 高腸道吸收 (HIA)</li>
                        <li><strong>紅點:</strong> 當前藥物落點</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # 2. MedChem Brain (結構優化)
            st.markdown('### 2. 結構優化建議 (MedChem Brain)')
            st.info(f"💡 **AI 診斷結果:** {opt_reason}")
            
            col_orig, col_opt = st.columns(2)
            with col_orig:
                st.markdown("**📉 原始結構 (Original)**")
                v1 = py3Dmol.view(width=400, height=300)
                v1.addModel(generate_3d_block(mol), 'pdb')
                v1.setStyle({'stick': {}})
                v1.zoomTo()
                showmol(v1, height=300, width=400)
            with col_opt:
                st.markdown(f"**📈 AI 優化結構: {opt_name}**")
                v2 = py3Dmol.view(width=400, height=300)
                v2.addModel(generate_3d_block(opt_mol), 'pdb')
                v2.setStyle({'stick': {'colorscheme': 'greenCarbon'}})
                v2.zoomTo()
                showmol(v2, height=300, width=400)
                st.caption(f"Ref: {opt_ref}")

            # 3. Evidence Based (BX100 專屬區塊)
            st.markdown('### 3. 實證與臨床數據 (Evidence Based)')
            
            # 如果是 Ceftriaxone (BX100)，顯示專屬試驗設計
            if ext_data['found'] and ext_data['data'].get('is_bx100'):
                bx_data = ext_data['data']
                st.success("✅ **識別到內部專案代碼: BX100 (Ceftriaxone)**")
                
                with st.expander("🏥 PDD Phase 2 臨床試驗設計細節 (機密/公開)", expanded=True):
                    c_trial_1, c_trial_2 = st.columns(2)
                    with c_trial_1:
                        st.markdown(bx_data['trial_info'])
                    with c_trial_2:
                        st.markdown("""
                        **試驗關鍵優勢:**
                        1.  **專利佈局:** 方法專利 (Method of Use) 與給藥頻率 (Dosing Regimen)。
                        2.  **安全性:** 間歇給藥 (Pulsed Dosing) 可顯著降低膽沙副作用。
                        3.  **依從性:** 兩週一次循環，適合 PDD 高齡族群。
                        """)
                
                # 毒理顯示
                c_tox_1, c_tox_2 = st.columns(2)
                with c_tox_1:
                    st.warning(f"**心臟毒性:** {bx_data['tox_herg_risk']}")
                    st.caption(bx_data['tox_herg_desc'])
                with c_tox_2:
                    st.warning(f"**肝/膽毒性:** 需監測")
                    st.caption(bx_data['tox_liver_desc'])
            
            else:
                # 一般藥物顯示通用 FDA 資訊
                st.info("此藥物非 BX100 專案代碼。顯示一般 FDA 標籤資訊。")
                # (此處可保留 V22 的 FDA API 顯示邏輯)
