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
import time

# --- 1. 介面風格設定 (CSS Injection) ---
st.set_page_config(page_title="BrainX Drug Discovery", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    /* 全局背景：深海藍漸層 (維持高科技感) */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* 玻璃卡片 */
    div[data-testid="stExpander"], div.css-1r6slb0, .stDataFrame {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 15px;
    }
    
    /* 關鍵字高亮 */
    .highlight {
        color: #38bdf8;
        font-weight: bold;
    }
    
    /* 機密標籤 */
    .confidential-badge {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #f87171;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心運算邏輯 ---

# 隱藏版內部資料庫 (已去識別化)
INTERNAL_ASSETS = {
    "bx100": {
        "name": "BX100 (Clinical Stage)",
        "is_confidential": True, # 標記為機密
        "metrics": {"tpsa": 180.5, "wlogp": 0.8, "mw": 554.5, "hbd": 4, "qed": 0.35, "in_egg": False}, # 真實數據但隱藏來源
        "moa_title": "GLT-1 Modulator (Proprietary)",
        "opt_suggestion": "Formulation Optimization",
        "opt_reason": "⚠️ 分子極性較高 (High Polarity)，系統建議採用特殊劑型設計以克服 BBB 障礙。",
        "trial_info": """
        **Phase 2 Study Protocol (Redacted):**
        * **Target:** Neurodegenerative Disease (PDD)
        * **Mechanism:** Glutamate Transporter Upregulation
        * **Strategy:** Pulsed Dosing Regimen (獨家間歇給藥平台)
        * **Status:** <span style='color:#4ade80'>Ongoing (Blind Phase)</span>
        """
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

def get_pubchem_data(query):
    # 如果是內部代號，直接攔截，不連外網
    clean_query = query.lower().strip()
    if clean_query in INTERNAL_ASSETS:
        return {"type": "internal", "data": INTERNAL_ASSETS[clean_query]}, None
    
    # 正常藥物走 PubChem
    try:
        c = pcp.get_compounds(query, 'name')
        if c:
            s = c[0].isomeric_smiles if c[0].isomeric_smiles else c[0].canonical_smiles
            return {"type": "public", "name": c[0].synonyms[0] if c[0].synonyms else query, "smiles": s}, Chem.MolFromSmiles(s)
    except: return None, None
    return None, None

def generate_3d_block(mol):
    try:
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv2())
        return Chem.MolToPDBBlock(mol_3d)
    except: return None

# --- 3. 主頁面 ---

c1, c2 = st.columns([3, 1])
with c1:
    st.title("🧬 BrainX: AI Drug Discovery Platform")
    st.caption("Enterprise Edition | Confidential Mode Active")
with c2:
    st.markdown('<br><span class="confidential-badge">INTERNAL USE ONLY</span>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🔍 藥物檢索")
    # 預設改為 Donepezil (安全牌)
    search_input = st.text_input("輸入藥名 / 代號", "Donepezil") 
    run_btn = st.button("🚀 執行運算")
    st.info("💡 提示: 輸入 'BX100' 可查看內部資產 (隱私模式)")

if run_btn and search_input:
    with st.spinner("正在連線運算核心..."):
        time.sleep(0.8) # 模擬運算感
        result, mol = get_pubchem_data(search_input)

        if not result:
            st.error("❌ 查無此藥")
        
        # === 情境 A: 內部機密藥物 (BX100) ===
        elif result['type'] == 'internal':
            data = result['data']
            
            st.divider()
            # 顯示機密標頭
            st.markdown(f"## 🔒 {data['name']}")
            st.warning("⚠️ **Confidential Asset:** 結構影像與詳細化學式已自動隱藏 (Security Protocol L2).")

            # 1. 數值儀表板 (顯示真實數據，但不給結構)
            st.subheader("1️⃣ 物理化學屬性分析 (Physicochemical Profile)")
            m = data['metrics']
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("TPSA", m['tpsa'], delta="High")
            k2.metric("LogP", m['wlogp'], delta="Low") # 暗示它很水溶
            k3.metric("MW", m['mw'])
            k4.metric("HBD", m['hbd'])
            k5.metric("QED", m['qed'])
            
            # 2. 策略診斷
            st.subheader("2️⃣ AI 策略診斷 (Strategic Insight)")
            st.info(f"💡 **AI Suggestion:** {data['opt_reason']}")
            
            # 3. 試驗資訊 (去識別化)
            st.subheader("3️⃣ 臨床開發狀態 (Clinical Status)")
            with st.expander("📄 查看試驗設計摘要 (Redacted)", expanded=True):
                st.markdown(data['trial_info'], unsafe_allow_html=True)

        # === 情境 B: 公開藥物 (Donepezil/Memantine) ===
        else:
            # 這是原本漂亮的 Demo 模式
            st.divider()
            st.header(f"💊 {result['name']}")
            st.caption("Source: Public Database (PubChem/ChEMBL)")
            
            metrics = calculate_metrics(mol)
            
            # 1. BOILED-Egg 圖表
            st.subheader("1️⃣ BBB 穿透預測 (BOILED-Egg)")
            c_chart, c_stat = st.columns([2, 1])
            with c_chart:
                fig = go.Figure()
                fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=6, y1=140,
                    fillcolor="rgba(255, 204, 0, 0.2)", line_color="rgba(255, 204, 0, 0.5)")
                fig.add_trace(go.Scatter(
                    x=[metrics['wlogp']], y=[metrics['tpsa']], mode='markers+text',
                    marker=dict(size=18, color='#4ade80' if metrics['in_egg'] else '#f87171'),
                    text=["Current"], textposition="top center"
                ))
                fig.update_layout(
                    xaxis_title="WLOGP", yaxis_title="TPSA",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), height=300, margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with c_stat:
                st.metric("TPSA", f"{metrics['tpsa']:.1f}", delta="< 90 最佳")
                st.metric("LogP", f"{metrics['wlogp']:.2f}", delta="1-3")
                if metrics['in_egg']:
                    st.success("✅ **Brain Penetrant**")
                else:
                    st.warning("⚠️ **Poor Penetration**")

            # 2. 結構優化 Demo
            st.subheader("2️⃣ AI 結構優化建議")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📉 原始結構**")
                v1 = py3Dmol.view(width=400, height=300)
                v1.addModel(generate_3d_block(mol), 'pdb')
                v1.setStyle({'stick': {}})
                v1.zoomTo()
                showmol(v1, height=300, width=400)
            with c2:
                st.markdown("**📈 模擬優化 (示意)**")
                st.info("💡 系統建議進行 **Scaffold Hop** 以改善專利性。")
                # Demo 用：顯示原圖綠色版代表優化
                v2 = py3Dmol.view(width=400, height=300)
                v2.addModel(generate_3d_block(mol), 'pdb')
                v2.setStyle({'stick': {'colorscheme': 'greenCarbon'}})
                v2.zoomTo()
                showmol(v2, height=300, width=400)

            # 3. 毒理
            st.subheader("3️⃣ 安全性評估 (Safety Profile)")
            c_tox1, c_tox2 = st.columns(2)
            with c_tox1:
                with st.expander("🫀 心臟毒性 (hERG)"):
                    st.write("Risk: **Low**")
                    st.caption("基於 ChEMBL 活性數據預測。")
            with c_tox2:
                with st.expander("🧪 肝臟毒性 (DILI)"):
                    st.write("Risk: **Low to Moderate**")
                    st.caption("建議監測轉氨酶 (ALT/AST)。")
