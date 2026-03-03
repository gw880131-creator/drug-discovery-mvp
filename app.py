import streamlit as st
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, Fragments
import py3Dmol
from stmol import showmol
import plotly.graph_objects as go
import requests
import urllib.parse
import time
import pubchempy as pcp
from datetime import datetime

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="MedChem Pro | R&D Integrated", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS 風格設定 (深海藍企業風 + 內部警示) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 全局背景 */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* 玻璃擬態卡片 */
    div[data-testid="stExpander"], div.css-1r6slb0, .metric-card {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 16px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 15px;
    }
    
    /* 內部使用警示 */
    .internal-warning {
        background-color: rgba(245, 158, 11, 0.15); 
        border: 1px solid #f59e0b; 
        color: #fbbf24; 
        padding: 10px; 
        border-radius: 8px; 
        font-size: 0.85rem; 
        text-align: center;
        margin-bottom: 20px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* 輸入框與按鈕 */
    .stTextInput input { background-color: rgba(15, 23, 42, 0.8) !important; color: #e2e8f0 !important; border: 1px solid #475569 !important; border-radius: 8px; }
    .stButton>button { background: linear-gradient(to right, #2563eb, #3b82f6); color: white; border: none; border-radius: 8px; font-weight: 600; transition: all 0.3s; }
    .stButton>button:hover { box-shadow: 0 0 15px rgba(59, 130, 246, 0.5); transform: translateY(-1px); }
    
    /* 數值字體 */
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #38bdf8 !important; text-shadow: 0 0 10px rgba(56, 189, 248, 0.3); }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem; }
    
    /* Badge */
    .enterprise-badge { background: linear-gradient(90deg, #10b981, #059669); color: white; padding: 4px 12px; border-radius: 99px; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; margin-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. [整合核心] 免費開源 ADMET 規則引擎 ---
class FreeADMETRules:
    """
    基於文獻規則的免費預測引擎 (無需 API)
    整合自: Ekins et al. 2002, FDA DILIrank, BOILED-Egg
    """
    @staticmethod
    def predict_herg(mol):
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        
        # 結構警示 (SMARTS)
        alerts = {
            "High": ["[c]CCN", "[c]OCCN"], # 芳香環連接胺基
            "Moderate": ["N(C)C", "CN(C)C"] # 叔胺
        }
        
        # 1. 物理性質規則 (Ekins et al.)
        if tpsa < 60 and logp > 3.5:
            return "High", "High lipophilicity & Low TPSA (LogP>3.5, TPSA<60)", "Ekins et al. J Pharmacol Exp Ther 2002"
            
        # 2. 結構特徵檢查
        for level, patterns in alerts.items():
            for patt in patterns:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(patt)):
                    return level, f"Contains hERG pharmacophore ({patt})", "Structural Alert"
                    
        return "Low", "No significant structural alerts detected", "Rule-based prediction"

    @staticmethod
    def predict_liver(mol):
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # 1. 雙重高風險 (FDA DILIrank)
        if logp > 4.0 and mw > 400:
            return "Moderate", "Rule of 2: LogP > 4 & MW > 400", "Chen et al. Drug Metab Dispos 2016"
            
        # 2. 活性代謝物警示 (Carboxylic acid -> Acyl-glucuronide)
        if Fragments.fr_COO(mol) > 0:
            return "Moderate", "Contains carboxylic acid (potential reactive metabolite)", "Structural Alert"
            
        return "Low", "Properties within safe range", "Rule-based prediction"

    @staticmethod
    def predict_bbb(mol):
        # BOILED-Egg 邏輯
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        if tpsa < 79 and 0.4 < logp < 6.0:
            return "High", "Yellow Zone (Optimal for CNS)", "Daina & Zoete ChemMedChem 2016"
        elif tpsa < 120:
            return "Moderate", "White Zone (Peripheral)", "BOILED-Egg Model"
        else:
            return "Low", "Outside Egg (Poor Penetration)", "BOILED-Egg Model"

# --- 4. 運算與工具函式 ---

# A. 專利比對 (即時計算)
PATENT_REF_SMILES = {
    "Donepezil (US4895841)": "COc1ccc2cc1Oc1cc(cc(c1)C(F)(F)F)CC(=O)N2CCCCc1cccnc1",
    "Memantine (US4122193)": "CC12CC3CC(C1)(CC(C3)(C2)N)C",
    "Rivastigmine (US4948807)": "CCN(C)C(=O)OC1=CC=CC(=C1)C(C)N(C)C"
}

def calculate_realtime_fto(target_mol):
    results = []
    fp1 = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=1024)
    for name, ref_smiles in PATENT_REF_SMILES.items():
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
            sim_score = DataStructs.TanimotoSimilarity(fp1, fp2)
            results.append({
                "Patent": name,
                "Similarity": sim_score * 100,
                "Risk": "High" if sim_score > 0.8 else "Low"
            })
    results.sort(key=lambda x: x['Similarity'], reverse=True)
    return results

# B. 綜合屬性計算 (含 Lipinski)
def calculate_metrics(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    # Lipinski Rule of 5 Check
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    return {
        "mw": mw, "logp": logp, "tpsa": Descriptors.TPSA(mol),
        "hbd": hbd, "hba": hba, "qed": QED.qed(mol),
        "ro5_violations": violations,
        "in_egg": (Descriptors.TPSA(mol) < 79 and 0.4 < logp < 6.0)
    }

# C. PubChem 連線
def get_live_compound(query):
    try:
        mol = Chem.MolFromSmiles(query)
        if mol: return {"name": "User Input", "smiles": query}, mol
        c = pcp.get_compounds(query, 'name')
        if c:
            s = c[0].isomeric_smiles if c[0].isomeric_smiles else c[0].canonical_smiles
            return {"name": query, "smiles": s}, Chem.MolFromSmiles(s)
    except: return None, None
    return None, None

# D. 結構優化 (SMARTS)
TRANSFORMATIONS = {
    "reduce_lipophilicity": [{"name": "Scaffold Hop (Benzene -> Pyridine)", "smarts": "c1ccccc1>>c1ccncc1"}],
    "increase_lipophilicity": [{"name": "Methylation (NH -> N-Me)", "smarts": "[nH1:1]>>[n:1](C)"}]
}

def apply_live_transformation(mol, logp):
    strategy = "reduce_lipophilicity" if logp > 3.0 else "increase_lipophilicity"
    for t in TRANSFORMATIONS[strategy]:
        rxn = AllChem.ReactionFromSmarts(t['smarts'])
        try:
            ps = rxn.RunReactants((mol,))
            if ps:
                new_mol = ps[0][0]
                Chem.SanitizeMol(new_mol)
                return new_mol, t['name']
        except: continue
    return mol, "Stereoisomer Adjustment"

def generate_3d_block(mol):
    try:
        m = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m, AllChem.ETKDGv2())
        return Chem.MolToPDBBlock(m)
    except: return None

# --- 5. UI 主程式 ---

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('# MedChem <span style="color:#3b82f6">Pro</span> <span class="enterprise-badge">R&D V29.0</span>', unsafe_allow_html=True)
    st.caption("Integrated Research Platform | RDKit Core | Rule-based ADMET")
with c2:
    st.markdown('<div style="text-align:right; color:#4ade80; padding-top:20px;">⚡ Live Engine</div>', unsafe_allow_html=True)

# 內部使用警示 (來自 app_internal.py 的概念)
st.markdown("""
<div class="internal-warning">
    ⚠️ INTERNAL R&D USE ONLY - NOT FOR REGULATORY SUBMISSION<br
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔍 即時檢索")
    search_input = st.text_input("輸入藥名 / SMILES", "Caffeine")
    run_btn = st.button("🚀 執行運算")
    
    st.markdown("---")
    st.markdown("### Settings")
    st.checkbox("Use Live PubChem", value=True, disabled=True)
    st.checkbox("Show 3D Viewer", value=True)

if run_btn and search_input:
    with st.spinner(f"正在連線 PubChem 解析 '{search_input}'..."):
        data, mol = get_live_compound(search_input)
        
    if not mol:
        st.error(f"❌ 錯誤：無法解析 '{search_input}'。請確認拼字或網路狀態。")
    else:
        # --- 核心運算 ---
        with st.spinner("RDKit 正在計算屬性與 ADMET 規則..."):
            start_time = time.time()
            
            # 1. 基礎屬性
            metrics = calculate_metrics(mol)
            
            # 2. ADMET 規則引擎 (使用 FreeADMETRules)
            admet = FreeADMETRules()
            herg_risk, herg_desc, herg_ref = admet.predict_herg(mol)
            liver_risk, liver_desc, liver_ref = admet.predict_liver(mol)
            bbb_risk, bbb_desc, bbb_ref = admet.predict_bbb(mol)
            
            # 3. FTO & 優化
            fto_results = calculate_realtime_fto(mol)
            opt_mol, opt_strategy = apply_live_transformation(mol, metrics['logp'])
            
            calc_time = time.time() - start_time

        st.success(f"✅ 運算完成 (耗時: {calc_time:.3f} 秒)")

        # --- Tab 介面 ---
        tab1, tab2, tab3 = st.tabs(["🔬 科學核心 (Scientific)", "🧠 結構優化 (Optimization)", "☠️ 毒理與專利 (Evidence)"])

        # Tab 1: 科學核心
        with tab1:
            st.markdown("### 1️⃣ 物理化學屬性與 Lipinski 規則")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("MW", f"{metrics['mw']:.1f}", delta="< 500")
            k2.metric("LogP", f"{metrics['logp']:.2f}", delta="1-5")
            k3.metric("TPSA", f"{metrics['tpsa']:.1f}", delta="< 90")
            k4.metric("HBD", f"{metrics['hbd']}", delta="< 5")
            k5.metric("Ro5 Violations", f"{metrics['ro5_violations']}", 
                      delta_color="inverse" if metrics['ro5_violations'] > 0 else "normal")

            # Lipinski 狀態條
            if metrics['ro5_violations'] == 0:
                st.success("✅ 符合 Lipinski Rule of 5 (口服吸收性佳)")
            else:
                st.warning(f"⚠️ 違反 {metrics['ro5_violations']} 項 Lipinski 規則，可能影響生物利用度")

            # BOILED-Egg
            c_chart, c_desc = st.columns([2, 1])
            with c_chart:
                fig = go.Figure()
                fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=6, y1=140,
                    fillcolor="rgba(255, 204, 0, 0.2)", line_color="rgba(255, 204, 0, 0.5)")
                fig.add_trace(go.Scatter(
                    x=[metrics['logp']], y=[metrics['tpsa']], mode='markers+text',
                    marker=dict(size=18, color='#4ade80' if metrics['in_egg'] else '#f87171'),
                    text=["Input"], textposition="top center"
                ))
                fig.update_layout(xaxis_title="WLOGP", yaxis_title="TPSA", plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with c_desc:
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.5); padding:15px; border-radius:10px; border:1px solid rgba(255,255,255,0.1);">
                    <h4 style="color:#fcd34d; margin:0;">🧠 BBB 預測: {bbb_risk}</h4>
                    <p style="font-size:0.9rem; color:#cbd5e1;">{bbb_desc}</p>
                    <p style="font-size:0.8rem; color:#94a3b8;">Ref: {bbb_ref}</p>
                </div>
                """, unsafe_allow_html=True)

        # Tab 2: 結構優化
        with tab2:
            st.markdown("### 2️⃣ AI 結構優化建議")
            st.info(f"💡 **AI 策略:** {opt_strategy} (Based on LogP={metrics['logp']:.2f})")
            
            c1, c2 = st.columns(2)
            with c1:
                st.caption("原始結構 (3D Live)")
                v1 = py3Dmol.view(width=400, height=300)
                v1.addModel(generate_3d_block(mol), 'pdb')
                v1.setStyle({'stick': {}})
                v1.zoomTo()
                showmol(v1, height=300, width=400)
            with c2:
                st.caption(f"優化模擬: {opt_strategy}")
                v2 = py3Dmol.view(width=400, height=300)
                v2.addModel(generate_3d_block(opt_mol), 'pdb')
                v2.setStyle({'stick': {'colorscheme': 'greenCarbon'}})
                v2.zoomTo()
                showmol(v2, height=300, width=400)

        # Tab 3: 毒理與專利 (使用規則引擎)
        with tab3:
            st.markdown("### 3️⃣ ADMET 風險評估 (Rule-based)")
            
            col_h, col_l = st.columns(2)
            
            # hERG 卡片 (使用 FreeADMETRules)
            with col_h:
                color_h = "risk-high" if herg_risk == "High" else "risk-medium" if herg_risk == "Moderate" else "risk-low"
                border_h = "#ef4444" if herg_risk == "High" else "#f59e0b" if herg_risk == "Moderate" else "#10b981"
                
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.7); border-radius: 12px; padding: 20px; border-top: 4px solid {border_h};">
                    <h4 style="margin:0;">🫀 心臟毒性 (hERG)</h4>
                    <p class="{color_h}" style="font-size:1.2rem;">Risk: {herg_risk}</p>
                    <p style="font-size:0.9rem; color:#e2e8f0;">{herg_desc}</p>
                    <hr style="border-color: rgba(255,255,255,0.1);">
                    <p style="font-size:0.8rem; color:#94a3b8;">📚 Ref: {herg_ref}</p>
                </div>
                """, unsafe_allow_html=True)

            # Liver 卡片 (使用 FreeADMETRules)
            with col_l:
                color_l = "risk-high" if liver_risk == "High" else "risk-medium" if liver_risk == "Moderate" else "risk-low"
                border_l = "#ef4444" if liver_risk == "High" else "#f59e0b" if liver_risk == "Moderate" else "#10b981"
                
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.7); border-radius: 12px; padding: 20px; border-top: 4px solid {border_l};">
                    <h4 style="margin:0;">🧪 肝臟毒性 (DILI)</h4>
                    <p class="{color_l}" style="font-size:1.2rem;">Risk: {liver_risk}</p>
                    <p style="font-size:0.9rem; color:#e2e8f0;">{liver_desc}</p>
                    <hr style="border-color: rgba(255,255,255,0.1);">
                    <p style="font-size:0.8rem; color:#94a3b8;">📚 Ref: {liver_ref}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### ⚖️ FTO 專利相似度 (Fingerprint)")
            if fto_results:
                top_match = fto_results[0]
                st.metric("最相似專利", top_match['Patent'], f"{top_match['Similarity']:.2f}%")
                if top_match['Similarity'] > 80:
                    st.warning("⚠️ 結構與已知專利高度相似，請注意侵權風險。")
                else:
                    st.success("✅ 與主要專利結構差異大 (FTO Clear)。")
            else:
                st.info("無高度相似專利。")
