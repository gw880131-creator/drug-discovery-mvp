import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import hashlib
from datetime import datetime
import requests
import pubchempy as pcp
# --- 關鍵的這一行必須存在 ---
from chembl_webresource_client.new_client import new_client 
# ------------------------
import plotly.express as px
import plotly.graph_objects as go
import py3Dmol
from stmol import showmol

# 匯入 RDKit 相關模組
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs
from rdkit.Chem import Fragments
import urllib.parse
import time

# ==================== 頁面設定與淺色 CSS ====================
st.set_page_config(
    page_title="MedChem Pro | Enterprise R&D Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 1. 全局背景：乾淨的淺灰白漸層 */
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); color: #1e293b; font-family: 'Inter', sans-serif; }
    
    /* 2. 側邊欄：淡雅的灰白色 */
    section[data-testid="stSidebar"] { background-color: #f1f5f9; border-right: 1px solid #cbd5e1; }
    
    /* 3. 玻璃擬態卡片：半透明白色 */
    div[data-testid="stExpander"], div.css-1r6slb0, .metric-card {
        background: rgba(255, 255, 255, 0.8) !important; backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 1); border-radius: 16px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); padding: 20px; margin-bottom: 15px;
    }
    
    /* 4. 輸入框與按鈕 */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div { 
        background-color: #ffffff !important; color: #1e293b !important; border: 1px solid #cbd5e1 !important; border-radius: 8px; 
    }
    .stButton>button { background: linear-gradient(to right, #2563eb, #3b82f6); color: white; border: none; border-radius: 8px; font-weight: 600; }
    
    /* =================【淺色主題：數字與標題】================= */
    /* 指標數值：深藍色，放大 */
    div[data-testid="stMetricValue"] { 
        font-family: 'JetBrains Mono', monospace; color: #1e40af !important; font-size: 2.5rem !important; text-shadow: none !important;
    }
    
    /* 強制將指標標題 (MW, LogP 等) 改為清晰的深灰色 */
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { 
        color: #475569 !important; font-size: 1.1rem !important; font-weight: 800 !important; letter-spacing: 1px !important; text-shadow: none !important;
    }
    /* ========================================================= */

    /* 5. 標題與一般文字顏色 */
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
    p, li { color: #334155; }

    /* 6. 內部警示與風險標籤 */
    .internal-warning {
        background-color: #fef3c7; border: 1px solid #f59e0b; color: #b45309; padding: 10px; border-radius: 8px; font-size: 0.85rem; text-align: center; margin-bottom: 20px; font-weight: 600; letter-spacing: 0.5px;
    }
    .risk-high { color: #ef4444; font-weight: bold; }
    .risk-medium { color: #f59e0b; font-weight: bold; }
    .risk-low { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== 救命字典 (Demo防斷網) ====================
SAFE_DEMO_DB = {
    "donepezil": "COc1ccc2cc1Oc1cc(cc(c1)C(F)(F)F)CC(=O)N2CCCCc1cccnc1",
    "memantine": "CC12CC3CC(C1)(CC(C3)(C2)N)C",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "amoxicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(cc3)O)N)C(=O)O)C",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
}

# ==================== 公開資料庫與靶點預測 API ====================
class PublicDatabaseAPI:
    def __init__(self):
        # 鎖定研發專案核心靶點 (AD/PD 相關通用關鍵字)
        self.ad_pd_keys = ["EAAT2", "GLT-1", "BACE1", "AMYLOID", "TAU", "MAO-B", "GSK-3", "ACHE"]

    def query_pubchem(self, identifier, id_type="name"):
        """【真·即時查詢】獲取最新結構數據並計算基礎描述符"""
        import urllib.parse
        clean_query = identifier.strip()
        try:
            c = pcp.get_compounds(clean_query, id_type)
            if not c:
                c = pcp.get_compounds(clean_query, 'searchtype=synonym')
            if not c: return None
            comp = c[0]
            smiles = comp.isomeric_smiles or comp.canonical_smiles
            mol = Chem.MolFromSmiles(smiles)
            return {
                'cid': comp.cid, 'name': clean_query.title(), 'smiles': smiles,
                'mw': Descriptors.MolWt(mol), 'logp': Descriptors.MolLogP(mol), 'tpsa': Descriptors.TPSA(mol)
            }
        except Exception: return None

    def get_modification_suggestions(self, result):
        """【藥化決策引擎】去標籤化版本：實時預測 P-gp 風險與結構整形建議"""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(result['smiles'])
        if not mol: return ["❌ 分子結構解析失敗"]
        
        logp, tpsa = result['logp'], result['tpsa']
        suggestions = []
        
        # 1. 專業藥化 SMARTS 偵測 (隱藏特定開發代號)
        pgp_pattern = Chem.MolFromSmarts('[NX3](C)(C)C')  # 偵測三級胺
        acid_pattern = Chem.MolFromSmarts('C(=O)[O;H1,H0-]') # 偵測羧酸/羧酸根
        ester_pattern = Chem.MolFromSmarts('C(=O)O[C,H]') # 偵測酯鍵

        # 2. 實時 CNS MPO 運算
        mpo_score = 0
        if 1.0 < logp < 3.0: mpo_score += 1.0
        if tpsa < 75: mpo_score += 1.0
        if Descriptors.NumHDonors(mol) <= 1: mpo_score += 1.0

        suggestions.append(f"📊 **實時 CNS MPO 指數: {mpo_score}/3.0**")

        # 3. 基於化學特徵的專業建議
        if mol.HasSubstructMatch(acid_pattern):
            suggestions.append("🚫 **結構性質提示**: 偵測到酸性基團（羧酸根）。這類基團在生理 pH 下易解離帶電，是阻礙穿透血腦屏障 (BBB) 的主因。")
            suggestions.append("🔹 **策略**: 建議考慮酯化前藥設計，或置換為中性生物電子等排體以優化入腦能力。")
            

        if mol.HasSubstructMatch(pgp_pattern) and logp > 3.5:
            suggestions.append("⚠️ **轉運體風險**: 偵測到潛在的 P-gp 底物特徵，分子可能面臨外排幫浦干擾，降低腦內有效濃度。")
            

        if mol.HasSubstructMatch(ester_pattern):
            suggestions.append("🔬 **代謝穩定性**: 含有酯鍵結構。請注意在血漿中可能存在的快速水解風險。")

        if mpo_score >= 2.5 and not mol.HasSubstructMatch(acid_pattern):
            suggestions.append("✅ **結構潛力評估**: 該分子具備良好的 CNS 類藥性空間分布，適合進行細胞活性篩選。")

        return suggestions

    def get_clinical_summary(self, drug_name):
        """【即時檢索】Wikipedia + ChEMBL 雙重檢索機制"""
        import urllib.parse
        search_name = drug_name.strip()
        try:
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_name.capitalize())}"
            res = requests.get(wiki_url, timeout=10)
            if res.status_code == 200:
                return res.json().get('extract', "找到紀錄但無摘要。")

            mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?pref_name__icontains={search_name}&format=json"
            mol_res = requests.get(mol_url, timeout=10)
            if mol_res.status_code == 200:
                mols = mol_res.json().get('molecules', [])
                if mols:
                    chembl_id = mols[0].get('molecule_chembl_id')
                    mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism?molecule_chembl_id={chembl_id}&format=json"
                    mech_res = requests.get(mech_url, timeout=10)
                    if mech_res.status_code == 200:
                        mechanisms = mech_res.json().get('mechanisms', [])
                        if mechanisms:
                            return "\n".join([f"🎯 **實時機制**: {m.get('mechanism_of_action')} (靶點: {m.get('target_name')})" for m in mechanisms])
            return f"❌ 實時檢索完成：'{search_name}' 無公開臨床紀錄。"
        except: return "資料庫服務暫時無法連線。"

    def get_pubmed_details(self, drug_name):
        """抓取最新的 EAAT2 與神經保護相關科研證據"""
        import urllib.parse
        search_term = f"({drug_name}) AND (EAAT2 OR GLT-1 OR neuroprotection)"
        try:
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(search_term)}&retmode=json&retmax=3"
            res = requests.get(search_url, timeout=10)
            id_list = res.json().get('esearchresult', {}).get('idlist', [])
            if not id_list: return []
            ids = ",".join(id_list)
            sum_res = requests.get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids}&retmode=json", timeout=10)
            result_set = sum_res.json().get('result', {})
            return [{"title": result_set.get(pmid, {}).get('title', '無標題'), "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"} for pmid in id_list]
        except: return []

    def predict_targets(self, smiles, drug_name):
        """【人體數據特化版】結合相似度搜尋，且僅鎖定 Homo sapiens 靶點數據"""
        import urllib.parse
        try:
            base_url = "https://www.ebi.ac.uk/chembl/api/data"
            safe_smiles = urllib.parse.quote(smiles)
            
            # 1. 執行相似度檢索 (閾值 70% 確保有結果)
            res = requests.get(f"{base_url}/similarity/{safe_smiles}/70?format=json", timeout=20)
            
            targets_map = {}
            mols = []
            if res.status_code == 200:
                mols = res.json().get('molecules', [])
            
            # 2. 備案：若相似度無果則執行子結構搜尋
            if not mols:
                sub_res = requests.get(f"{base_url}/substructure/{safe_smiles}?limit=5&format=json", timeout=20)
                if sub_res.status_code == 200:
                    mols = sub_res.json().get('molecules', [])

            if mols:
                for m in mols[:5]:
                    chembl_id = m['molecule_chembl_id']
                    # 獲取活性數據，並加入種屬過濾
                    act_res = requests.get(f"{base_url}/activity?molecule_chembl_id={chembl_id}&limit=50&format=json", timeout=15)
                    if act_res.status_code == 200:
                        for act in act_res.json().get('activities', []):
                            # --- 關鍵過濾：僅保留人類數據 ---
                            organism = act.get('target_organism', '')
                            if organism == "Homo sapiens":
                                t_name = act.get('target_pref_name')
                                if t_name and any(key in t_name.upper() for key in self.ad_pd_keys):
                                    # 使用 pChEMBL 值（如 pIC50）作為權重，無則給予基礎分
                                    score = float(act.get('pchembl_value') or 5.0)
                                    targets_map[t_name] = targets_map.get(t_name, 0) + score
            
            if targets_map:
                sorted_res = sorted(targets_map.items(), key=lambda x: x[1], reverse=True)[:8]
                max_v = max([x[1] for x in sorted_res])
                return [{"Target": t[0], "Score": round((t[1]/max_v)*99.5, 1)} for t in sorted_res]
        except: pass
        return []
# ==================== ADMET 規則引擎 ====================
class FreeADMETRules:
    @staticmethod
    def predict_herg(mol):
        """預測 hERG 心臟毒性風險"""
        tpsa, logp = Descriptors.TPSA(mol), Descriptors.MolLogP(mol)
        alerts = {"High": ["[c]CCN", "[c]OCCN"], "Moderate": ["N(C)C", "CN(C)C"]}
        if tpsa < 60 and logp > 3.5: 
            return "High", "High lipophilicity & Low TPSA", "Ekins et al. 2002"
        for level, patterns in alerts.items():
            for patt in patterns:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(patt)): 
                    return level, f"Contains hERG pharmacophore", "Structural Alert"
        return "Low", "No significant alerts", "Rule-based"

    @staticmethod
    def predict_liver(mol):
        """預測肝毒性 (DILI) 風險"""
        if Descriptors.MolLogP(mol) > 4.0 and Descriptors.MolWt(mol) > 400: 
            return "Moderate", "Rule of 2: LogP > 4 & MW > 400", "Chen et al. 2016"
        if Fragments.fr_COO(mol) > 0: 
            return "Moderate", "Contains carboxylic acid", "Structural Alert"
        return "Low", "Properties within safe range", "Rule-based"

    @staticmethod
    def predict_bbb(mol):
        """預測血腦屏障 (BBB) 通透性"""
        logp, tpsa = Descriptors.MolLogP(mol), Descriptors.TPSA(mol)
        if tpsa < 79 and 0.4 < logp < 6.0: 
            return "High", "Yellow Zone (Optimal for CNS)", "BOILED-Egg Model"
        elif tpsa < 120: 
            return "Moderate", "White Zone (Peripheral)", "BOILED-Egg Model"
        else: 
            return "Low", "Outside Egg (Poor Penetration)", "BOILED-Egg Model"
# ==================== 3D 結構生成輔助函式 ====================
def generate_3d_pdb(mol):
    """將 RDKit 分子物件轉換為 3D PDB 格式"""
    try:
        # 1. 為分子添加氫原子 (Add Hydrogens)
        mol_3d = Chem.AddHs(mol)
        
        # 2. 執行 3D 構象嵌入 (3D Embedding)
        # 使用 ETKDGv2 演算法生成立體座標
        params = AllChem.ETKDGv2()
        AllChem.EmbedMolecule(mol_3d, params)
        
        # 3. 結構優化 (力場優化，可選但建議加上以獲得更合理的鍵長鍵角)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        
        # 4. 轉換為 PDB 文字塊以供 py3Dmol 讀取
        return Chem.MolToPDBBlock(mol_3d)
    except Exception as e:
        # 若嵌入失敗 (例如分子太大或太碎片化)，回傳 None
        print(f"3D 嵌入失敗: {e}")
        return None
# ==================== 主程式 ====================
def main():
    public_api = PublicDatabaseAPI()
    admet = FreeADMETRules()
    
    st.markdown('<div class="internal-warning">⚠️ INTERNAL R&D USE ONLY - NOT FOR REGULATORY SUBMISSION</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🧬 Navigation")
        page = st.radio("Select Module", ["🌐 Drug Query & Target Prediction", "🏠 Internal Dashboard", "📝 Database Settings"])

    if page == "🌐 Drug Query & Target Prediction":
        st.header("Drug Query & AI Target Prediction")
        st.caption("輸入藥物，系統將即時解析結構、預測靶點 (Target) 並評估 ADMET 風險。")
        
        query = st.text_input("Enter Drug Name or SMILES", "Donepezil")
        
        # 預設 result 為 None，防止 UnboundLocalError
        result = None 

        if st.button("🚀 Analyze & Predict", use_container_width=True):
            with st.spinner("Running AI Models and Target Prediction..."):
                # 執行即時搜尋
                result = public_api.query_pubchem(query, "name" if "1" not in query and "C" not in query else "smiles")
                
                if not result:
                    st.error("❌ 無法解析分子結構，請檢查輸入。")

        # =========================================================
        # 核心修正點：只有當 result 成功獲取後，才執行下方的 UI 渲染
        # =========================================================
        if result:
            mol = Chem.MolFromSmiles(result['smiles'])
            
            # --- 區塊 1: 物理化學儀表板 ---
            st.markdown("### 1️⃣ Physicochemical Profile")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("MW", f"{result['mw']:.1f}")
            k2.metric("LogP", f"{result['logp']:.2f}")
            k3.metric("TPSA", f"{result['tpsa']:.1f}")
            k4.metric("HBD", f"{Descriptors.NumHDonors(mol)}")
            k5.metric("QED", f"{QED.qed(mol):.2f}")

            # --- 區塊 1.2: 臨床背景 ---
            st.markdown("### 📚 Clinical Background & Mechanism")
            clinical_info = public_api.get_clinical_summary(query)
            st.write(clinical_info)

            # --- 區塊 1.3: PubMed 文獻追蹤 ---
            st.markdown("### 🔬 Related Scientific Publications (PubMed)")
            pubmed_results = public_api.get_pubmed_details(query)
            if pubmed_results:
                for paper in pubmed_results:
                    st.markdown(f"📄 **{paper['title']}**")
                    st.markdown(f"🔗 [查看原文]({paper['link']})")
                    st.divider()
            else:
                st.info("目前 PubMed 暫無直接關聯之研究文獻。")

            # --- 區塊 1.5: 藥化修飾建議 ---
            st.markdown("### 🛠️ AI Chemical Modification Suggestions")
            mod_suggestions = public_api.get_modification_suggestions(result)
            for advice in mod_suggestions:
                st.info(advice)

            # --- 區塊 2: AI 靶點預測 (僅限 Homo sapiens) ---
            st.markdown("### 🎯 2️⃣ AI Target Prediction & PubMed Evidence")
            col_chart, col_papers = st.columns([3, 2])
            
            with col_chart:
                # 執行我們寫入的「人體數據」過濾與「子結構備案」邏輯
                targets_data = public_api.predict_targets(result['smiles'], result['name'])
                if targets_data:
                    df_targets = pd.DataFrame(targets_data)
                    fig_t = px.bar(df_targets, x="Score", y="Target", orientation='h', 
                                   color="Score", color_continuous_scale="Blues")
                    fig_t.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_t, use_container_width=True)
                else:
                    st.warning("⚠️ 目前人體資料庫無相似活性紀錄。")
            
            with col_papers:
                st.markdown("#### 🔬 Latest Evidence")
                if pubmed_results:
                    for paper in pubmed_results:
                        st.markdown(f"📄 **{paper['title']}**")
                        st.divider()
                else:
                    st.info("暫無關聯文獻。")

            # --- 區塊 3: 3D 結構 ---
            st.markdown("### 3️⃣ BBB Penetration & 3D Structure")
            c_chart, c_3d = st.columns(2)
            with c_chart:
                fig = go.Figure()
                fig.add_shape(type="circle", x0=0, y0=0, x1=6, y1=140, fillcolor="rgba(255, 204, 0, 0.2)")
                fig.add_trace(go.Scatter(x=[result['logp']], y=[result['tpsa']], mode='markers+text', text=[result['name']]))
                st.plotly_chart(fig, use_container_width=True)
            with c_3d:
                pdb_data = generate_3d_pdb(mol)
                if pdb_data:
                    v1 = py3Dmol.view(width=400, height=300)
                    v1.addModel(pdb_data, 'pdb')
                    v1.setStyle({'stick': {}})
                    v1.zoomTo()
                    showmol(v1, height=300, width=400)

            # --- 區塊 4: ADMET 規則 ---
            st.markdown("### 4️⃣ ADMET Risk Assessment")
            herg_r, herg_d, _ = admet.predict_herg(mol)
            liv_r, liv_d, _ = admet.predict_liver(mol)
            bbb_r, bbb_d, _ = admet.predict_bbb(mol)
            
            col_h, col_l, col_b = st.columns(3)
            # ... (此處保留您的卡片顯示代碼) ...

    elif page in ["🏠 Internal Dashboard", "📝 Database Settings"]:
        st.info("此模組為內部功能演示版。")
    main()
