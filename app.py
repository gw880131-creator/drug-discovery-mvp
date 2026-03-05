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
        # 鎖定您的研究專案核心靶點
        self.ad_pd_keys = ["EAAT2", "GLT-1", "BACE1", "AMYLOID", "TAU", "MAO-B", "GSK-3", "ACHE"]

    def query_pubchem(self, identifier, id_type="name"):
        """【真·即時查詢】自動解析結構"""
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
    def get_pubmed_details(self, drug_name):
        """【全新功能】抓取 PubMed 論文標題與連結，為研發提供即時佐證"""
        import urllib.parse
        # 針對 EAAT2 與神經保護路徑進行專業檢索
        search_term = f"({drug_name}) AND (EAAT2 OR GLT-1 OR neuroprotection)"
        try:
            # 1. 搜尋 PMID
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(search_term)}&retmode=json&retmax=3"
            res = requests.get(search_url, timeout=10)
            id_list = res.json().get('esearchresult', {}).get('idlist', [])
            
            if not id_list: return []
            
            # 2. 根據 PMID 抓取標題 (Summary)
            ids = ",".join(id_list)
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids}&retmode=json"
            sum_res = requests.get(summary_url, timeout=10)
            result_set = sum_res.json().get('result', {})
            
            papers = []
            for pmid in id_list:
                title = result_set.get(pmid, {}).get('title', '無標題資訊')
                papers.append({"title": title, "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
            return papers
        except: return []
    def get_clinical_summary(self, drug_name):
        """【實時檢索修正】改用模糊檢索與多重回退機制，確保 Ceftriaxone 等藥物必能查獲"""
        import urllib.parse
        search_name = drug_name.strip()
        
        try:
            # 1. Wikipedia 檢索 (加入歸一化：首字母大寫)
            wiki_name = search_name.capitalize()
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(wiki_name)}"
            res = requests.get(wiki_url, timeout=10)
            if res.status_code == 200:
                return res.json().get('extract', "找到紀錄但無摘要。")

            # 2. ChEMBL 檢索修正：捨棄 iexact (精確)，改用模糊搜尋 (__contains) 或 ID 轉換
            # 先找分子 ID，再找機制，這才是專業藥化系統的標準作法
            mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?pref_name__icontains={search_name}&format=json"
            mol_res = requests.get(mol_url, timeout=10)
            
            if mol_res.status_code == 200:
                mols = mol_res.json().get('molecules', [])
                if mols:
                    chembl_id = mols[0].get('molecule_chembl_id')
                    # 再次請求該 ID 的作用機制
                    mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism?molecule_chembl_id={chembl_id}&format=json"
                    mech_res = requests.get(mech_url, timeout=10)
                    
                    if mech_res.status_code == 200:
                        mechanisms = mech_res.json().get('mechanisms', [])
                        if mechanisms:
                            return "\n".join([f"🎯 **實時藥理**: {m.get('mechanism_of_action')} (靶點: {m.get('target_name')})" for m in mechanisms])

            return f"❌ 實時檢索仍無匹配。請檢查輸入名稱或參考下方由 ChEMBL 實時運算的靶點預測圖表。"
            
        except Exception as e:
            return f"⚠️ 網路連線波動: {str(e)}"

    def get_pubmed_links(self, drug_name):
        """【PubMed 自動檢索】抓取最新的 EAAT2 相關文獻"""
        import urllib.parse
        search_term = f"({drug_name}) AND (EAAT2 OR GLT-1 OR neuroprotection)"
        try:
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(search_term)}&retmode=json&retmax=5"
            res = requests.get(search_url, timeout=10)
            if res.status_code == 200:
                id_list = res.json().get('esearchresult', {}).get('idlist', [])
                if not id_list: return "目前 PubMed 暫無此藥物與 EAAT2 關聯的直接文獻。"
                return "\n".join([f"- [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" for pmid in id_list])
        except: return "PubMed 連線異常。"
        return "檢索中..."

    def predict_targets(self, smiles, drug_name):
        """【優化版即時預測】自動擴大搜尋半徑以確保結果呈現"""
        import urllib.parse
        try:
            base_url = "https://www.ebi.ac.uk/chembl/api/data"
            # 若 80% 相似度無結果，則自動降低至 70% 擴大檢索範圍
            for threshold in [80, 70]:
                res = requests.get(f"{base_url}/similarity/{urllib.parse.quote(smiles)}/{threshold}?format=json", timeout=300)
                if res.status_code == 200 and res.json().get('molecules'):
                    similar_mols = res.json()['molecules'][:10] # 增加比對樣本數
                    targets_map = {}
                    for m in similar_mols:
                        sim_score = float(m['similarity'])
                        act_res = requests.get(f"{base_url}/activity?molecule_chembl_id={m['molecule_chembl_id']}&limit=30&format=json", timeout=60)
                        if act_res.status_code == 200:
                            for act in act_res.json().get('activities', []):
                                t_name = act.get('target_pref_name')
                                species = act.get('target_organism', 'Unknown')
                                if t_name and "unspecified" not in t_name.lower():
                                    weight = 5.0 if any(key in t_name.upper() for key in self.ad_pd_keys) else 1.0
                                    display_name = f"{t_name} [{species}]"
                                    targets_map[display_name] = targets_map.get(display_name, 0) + (sim_score * weight)
                    
                    if targets_map:
                        sorted_res = sorted(targets_map.items(), key=lambda x: x[1], reverse=True)[:8]
                        max_v = sorted_res[0][1]
                        return [{"Target": t[0], "Score": round((t[1]/max_v)*99.5, 1), "Class": "ChEMBL Live"} for t in sorted_res]
            
        except Exception: pass
        return [{"Target": "擴大搜尋後仍無相似活性紀錄", "Score": 0.0, "Class": "N/A"}]

    def get_modification_suggestions(self, result):
        """【藥化專家系統】使用 SMARTS 進行精準官能基偵測與 CNS MPO 運算"""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(result['smiles'])
        if not mol: return ["❌ 分子結構解析失敗"]
        
        logp, tpsa = result['logp'], result['tpsa']
        suggestions = []
        
        # 1. 使用 SMARTS 定義藥化關鍵官能基 (這能解決 Fragments 報錯並提升精準度)
        # 偵測羥基 (-OH), 羧酸 (-COOH), 與氟原子 (F)
        oh_pattern = Chem.MolFromSmarts('[OX2H]')
        acid_pattern = Chem.MolFromSmarts('C(=O)[O;H1,H0-]')
        f_pattern = Chem.MolFromSmarts('[F]')
        
        oh_count = len(mol.GetSubstructMatches(oh_pattern))
        has_acid = mol.HasSubstructMatch(acid_pattern)
        f_count = len(mol.GetSubstructMatches(f_pattern))

        # 2. 實時計算 CNS MPO 分數 (藥化入腦潛力指標)
        mpo_score = 0
        if 1.0 < logp < 3.0: mpo_score += 1.0
        if tpsa < 75: mpo_score += 1.0
        if Descriptors.NumHDonors(mol) <= 1: mpo_score += 1.0

        suggestions.append(f"📊 **實時 CNS MPO 指數: {mpo_score}/3.0**")

        # 3. 差異化建議：區分為何藥物入腦困難
        # 針對 BrainX 的 EAAT2 誘導劑研究，酸性基團是關鍵瓶頸
        if has_acid:
            suggestions.append("🚫 **關鍵瓶頸：酸性基團**。偵測到羧酸根。在生理 pH 下帶負電，這會導致分子極難穿透血腦屏障 (BBB)。")
            suggestions.append("🔹 **策略**: 可嘗試將其置換為四唑 (Tetrazole) 或進行前藥 (Prodrug) 化。")
            
        
        if oh_count > 2:
            suggestions.append(f"⚠️ **極性負擔**: 含有 {oh_count} 個羥基。")
            suggestions.append("🔹 **策略**: 羥基過多會顯著增加氫鍵給體數量，建議透過烷基化或隱藏在分子內氫鍵中來改善。")

        if f_count == 0 and logp < 2.0:
            suggestions.append("💡 **優化路徑**: 目前結構缺乏氟原子 (F)。")
            suggestions.append("🔹 **策略**: 引入氟原子通常能提升代謝穩定性並微調親脂性，有利於 AD 藥物的腦部滲透。")

        if mpo_score >= 2.5 and not has_acid:
            suggestions.append("✅ **結構參數理想**: 理化性質與成功入腦的神經藥物高度契合。")
            
        return suggestions
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

    # --- Page: Public DB & Target Prediction ---
    if page == "🌐 Drug Query & Target Prediction":
        st.header("Drug Query & AI Target Prediction")
        st.caption("輸入藥物，系統將即時解析結構、預測靶點 (Target) 並評估 ADMET 風險。")
        
        query = st.text_input("Enter Drug Name or SMILES (e.g., Donepezil, Amoxicillin)", "Donepezil")
        if st.button("🚀 Analyze & Predict", use_container_width=True):
            with st.spinner("Running AI Models and Target Prediction..."):
                
                result = public_api.query_pubchem(query, "name" if "1" not in query and "C" not in query else "smiles")
                
                if not result:
                    st.error("❌ 無法解析分子結構，請檢查輸入。")
                else:
                    mol = Chem.MolFromSmiles(result['smiles'])
                    
                    # === 區塊 1: 物理化學儀表板 ===
                    st.markdown("### 1️⃣ Physicochemical Profile")
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("MW", f"{result['mw']:.1f}")
                    k2.metric("LogP", f"{result['logp']:.2f}")
                    k3.metric("TPSA", f"{result['tpsa']:.1f}")
                    k4.metric("HBD", f"{Descriptors.NumHDonors(mol)}")
                    k5.metric("QED", f"{QED.qed(mol):.2f}")
                    
                    with st.expander("📖 點擊查看：五大指標科學原理詳解 (Scientific Rationale)", expanded=False):
                        st.markdown("""
                        | 指標 (Metric) | 理想範圍 | 科學原理 (Scientific Rationale) |
                        | :--- | :--- | :--- |
                        | **TPSA** (極性表面積) | < 79 Å² | **反映去溶劑化能。** TPSA 過高代表能障過大，難以入腦。 |
                        | **LogP** (親脂性) | 0.4 - 6.0 | **決定磷脂雙分子層的親和力。** 需具備適當脂溶性以穿透細胞膜。 |
                        | **MW** (分子量) | < 360 Da | **空間障礙。** 分子量越小，擴散係數越高。 |
                        | **HBD** (氫鍵給體) | < 1 | **水合層效應。** 氫鍵給體易與水形成強鍵結，阻礙穿透。 |
                        | **pKa** (酸鹼度) | 7.5 - 8.5 | **離子化狀態。** 只有未帶電的中性分子能有效藉由被動擴散通過。 |
                        """)
                    # === 區塊 1.2: 臨床背景分析 (DrugBank/Wiki 聯網) ===
                    st.markdown("### 📚 Clinical Background & Mechanism")
                    clinical_info = public_api.get_clinical_summary(query)
                    st.write(clinical_info)
                    # === 區塊 1.3: PubMed 文獻追蹤 (針對 EAAT2 / 神經保護) ===
                    st.markdown("### 🔬 Related Scientific Publications (PubMed)")
                    pubmed_results = public_api.get_pubmed_links(query)
                    st.markdown(pubmed_results)
                    # === 區塊 1.5: 化學修飾專家建議 (針對 AD/PD) ===
                    st.markdown("### 🛠️ AI Chemical Modification Suggestions")
                    mod_suggestions = public_api.get_modification_suggestions(result)
                    for advice in mod_suggestions:
                        st.info(advice)
                   # === 區塊 2: AI 靶點預測與文獻連動 ===
                    st.markdown("### 🎯 2️⃣ AI Target Prediction & PubMed Evidence")
                    
                    col_chart, col_papers = st.columns([3, 2])
                    
                    with col_chart:
                        targets_data = public_api.predict_targets(result['smiles'], result['name'])
                        if targets_data:
                            df_targets = pd.DataFrame(targets_data)
                            fig_t = px.bar(df_targets, x="Score", y="Target", orientation='h', 
                                           color="Score", color_continuous_scale="Blues")
                            fig_t.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                            st.plotly_chart(fig_t, use_container_width=True)
                        else:
                            st.warning("目前資料庫無相似活性紀錄。")
                            
                    with col_papers:
                        st.markdown("#### 🔬 Latest Evidence (PubMed)")
                        paper_list = public_api.get_pubmed_details(query)
                        if paper_list:
                            for paper in paper_list:
                                st.markdown(f"📄 **{paper['title']}**")
                                st.markdown(f"🔗 [查看原文]({paper['link']})")
                                st.divider()
                        else:
                            st.info("暫無直接關聯之 EAAT2 相關文獻。")
                    
                    # === 區塊 3: BOILED-Egg & 3D Viewer ===
                    st.markdown("### 3️⃣ BBB Penetration & 3D Structure")
                    c_chart, c_3d = st.columns(2)
                    with c_chart:
                        fig = go.Figure()
                        fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=6, y1=140, fillcolor="rgba(255, 204, 0, 0.2)", line_color="rgba(255, 204, 0, 0.5)")
                        fig.add_trace(go.Scatter(x=[result['logp']], y=[result['tpsa']], mode='markers+text', marker=dict(size=18, color='#3b82f6'), text=[result['name']], textposition="top center"))
                        fig.update_layout(xaxis_title="WLOGP", yaxis_title="TPSA", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'), height=350, margin=dict(t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    with c_3d:
                        v1 = py3Dmol.view(width=400, height=300)
                        pdb_data = generate_3d_pdb(mol)
                        if pdb_data:
                            v1.addModel(pdb_data, 'pdb')
                            v1.setStyle({'stick': {}})
                            v1.zoomTo()
                            showmol(v1, height=300, width=400)
                        else:
                            st.warning("無法生成 3D 結構")
                    
                    # === 區塊 4: ADMET 規則引擎 ===
                    st.markdown("### 4️⃣ ADMET Risk Assessment")
                    herg_r, herg_d, herg_ref = admet.predict_herg(mol)
                    liv_r, liv_d, liv_ref = admet.predict_liver(mol)
                    bbb_r, bbb_d, bbb_ref = admet.predict_bbb(mol)
                    
                    col_h, col_l, col_b = st.columns(3)
                    with col_h:
                        c_code = "risk-high" if herg_r == "High" else "risk-medium" if herg_r == "Moderate" else "risk-low"
                        b_code = "#ef4444" if herg_r == "High" else "#f59e0b" if herg_r == "Moderate" else "#10b981"
                        st.markdown(f'<div style="background:rgba(255,255,255,0.8); border-radius:12px; padding:15px; border-top:4px solid {b_code}; box-shadow:0 2px 4px rgba(0,0,0,0.05);"><h4>🫀 hERG Risk</h4><p class="{c_code}" style="font-size:1.2rem;">{herg_r}</p><p style="font-size:0.8rem;color:#64748b;">{herg_d}</p></div>', unsafe_allow_html=True)
                    with col_l:
                        c_code = "risk-high" if liv_r == "High" else "risk-medium" if liv_r == "Moderate" else "risk-low"
                        b_code = "#ef4444" if liv_r == "High" else "#f59e0b" if liv_r == "Moderate" else "#10b981"
                        st.markdown(f'<div style="background:rgba(255,255,255,0.8); border-radius:12px; padding:15px; border-top:4px solid {b_code}; box-shadow:0 2px 4px rgba(0,0,0,0.05);"><h4>🧪 Liver DILI</h4><p class="{c_code}" style="font-size:1.2rem;">{liv_r}</p><p style="font-size:0.8rem;color:#64748b;">{liv_d}</p></div>', unsafe_allow_html=True)
                    with col_b:
                        c_code = "risk-high" if bbb_r == "Low" else "risk-medium" if bbb_r == "Moderate" else "risk-low"
                        b_code = "#ef4444" if bbb_r == "Low" else "#f59e0b" if bbb_r == "Moderate" else "#10b981"
                        st.markdown(f'<div style="background:rgba(255,255,255,0.8); border-radius:12px; padding:15px; border-top:4px solid {b_code}; box-shadow:0 2px 4px rgba(0,0,0,0.05);"><h4>🧠 BBB Penetration</h4><p class="{c_code}" style="font-size:1.2rem;">{bbb_r}</p><p style="font-size:0.8rem;color:#64748b;">{bbb_d}</p></div>', unsafe_allow_html=True)

    # 其他佔位分頁
    elif page in ["🏠 Internal Dashboard", "📝 Database Settings"]:
        st.info("此模組為內部功能演示版，請切換至 'Drug Query' 體驗核心預測引擎。")

if __name__ == "__main__":
    main()
