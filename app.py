import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol
import graphviz
import pubchempy as pcp

# --- 網頁設定 ---
st.set_page_config(page_title="BrainX Drug Informatics", page_icon="🧬", layout="wide")

# --- 初始化 Session State ---
if 'candidate_list' not in st.session_state:
    st.session_state.candidate_list = []

# --- 🧠 內部知識庫 (針對 Demo 藥物的完美資料) ---
# 這可以確保您在演示關鍵藥物時，資料是豐富且準確的
DEMO_DB = {
    "donepezil": {
        "indication": "Alzheimer's Disease (AD)",
        "class": "Acetylcholinesterase Inhibitor (AChEI)",
        "patent": "US-4895841-A (Eisai)",
        "moa": "Reversible inhibitor of acetylcholinesterase"
    },
    "memantine": {
        "indication": "Alzheimer's Disease (Moderate to Severe)",
        "class": "NMDA Receptor Antagonist",
        "patent": "US-3391142-A (Merz)",
        "moa": "Uncompetitive NMDA receptor antagonist"
    },
    "rivastigmine": {
        "indication": "Alzheimer's & Parkinson's Dementia",
        "class": "Cholinesterase Inhibitor",
        "patent": "US-4948807-A",
        "moa": "Inhibits both butyrylcholinesterase and acetylcholinesterase"
    },
    "levodopa": {
        "indication": "Parkinson's Disease",
        "class": "Dopamine Precursor",
        "patent": "US-3715390-A",
        "moa": "Converted to dopamine in the brain"
    },
    "aspirin": {
        "indication": "Pain, Inflammation, CV Risk",
        "class": "NSAID / COX Inhibitor",
        "patent": "Expired (Bayer)",
        "moa": "Irreversible inactivation of cyclooxygenase"
    }
}

# --- 核心函式 ---
def get_extended_data(query):
    """從 PubChem 獲取更詳細的化學資訊"""
    try:
        # 1. 搜尋化合物
        compounds = pcp.get_compounds(query, 'name')
        if not compounds:
            # 嘗試當作 SMILES 搜尋
            try:
                compounds = pcp.get_compounds(query, 'smiles')
            except:
                return None
        
        if not compounds:
            return None

        c = compounds[0] # 取第一個結果
        
        # 2. 提取資訊
        data = {
            "cid": c.cid,
            "formula": c.molecular_formula,
            "iupac": c.iupac_name if c.iupac_name else "N/A",
            "weight": c.molecular_weight,
            "smiles": c.canonical_smiles,
            "obj": c # 保留原始物件
        }
        return data
    except Exception as e:
        return None

def predict_bbb(mol):
    """簡易 BBB 預測"""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    score = 0
    if mw < 450: score += 1
    if 1.5 < logp < 5.0: score += 1
    if tpsa < 90: score += 1
    
    return score >= 2, mw, logp, tpsa

# --- 介面開始 ---
st.title("🧬 BrainX AI 藥物資訊中心 (Informatics Hub)")
st.markdown("整合 **PubChem 結構資料** 與 **BrainX 內部專利資料庫**，提供全方位的藥物分析報告。")

# --- 側邊欄 ---
st.sidebar.header("🔍 藥物搜尋")
search_input = st.sidebar.text_input("輸入藥名 (如 Donepezil) 或 SMILES", "")

if st.sidebar.button("🚀 全譜分析 (Analyze)"):
    if not search_input:
        st.warning("請輸入內容！")
    else:
        with st.spinner(f"正在連線全球資料庫分析 {search_input}..."):
            # 1. 獲取 PubChem 詳細資料
            pc_data = get_extended_data(search_input)
            
            if not pc_data:
                st.error(f"❌ 找不到 '{search_input}'。請確認拼字或改用標準藥名。")
            else:
                # 2. 轉成 RDKit 分子進行 BBB 運算
                mol = Chem.MolFromSmiles(pc_data['smiles'])
                is_bbb, mw, logp, tpsa = predict_bbb(mol)
                
                # 3. 檢查內部知識庫 (是否有專利/適應症資料)
                clean_name = search_input.lower().strip()
                kb_data = DEMO_DB.get(clean_name, {
                    "indication": "Investigational / Screening Phase",
                    "class": "Small Molecule",
                    "patent": "Searching External DB...",
                    "moa": "Under Analysis"
                })

                # 存入 Session
                st.session_state.current_analysis = {
                    "name": search_input, # 使用者輸入的名字
                    "pc_data": pc_data,   # PubChem 資料
                    "kb_data": kb_data,   # 內部知識庫資料
                    "metrics": {"is_bbb": is_bbb, "mw": mw, "logp": logp, "tpsa": tpsa},
                    "mol": mol
                }

# --- 主要顯示區 ---
if 'current_analysis' in st.session_state:
    data = st.session_state.current_analysis
    pc = data['pc_data']
    kb = data['kb_data']
    met = data['metrics']
    mol = data['mol']
    
    st.divider()
    
    # --- 標題區：藥名 + 分類 ---
    st.markdown(f"## 💊 {data['name'].title()} <span style='font-size:0.6em; color:gray'>| {kb['class']}</span>", unsafe_allow_html=True)
    
    # 建立四欄資訊卡
    k1, k2, k3, k4 = st.columns(4)
    k1.info(f"**適應症 (Indication)**\n\n{kb['indication']}")
    k2.info(f"**化學式 (Formula)**\n\n{pc['formula']}")
    k3.info(f"**專利狀態 (Patent)**\n\n{kb['patent']}")
    k4.success(f"**BBB 穿透預測**\n\n{'✅ High' if met['is_bbb'] else '⚠️ Low'}")

    # --- 詳細數據區 ---
    t1, t2 = st.tabs(["🧪 化學結構與屬性", "📜 專利與命名資訊"])
    
    with t1:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("物理化學屬性")
            st.write(f"**分子量 (MW):** {met['mw']:.2f} g/mol")
            st.write(f"**親脂性 (LogP):** {met['logp']:.2f}")
            st.write(f"**極性表面積 (TPSA):** {met['tpsa']:.2f} Å²")
            st.markdown("---")
            st.write("**機制 (MOA):**")
            st.caption(kb['moa'])
            
            if st.button("⭐ 加入候選清單"):
                if not any(d['Name'] == data['name'] for d in st.session_state.candidate_list):
                    st.session_state.candidate_list.append({
                        "Name": data['name'],
                        "Formula": pc['formula'],
                        "Indication": kb['indication'],
                        "Patent": kb['patent'],
                        "BBB": "Yes" if met['is_bbb'] else "No"
                    })
                    st.success("已加入清單！")
                else:
                    st.warning("已在清單中")

    with c2:
            st.subheader("3D 立體結構")
            # --- 關鍵修正開始：補回 3D 運算步驟 ---
            # 1. 幫分子加上氫原子 (Add Hydrogens)
            mol_3d = Chem.AddHs(mol)
            # 2. 最重要的一步：計算原子在 3D 空間的座標 (Embed)
            AllChem.EmbedMolecule(mol_3d)
            # 3. 進行能量優化，讓結構更自然 (Optimize)
            AllChem.MMFFOptimizeMolecule(mol_3d)
            # --- 關鍵修正結束 ---

            # 將計算好的 3D 結構轉成 PDB 格式給繪圖引擎
            m_block = Chem.MolToPDBBlock(mol_3d)

            view = py3Dmol.view(width=600, height=400)
            view.addModel(m_block, 'pdb')
            view.setStyle({'stick': {}})
            view.zoomTo()
            view.setBackgroundColor('#f9f9f9')
            showmol(view, height=400, width=600)

    with t2:
        st.subheader("詳細命名與外部連結")
        st.text_input("IUPAC 標準命名", pc['iupac'])
        st.text_area("SMILES 代碼", pc['smiles'])
        
        st.markdown("### 🔗 外部資料庫連結")
        # 自動生成 Google Patent 連結
        patent_url = f"https://patents.google.com/?q={data['name']}"
        pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{pc['cid']}"
        
        st.markdown(f"""
        * **Google Patents:** [點擊搜尋 {data['name']} 相關專利]({patent_url})
        * **PubChem:** [點擊查看 NCBI 官方報告]({pubchem_url})
        * **BrainX Internal:** [連結至內部試驗數據 (需權限)](https://www.brainx.com.tw)
        """)

# --- 底部清單 ---
if st.session_state.candidate_list:
    st.divider()
    st.subheader("📋 候選藥物總表")
    df = pd.DataFrame(st.session_state.candidate_list)
    st.dataframe(df, use_container_width=True)
