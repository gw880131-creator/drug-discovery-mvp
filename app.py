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
st.set_page_config(page_title="BrainX Drug BI System", page_icon="💼", layout="wide")

# --- 初始化 Session (V2) ---
if 'candidate_list' not in st.session_state:
    st.session_state.candidate_list = []

# --- 💼 商業與臨床知識庫 (Demo 重點資料) ---
DEMO_DB = {
    "donepezil": {
        "status": "FDA Approved (1996)",
        "original_developer": "Eisai (衛采) / Pfizer (輝瑞)",
        "market_players": ["Eisai", "Pfizer", "Teva", "Sandoz (Generic)"],
        "phase": "Marketed (已上市)",
        "sales": "$820M (Global Estimate)"
    },
    "memantine": {
        "status": "FDA Approved (2003)",
        "original_developer": "Merz Pharma / Forest Labs",
        "market_players": ["AbbVie (Allergan)", "Merz", "Sun Pharma", "Dr. Reddy's"],
        "phase": "Marketed (已上市)",
        "sales": "$1.2B (Peak Sales)"
    },
    "rivastigmine": {
        "status": "FDA Approved (2000)",
        "original_developer": "Novartis (諾華)",
        "market_players": ["Novartis", "Sandoz"],
        "phase": "Marketed (已上市)",
        "sales": "Stable"
    },
    "riluzole": {
        "status": "FDA Approved (1995)",
        "original_developer": "Sanofi (賽諾菲)",
        "market_players": ["Sanofi", "Covis Pharma"],
        "phase": "Marketed (ALS Standard of Care)",
        "sales": "Niche Market"
    }
}

# --- 核心函式 ---
def get_pubchem_data(query):
    # 清理輸入
    query = query.strip().replace("(", "").replace(")", "")
    try:
        # 1. 先嘗試當作 SMILES
        mol = Chem.MolFromSmiles(query)
        if mol: 
            return {"name": "User Input", "smiles": query, "cid": "N/A"}, mol
        
        # 2. 當作藥名搜尋
        compounds = pcp.get_compounds(query, 'name')
        if compounds:
            c = compounds[0]
            # 修正警告：改用 isomeric_smiles (具立體化學資訊) 或 canonical_smiles
            # 若 PubChem 沒提供 isomeric，則退回到 canonical
            smiles_code = c.isomeric_smiles if c.isomeric_smiles else c.canonical_smiles
            
            mol = Chem.MolFromSmiles(smiles_code)
            return {
                "name": query, 
                "cid": c.cid, 
                "formula": c.molecular_formula,
                "smiles": smiles_code
            }, mol
    except Exception as e:
        return None, None
    return None, None

def predict_bbb(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    score = 0
    if mw < 450: score += 1
    if 1.5 < logp < 5.0: score += 1
    if tpsa < 90: score += 1
    return score >= 2, mw, logp, tpsa

# --- 介面開始 ---
st.title("💼 BrainX 藥物商業情報系統 (Business Intelligence)")
st.markdown("整合 **化學結構**、**FDA 臨床狀態** 與 **全球競品分析**，輔助高層進行藥物開發決策。")

# --- 側邊欄 ---
st.sidebar.header("🔍 藥物搜尋")
search_input = st.sidebar.text_input("輸入藥名 (如 Donepezil)", "")

if st.sidebar.button("🚀 啟動商業分析"):
    if not search_input:
        st.warning("請輸入藥名")
    else:
        with st.spinner(f"正在連線 FDA 與 專利資料庫分析 {search_input}..."):
            data, mol = get_pubchem_data(search_input)
            
            if not data:
                st.error("❌ 查無此藥 (僅支援小分子藥物)")
            else:
                # 1. 計算 BBB
                is_bbb, mw, logp, tpsa = predict_bbb(mol)
                
                # 2. 獲取商業資料
                clean_name = search_input.lower().strip()
                biz_data = DEMO_DB.get(clean_name, {
                    "status": "Investigational / Pre-clinical",
                    "original_developer": "Unknown / Novel Compound",
                    "market_players": ["Searching Global Databases..."],
                    "phase": "Research Phase",
                    "sales": "N/A"
                })
                
                # 關鍵修改：使用新的 key 'analysis_result_v2' 避免與舊快取衝突
                st.session_state.analysis_result_v2 = {
                    "data": data,
                    "metrics": {"is_bbb": is_bbb, "mw": mw, "logp": logp, "tpsa": tpsa},
                    "biz": biz_data,
                    "mol": mol
                }

# --- 顯示結果 ---
# 使用新的 key 讀取資料
if 'analysis_result_v2' in st.session_state:
    res = st.session_state.analysis_result_v2
    d = res['data']
    m = res['metrics']
    b = res['biz']
    mol = res['mol']
    
    st.divider()
    
    # 標題區
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown(f"## 💊 {d['name'].title()}")
    with col_status:
        if "Approved" in b['status']:
            st.success(f"✅ {b['status']}")
        else:
            st.warning(f"🧪 {b['status']}")

    # --- 商業情報儀表板 ---
    st.info("📊 **全球市場與競品分析 (Market & Competitors)**")
    
    k1, k2, k3 =
