import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol
import graphviz
import pubchempy as pcp

# --- 1. 網頁設定 (必須在最前面) ---
st.set_page_config(page_title="BrainX Drug BI System", page_icon="💼", layout="wide")

# --- 2. 商業與臨床知識庫 (Demo 資料) ---
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

# --- 3. 核心函式定義 ---
def get_pubchem_data(query):
    query = query.strip().replace("(", "").replace(")", "")
    try:
        # 嘗試當作 SMILES
        mol = Chem.MolFromSmiles(query)
        if mol: 
            return {"name": "User Input", "smiles": query, "cid": "N/A"}, mol
        
        # 嘗試當作藥名
        compounds = pcp.get_compounds(query, 'name')
        if compounds:
            c = compounds[0]
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

# --- 4. 主程式介面 ---
try:
    # 初始化 Session
    if 'candidate_list' not in st.session_state:
        st.session_state.candidate_list = []

    st.title("💼 BrainX 藥物商業情報系統 (Business Intelligence)")
    st.markdown("整合 **化學結構**、**FDA 臨床狀態** 與 **全球競品分析**，輔助高層進行藥物開發決策。")

    # --- 側邊欄 (使用 with 語法確保顯示) ---
    with st.sidebar:
        st.header("🔍 藥物搜尋")
        search_input = st.text_input("輸入藥名 (如 Donepezil)", "")
        run_btn = st.button("🚀 啟動商業分析")

    # --- 按下按鈕後的邏輯 ---
    if run_btn:
        if not search_input:
            st.warning("請輸入藥名")
        else:
            with st.spinner(f"正在連線 FDA 與 專利資料庫分析 {search_input}..."):
                data, mol = get_pubchem_data(search_input)
                
                if not data:
                    st.error("❌ 查無此藥 (可能為大分子藥物或拼字錯誤)")
                else:
                    is_bbb, mw, logp, tpsa = predict_bbb(mol)
                    
                    clean_name = search_input.lower().strip()
                    biz_data = DEMO_DB.get(clean_name, {
                        "status": "Investigational / Pre-clinical",
                        "original_developer": "Unknown / Novel Compound",
                        "market_players": ["Searching Global Databases..."],
                        "phase": "Research Phase",
                        "sales": "N/A"
                    })
                    
                    # 存入結果
                    st.session_state.analysis_result_v3 = {
                        "data": data,
                        "metrics": {"is_bbb": is_bbb, "mw": mw, "logp": logp, "tpsa": tpsa},
                        "biz": biz_data,
                        "mol": mol
                    }

    # --- 顯示結果區域 ---
    if 'analysis_result_v3' in st.session_state:
        res = st.session_state.analysis_result_v3
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

        # 商業情報儀表板
        st.info("📊 **全球市場與競品分析 (Market & Competitors)**")
        
        # 這裡改用最穩定的寫法
        cols_biz = st.columns(3)
        cols_biz[0].metric("原廠開發商", b['original_developer'])
        cols_biz[1].metric("目前臨床階段", b['phase'])
        cols_biz[2].metric("預估市場規模", b['sales'])
        
        st.markdown("---")
        
        # 詳細分頁
        t1, t2, t3 = st.tabs(["🏭 主要販售藥廠", "🧬 結構與 BBB", "🔬 全球臨床試驗"])
        
        with t1:
            st.subheader("主要市場玩家")
            st.markdown(f"目前生產 **{d['name'].title()}** 的主要藥廠：")
            
            p_cols = st.columns(4)
            for i, player in enumerate(b['market_players']):
                with p_cols[i % 4]:
                    st.button(player, key=f"player_{i}", disabled=True)
            
            if len(b['market_players']) == 1 and "Searching" in b['market_players'][0]:
                st.warning("⚠️ 此為新興或研究用藥物，尚無大型藥廠量產紀錄。")

        with t2:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("BBB 穿透預測", "Pass ✅" if m['is_bbb'] else "Fail ❌")
                st.metric("親脂性 (LogP)", f"{m['logp']:.2f}")
                st.metric("TPSA", f"{m['tpsa']:.2f}")
            with c2:
