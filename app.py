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

# --- 初始化 Session ---
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
    query = query.strip().replace("(", "").replace(")", "")
    try:
        mol = Chem.MolFromSmiles(query)
        if mol: 
            return {"name": "User Input", "smiles": query, "cid": "N/A"}, mol
        
        compounds = pcp.get_compounds(query, 'name')
        if compounds:
            c = compounds[0]
            # 修正警告：改用 isomeric_smiles
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
