import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs # 用來算相似度的工具
from stmol import showmol
import py3Dmol
import graphviz
import pubchempy as pcp

# --- 網頁設定 ---
st.set_page_config(page_title="BrainX Drug Discovery Pro", page_icon="🧬", layout="wide")

# --- 初始化 Session State ---
if 'candidate_list' not in st.session_state:
    st.session_state.candidate_list = []

# --- 🧠 已知藥物參考庫 (用來做 AI 比對的標準答案) ---
# AI 會比對輸入的藥跟這些藥像不像，如果像，就預測一樣的標靶
REFERENCE_DB = [
    {"name": "Donepezil", "smiles": "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC", "target": "AChE (乙醯膽鹼酯酶)", "role": "Alzheimer's Treatment"},
    {"name": "Memantine", "smiles": "CC12CC3CC(C1)(CC(C3)(C2)N)C", "target": "NMDA Receptor", "role": "Alzheimer's Treatment"},
    {"name": "Rivastigmine", "smiles": "CCN(C)C(=O)OC1=CC=CC(=C1)C(C)N(C)C", "target": "AChE / BuChE", "role": "Dementia Treatment"},
    {"name": "Levodopa", "smiles": "C(C(C(=O)O)N)C1=CC(=C(C=C1)O)O", "target": "Dopamine Receptor (Precursor)", "role": "Parkinson's Treatment"},
    {"name": "Riluzole", "smiles": "C1=CC(=C(C=C1)OC(F)(F)F)NC(=S)N", "target": "Glutamate Transporter / Na+ Channel", "role": "ALS Treatment"},
    {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": "COX-1 / COX-2", "role": "Inflammation"},
    {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "target": "Adenosine Receptor", "role": "Stimulant"}
]

# --- 核心函式 ---
def get_extended_data(query):
    """從 PubChem 獲取資料"""
    # 移除使用者不小心輸入的空白或括號
    query = query.strip().replace("(", "").replace(")", "")
    
    try:
        # 1. 先試著當作 SMILES
        mol = Chem.MolFromSmiles(query)
        if mol:
            return {"name": "User Input", "smiles": query, "formula": Chem.RdMolDescriptors.CalcMolFormula(mol), "cid": "N/A", "iupac": "N/A"}
        
        # 2. 如果不是 SMILES，去搜尋藥名
        compounds = pcp.get_compounds(query, 'name')
        if compounds:
            c = compounds[0]
            return {
                "name": query,
                "cid": c.cid,
                "formula": c.molecular_formula,
                "iupac": c.iupac_name if c.iupac_name else "N/A",
                "weight": c.molecular_weight,
                "smiles": c.canonical_smiles
            }
    except:
        return None
    return None

def predict_target_by_similarity(user_mol):
    """
    AI 標靶預測核心：
    計算輸入藥物與資料庫藥物的『相似度 (Tanimoto Similarity)』。
    如果長得像 Donepezil，那它的標靶很可能就是 AChE。
    """
    # 1. 計算使用者藥物的指紋 (Fingerprint)
    user_fp = AllChem.GetMorganFingerprintAsBitVect(user_mol, 2, nBits=1024)
    
    best_match = None
    highest_score = 0.0
    
    # 2. 跟資料庫裡的每一個藥比對
    for ref_drug in REFERENCE_DB:
        ref_mol = Chem.MolFromSmiles(ref_drug['smiles'])
        if ref_mol:
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
