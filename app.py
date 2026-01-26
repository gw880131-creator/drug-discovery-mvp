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
st.set_page_config(page_title="BrainX Drug Screener", page_icon="🧠", layout="wide")

# --- 初始化「暫存記憶體」 (用來存您挑選的藥) ---
if 'candidate_list' not in st.session_state:
    st.session_state.candidate_list = []

# --- 核心函式 ---
def predict_bbb(mol):
    """
    簡易 BBB 穿透預測規則 (基於醫藥化學通則):
    通常 MW < 450 且 1.5 < LogP < 5.0 的小分子較容易通過血腦屏障。
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol) # 極性表面積
    
    # 這是非常經典的 BBB 預測法則 (Rule of Thumb)
    score = 0
    if mw < 450: score += 1
    if 1.5 < logp < 5.0: score += 1
    if tpsa < 90: score += 1
    
    is_permeable = score >= 2 # 只要符合其中兩項，判定為可穿透
    
    return is_permeable, mw, logp, tpsa

def get_structure(text):
    """嘗試從藥名或 SMILES 取得結構"""
    mol = Chem.MolFromSmiles(text)
    if mol: return mol, text, "SMILES Input"
    try:
        c = pcp.get_compounds(text, 'name')
        if c: return Chem.MolFromSmiles(c[0].canonical_smiles), c[0].canonical_smiles, "PubChem"
    except: pass
    return None, None, None

# --- 介面開始 ---
st.title("🧠 BrainX AI 藥物篩選與收藏系統")
st.markdown("輸入藥名或結構，AI 即時預測 **血腦屏障 (BBB)** 穿透性，並可將有潛力的藥物 **加入候選清單**。")

# --- 區塊 1: 搜尋與分析 ---
st.sidebar.header("🔍 藥物搜尋 (Search)")
search_input = st.sidebar.text_input("輸入藥名 (如 Levodopa) 或 SMILES", "")

if st.sidebar.button("🚀 開始分析"):
    if not search_input:
        st.warning("請輸入內容！")
    else:
        with st.spinner(f"正在分析 {search_input}..."):
            mol, smiles, source = get_structure(search_input)
            
            if not mol:
                st.error("❌ 找不到此藥物結構，請確認拼字。")
            else:
                # 1. 執行 BBB 預測
                is_bbb, mw, logp, tpsa = predict_bbb(mol)
                
                # 存入 Session State 供後續顯示
                st.session_state.current_analysis = {
                    "name": search_input,
                    "smiles": smiles,
                    "is_bbb": is_bbb,
                    "mw": mw,
                    "logp": logp,
                    "tpsa": tpsa,
                    "mol": mol # 暫存分子物件畫圖用
                }

# --- 顯示分析結果 (如果有的話) ---
if 'current_analysis' in st.session_state:
    data = st.session_state.current_analysis
    mol = data['mol']
    
    st.divider()
    st.subheader(f"🧪 分析結果: {data['name']}")
    
    # 版面：左邊數據 + BBB，中間 3D，右邊基因圖
    col1, col2, col3 = st.columns([1, 2, 1.5])
    
    with col1:
        st.markdown("### 🩸 血腦屏障 (BBB) 預測")
        if data['is_bbb']:
            st.success("✅ **高穿透率 (High Permeability)**")
            st.markdown("此藥物具有良好的親脂性與分子量，極有可能穿透 BBB。")
        else:
            st.error("⚠️ **穿透力不佳 (Low Permeability)**")
            st.markdown("分子過大或極性太高，建議進行結構修飾。")
            
        st.markdown("---")
        st.metric("分子量 (MW)", f"{data['mw']:.1f}")
        st.metric("親脂性 (LogP)", f"{data['logp']:.2f}")
        st.metric("極性表面積 (TPSA)", f"{data['tpsa']:.1f}")
        
        # 加入清單按鈕
        if st.button("⭐ 加入候選清單 (Add to List)"):
            # 檢查是否重複
            if not any(d['Name'] == data['name'] for d in st.session_state.candidate_list):
                st.session_state.candidate_list.append({
                    "Name": data['name'],
                    "BBB_Pass": "Yes" if data['is_bbb'] else "No",
                    "MW": round(data['mw'], 2),
                    "LogP": round(data['logp'], 2),
                    "SMILES": data['smiles']
                })
                st.toast(f"已將 {data['name']} 加入清單！")
            else:
                st.warning("此藥物已在清單中。")

    with col2:
        st.markdown("### 🧬 3D 結構視圖")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        view = py3D
