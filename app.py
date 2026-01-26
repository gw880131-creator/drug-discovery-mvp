import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol  # <--- 關鍵修正：這裡一定要引用它！
import graphviz
import pubchempy as pcp

# --- 網頁設定 ---
st.set_page_config(page_title="BrainX Drug Screener", page_icon="🧠", layout="wide")

# --- 初始化「暫存記憶體」 ---
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
    
    score = 0
    if mw < 450: score += 1
    if 1.5 < logp < 5.0: score += 1
    if tpsa < 90: score += 1
    
    is_permeable = score >= 2 
    
    return is_permeable, mw, logp, tpsa

def get_structure(text):
    """嘗試從藥名或 SMILES 取得結構"""
    # 移除使用者不小心輸入的空白或標點符號
    text = text.strip().replace("(", "").replace(")", "")
    
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
search_input = st.sidebar.text_input("輸入藥名 (如 Donepezil) 或 SMILES", "")

if st.sidebar.button("🚀 開始分析"):
    if not search_input:
        st.warning("請輸入內容！")
    else:
        with st.spinner(f"正在分析 {search_input}..."):
            mol, smiles, source = get_structure(search_input)
            
            if not mol:
                st.error(f"❌ 找不到 '{search_input}' 的結構。\n提示：此系統專用於「小分子藥物」，若為抗體藥物 (如 Lecanemab) 請切換至大分子模組。")
            else:
                # 1. 執行 BBB 預測
                is_bbb, mw, logp, tpsa = predict_bbb(mol)
                
                # 存入 Session State
                st.session_state.current_analysis = {
                    "name": search_input,
                    "smiles": smiles,
                    "is_bbb": is_bbb,
                    "mw": mw,
                    "logp": logp,
                    "tpsa": tpsa,
                    "mol": mol 
                }

# --- 顯示分析結果 ---
if 'current_analysis' in st.session_state:
    data = st.session_state.current_analysis
    mol = data['mol']
    
    st.divider()
    st.subheader(f"🧪 分析結果: {data['name']}")
    
    col1, col2, col3 = st.columns([1, 2, 1.5])
    
    with col1:
        st.markdown("### 🩸 血腦屏障 (BBB) 預測")
        if data['is_bbb']:
            st.success("✅ **高穿透率**")
            st.caption("具備良好的親脂性與分子量，極有可能穿透 BBB。")
        else:
            st.error("⚠️ **穿透力不佳**")
            st.caption("分子過大或極性太高，建議進行結構修飾。")
            
        st.markdown("---")
        st.metric("親脂性 (LogP)", f"{data['logp']:.2f}")
        st.metric("極性表面積 (TPSA)", f"{data['tpsa']:.1f}")
        st.metric("分子量 (MW)", f"{data['mw']:.1f}")
        
        if st.button("⭐ 加入候選清單 (Add to List)"):
            if not any(d['Name'] == data['name'] for d in st.session_state.candidate_list):
                st.session_state.candidate_list.append({
                    "Name": data['name'],
                    "BBB_Pass": "Yes" if data['is_bbb'] else "No",
                    "MW": round(data['mw'], 2),
                    "LogP": round(data['logp'], 2),
                    "SMILES": data['smiles']
                })
                st.success(f"已將 {data['name']} 加入清單！")
            else:
                st.warning("此藥物已在清單中。")

    with col2:
        st.markdown("### 🧬 3D 結構視圖")
        # 這裡會使用到 py3Dmol，一定要確認上面有 import
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        
        view = py3Dmol.view(width=500, height=400)
        pdb = Chem.MolToPDBBlock(mol)
        view.addModel(pdb, 'pdb')
        view.setStyle({'stick': {}})
        view.zoomTo()
        view.setBackgroundColor('#f9f9f9')
        showmol(view, height=400, width=500)

    with col3:
        st.markdown("### 🕸️ 基因關聯圖")
        graph = graphviz.Digraph()
        graph.attr(rankdir='TB', bgcolor='transparent')
        graph.node('D', data['name'], shape='doublecircle', style='filled', fillcolor='#E0F7FA')
        graph.node('GLT1', 'GLT-1 / EAAT2', shape='hexagon', style='filled', fillcolor='#FFCC80')
        graph.node('NMDA', 'NMDA Receptor', shape='ellipse')
        
        graph.edge('D', 'GLT1', label="Target", color='red')
        graph.edge('D', 'NMDA', label="Modulate", style='dashed')
        st.graphviz_chart(graph)

# --- 區塊 2: 候選藥物清單 ---
st.divider()
st.subheader("📋 我的候選藥物清單")

if len(st.session_state.candidate_list) > 0:
    df = pd.DataFrame(st.session_state.candidate_list)
    st.dataframe(df, column_config={"SMILES": None}, use_container_width=True)
    
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("🗑️ 清空清單"):
            st.session_state.candidate_list = []
            st.rerun()
    with c2:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下載清單報告 (CSV)", csv, "brainx_candidates.csv", "text/csv")

else:
    st.info("目前清單是空的。請在上方搜尋藥物並點擊「加入候選清單」。")
