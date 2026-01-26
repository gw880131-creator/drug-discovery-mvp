import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol
import graphviz

# --- 網頁設定 ---
st.set_page_config(page_title="BrainX Project Dashboard", page_icon="🧬", layout="wide")

# --- 模擬數據 (基於您的 Excel 截圖) ---
# 這裡建立了 5 個模擬的候選藥物資料
data = {
    "Compound_ID": [1727, 2130, 2284, 2554, 3108],
    "Name": ["4-Aminopyridine", "Amantadine", "Baclofen", "Carbamazepine", "Dipyridamole"],
    "CNS_Related": ["Yes", "Yes", "Yes", "Yes", "No"], # 模擬血腦屏障預測
    "Target_Gene": ["KCNA1", "NMDA (GRIN2A)", "GABAB (GABBR1)", "SCN1A", "PDE3A"],
    "SMILES": [
        "C1=CN=C(C=C1)N", 
        "C1C2CC3CC1CC(C2)(C3)N", 
        "C1C(C(=O)O)C(CC1)C2=CC=C(C=C2)Cl", 
        "C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N",
        "C1=CC=C(C=C1)N(CCO)CCO.C1=CC=C(C=C1)N(CCO)CCO.C2=NC3=C(N2)N(C(=N3)N(CCO)CCO)N(CCO)CCO" # 這裡簡化結構
    ],
    "Score": [0.88, 0.92, 0.85, 0.79, 0.45] # AI 預測分數
}
df = pd.DataFrame(data)

# --- 側邊欄：專案控制 ---
st.sidebar.title("📁 BrainX 專案管理")
project_phase = st.sidebar.radio("目前試驗階段 (Project Stage)", 
    ["Stage I: 標靶基因鎖定", "Stage II: 藥物比對與拓展", "Stage III: AI 藥物篩選 (FDA)"])

st.sidebar.markdown("---")
st.sidebar.info(f"**目前專案核心：** BX100\n**目標基因：** GLT-1 / EAAT2\n**候選藥物數：** {len(df)} compounds")

# --- 主畫面：標題與進度 ---
st.title("📊 BrainX AI 藥物開發試驗報告")

# 顯示進度條 (模擬您的 PPT 流程圖)
if "Stage I" in project_phase:
    st.progress(33)
    st.info("📌 **Stage I:** 正在建立 CNS 相關疾病 (PD, AD, ALS) 之候選基因清單。")
elif "Stage II" in project_phase:
    st.progress(66)
    st.success("📌 **Stage II (目前階段):** 已完成多基因分析網路。正在進行 **GLT-1 基因拓展** 與 **DGD 模組** 藥物比對。")
else:
    st.progress(100)
    st.warning("📌 **Stage III:** 進行 AI 藥物特徵微分分析與 FDA 藥物篩選 (PK/PSA/BBB 預測)。")

st.markdown("---")

# --- 區塊 1: 候選藥物總表 (模擬 Excel) ---
st.subheader("📋 候選藥物篩選列表 (Candidate Drug List)")
st.markdown("此表格展示經由 **DCB DGD 模組** 初步篩選之潛在藥物群。")

# 使用 Streamlit 的互動式表格，讓 CNS_Related 變色
def highlight_cns(val):
    color = '#d4edda' if val == 'Yes' else '#f8d7da' # Green for Yes, Red for No
    return f'background-color: {color}'

st.dataframe(
    df.style.applymap(highlight_cns, subset=['CNS_Related']),
    column_config={
        "Score": st.column_config.ProgressColumn("AI Affinity Score", format="%.2f", min_value=0, max_value=1),
        "SMILES": None # 隱藏太長的代碼，保持版面整潔
    },
    use_container_width=True
)

# --- 區塊 2: 詳細分析 (點擊後顯示) ---
st.subheader("🔍 單一藥物深度分析 (Deep Analysis)")

# 讓用戶選擇要看哪一個藥
selected_drug_name = st.selectbox("請選擇要分析的藥物 (Select Compound):", df['Name'])

# 抓取該藥物的資料
drug_data = df[df['Name'] == selected_drug_name].iloc[0]

# --- 顯示詳細資料 ---
c1, c2 = st.columns([1, 2])

with c1:
    st.markdown(f"### 💊 {selected_drug_name}")
    st.write(f"**Compound ID:** {drug_data['Compound_ID']}")
    
    # CNS 狀態燈號
    if drug_data['CNS_Related'] == 'Yes':
        st.success("✅ CNS Related: YES (可穿透血腦屏障)")
    else:
        st.error("❌ CNS Related: NO (無法穿透)")
        
    st.metric("主要標靶 (Target)", drug_data['Target_Gene'])
    st.metric("AI 結合親和力 (Score)", f"{drug_data['Score']}")

with c2:
    # 畫 3D 圖
    mol = Chem.MolFromSmiles(drug_data['SMILES'])
    if mol:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        
        view = py3Dmol.view(width=700, height=300)
        pdb_block = Chem.MolToPDBBlock(mol)
        view.addModel(pdb_block, 'pdb')
        view.setStyle({'stick': {}})
        view.setBackgroundColor('#f0f2f6') # 淡灰色背景比較像報告
        view.zoomTo()
        showmol(view, height=300, width=700)

# --- 區塊 3: 基因關聯圖 (Stage II 重點) ---
st.markdown("---")
st.subheader(f"🕸️ {selected_drug_name} 與 GLT-1 通路關聯圖 (Pathway Analysis)")

# 畫 DCB 風格的網路圖
graph = graphviz.Digraph()
graph.attr(rankdir='LR', bgcolor='transparent')

# 核心藥物
graph.node('D', f'{selected_drug_name}\n(Drug)', shape='doublecircle', style='filled', fillcolor='#4CAF50', fontcolor='white')

# 主要標靶
graph.node('T1', f"{drug_data['Target_Gene']}\n(Main Target)", shape='box', style='filled', fillcolor='#2196F3', fontcolor='white')

# GLT-1 (BrainX 核心)
graph.node('GLT1', 'GLT-1 / EAAT2\n(Core Target)', shape='hexagon', style='filled', fillcolor='#FF9800', fontcolor='black')

# 下游效應 (Downstream)
graph.node('E1', 'Neuroprotection\n(神經保護)', shape='ellipse', style='dashed')
graph.node('E2', 'Glutamate Uptake\n(麩胺酸回收)', shape='ellipse', style='dashed')

# 連線
graph.edge('D', 'T1', label=f"{drug_data['Score']}", penwidth='2')
graph.edge('T1', 'GLT1', label="regulation", style='dashed', color='gray')
graph.edge('GLT1', 'E1', color='#FF9800')
graph.edge('GLT1', 'E2', color='#FF9800')

# 顯示圖表
c3, c4 = st.columns([2, 1])
with c3:
    st.graphviz_chart(graph)
with c4:
    st.info("**分析解讀 (Insight):**")
    st.markdown(f"""
    此藥物透過作用於 **{drug_data['Target_Gene']}**，間接調節 **GLT-1 (EAAT2)** 的表現量。
    
    * **路徑強度:** {drug_data['Score']} (High Confidence)
    * **預期效果:** 增強麩胺酸回收能力，減少興奮性毒性。
    """)
    st.button("📄 下載詳細分析報告 (PDF)")
