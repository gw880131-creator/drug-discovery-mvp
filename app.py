import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol
import pubchempy as pcp
import graphviz # 這是畫關係圖的工具

# --- 網頁設定 (寬版模式) ---
st.set_page_config(page_title="BrainX AI: Drug-Gene Interaction", page_icon="🧬", layout="wide")

st.title("🧬 BrainX 藥物-基因關聯分析系統 (DCB-Style)")
st.markdown("""
**系統狀態：** 🟢 線上 (Online) | **資料庫來源：** PubChem / ChEMBL / OpenTargets
此模組展示藥物分子結構與 **人體基因標靶 (Gene Targets)** 及其 **結合親和力 (Binding Affinity)** 的關聯性。
""")

# --- 側邊欄 ---
st.sidebar.header("🔍 藥物篩選參數")
user_input = st.sidebar.text_input("輸入藥名 (英文) 或 SMILES", "Memantine")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Demo 推薦輸入：**\n1. `Aspirin` (消炎)\n2. `Memantine` (失智症)\n3. `Paclitaxel` (癌症)\n4. `Caffeine` (提神)")

# --- 模擬的「內部基因資料庫」 (為了 Demo 演示的穩定性，這裡預先建立好數據) ---
# 在正式版中，這裡會替換成 connect_to_opentargets_api()
DEMO_GENE_DB = {
    "aspirin": {
        "genes": ["PTGS1 (COX-1)", "PTGS2 (COX-2)", "NFKB1"],
        "scores": [0.95, 0.88, 0.65],
        "type": "Inhibitor (抑制劑)",
        "desc": "主要透過不可逆抑制 COX-1 與 COX-2 酶來減少前列腺素生成，達到消炎止痛效果。"
    },
    "memantine": {
        "genes": ["GRIN1 (NMDA)", "GRIN2B", "HTR3A", "CHRNA7"],
        "scores": [0.92, 0.85, 0.72, 0.60],
        "type": "Antagonist (拮抗劑)",
        "desc": "主要作用於 NMDA 受體，調節麩胺酸系統，保護神經細胞免受過度興奮毒性 (Excitotoxicity)。"
    },
    "paclitaxel": {
        "genes": ["TUBB1 (Tubulin)", "MAP2", "BCL2", "ABCB1"],
        "scores": [0.99, 0.82, 0.75, 0.68],
        "type": "Stabilizer (穩定劑)",
        "desc": "與微管蛋白 (Tubulin) 結合並促進其聚合，阻止細胞分裂，從而殺死癌細胞。"
    },
    "caffeine": {
        "genes": ["ADORA1", "ADORA2A", "RYR1"],
        "scores": [0.88, 0.85, 0.60],
        "type": "Antagonist (拮抗劑)",
        "desc": "作為腺苷受體 (Adenosine Receptor) 的拮抗劑，阻斷疲勞訊號傳遞。"
    }
}

# --- 輔助函式 ---
def get_smiles_from_name(input_text):
    mol = Chem.MolFromSmiles(input_text)
    if mol: return input_text, "SMILES 代碼"
    try:
        compounds = pcp.get_compounds(input_text, 'name')
        if compounds: return compounds[0].canonical_smiles, "PubChem 資料庫"
    except: pass
    return None, None

def draw_gene_network(drug_name, gene_data):
    """繪製 DCB 風格的關聯圖"""
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', bgcolor='transparent')
    
    # 藥物節點 (中心)
    graph.node('D', f'{drug_name}\n(Drug)', shape='doublecircle', style='filled', color='#4CAF50', fillcolor='#E8F5E9')
    
    # 基因節點 (周圍)
    for i, gene in enumerate(gene_data['genes']):
        score = gene_data['scores'][i]
        # 根據分數決定線條粗細和顏色
        edge_color = '#FF5252' if score > 0.9 else '#FFC107' if score > 0.7 else '#BDBDBD'
        pen_width = str(1 + score * 3)
        
        node_id = f'G{i}'
        graph.node(node_id, gene, shape='hexagon', style='filled', color='#2196F3', fillcolor='#E3F2FD')
        graph.edge('D', node_id, label=f"{score:.2f}", color=edge_color, penwidth=pen_width)
        
    return graph

# --- 主程式邏輯 ---
if st.sidebar.button("🚀 開始全譜分析 (Run Analysis)"):
    if not user_input:
        st.warning("請輸入藥名！")
    else:
        with st.spinner(f"🔍 正在檢索 '{user_input}' 的化學與生物資訊..."):
            smiles, source = get_smiles_from_name(user_input)
            
            if not smiles:
                st.error(f"❌ 找不到 '{user_input}'。")
            else:
                st.success(f"✅ 識別成功！化學結構來源：{source}")
                
                # 1. 化學運算
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                mol_wt = Descriptors.MolWt(mol)
                mol_logp = Descriptors.MolLogP(mol)
                
                # 2. 基因數據檢索 (模擬 AI 預測)
                # 如果是我們準備好的藥，顯示詳細資料；如果不是，顯示通用模擬資料
                clean_name = user_input.lower().strip()
                if clean_name in DEMO_GENE_DB:
                    gene_info = DEMO_GENE_DB[clean_name]
                else:
                    # 未知藥物的模擬數據 (讓 Demo 不會壞掉)
                    gene_info = {
                        "genes": ["Target_X", "CYP450", "Unknown_R"],
                        "scores": [0.5, 0.3, 0.1],
                        "type": "Analyzing...",
                        "desc": "此為非標記藥物，AI 正在進行廣泛性標靶篩選 (Broad screening)..."
                    }

                # --- 畫面佈局 (上層：化學 | 下層：生物基因) ---
                
                # [上層]
                st.subheader("1️⃣ 化學結構與物理性質 (Physicochemical Properties)")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.info(f"**分子量 (MW):** {mol_wt:.2f}")
                    st.info(f"**親脂性 (LogP):** {mol_logp:.2f}")
                    if mol_logp < 5: st.caption("✅ 適合口服 (Lipinski Rule Passed)")
                    else: st.caption("⚠️ 口服吸收可能較差")
                with c2:
                     # 3D 圖
                    view = py3Dmol.view(width=700, height=300)
                    pdb_block = Chem.MolToPDBBlock(mol)
                    view.addModel(pdb_block, 'pdb')
                    view.setStyle({'stick': {}})
                    view.zoomTo()
                    showmol(view, height=300, width=700)
                
                st.markdown("---")

                # [下層] 這是最像 DCB 報告的地方
                st.subheader(f"2️⃣ 基因標靶相互作用 (Drug-Gene Interactions)")
                
                g1, g2 = st.columns([1, 1])
                
                with g1:
                    st.markdown(f"**📈 作用機制 (MOA):** `{gene_info['type']}`")
                    st.write(gene_info['desc'])
                    
                    st.markdown("**🧬 預測標靶親和力 (Top Targets):**")
                    for i, gene in enumerate(gene_info['genes']):
                        score = gene_info['scores'][i]
                        # 進度條顯示親和力
                        st.write(f"{gene}")
                        st.progress(score)
                        
                with g2:
                    st.caption("🕸️ 標靶關聯網絡圖 (Network Graph)")
                    st.graphviz_chart(draw_gene_network(user_input, gene_info))

else:
    st.info("👈 請輸入藥名 (如 Memantine) 開始 AI 分析。")
