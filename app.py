import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol

# --- 網頁設定 ---
st.set_page_config(page_title="BrainX AI Demo", page_icon="💊", layout="wide")

st.title("🧬 BrainX 藥物結構 AI 運算平台 (MVP)")
st.markdown("""
此平台展示 BrainX 的 AI 藥物研發能力。
輸入 **SMILES 化學代碼**，AI 將即時計算分子屬性並生成 3D 結構。
""")

# --- 側邊欄：輸入區 ---
st.sidebar.header("🧪 參數設定")
# 預設給一個阿斯匹靈的 SMILES
default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" 
smiles_input = st.sidebar.text_area("輸入化學分子式 (SMILES)", default_smiles, height=100)

if st.sidebar.button("🚀 開始運算 (Run AI)"):
    if not smiles_input:
        st.warning("請輸入 SMILES 代碼！")
    else:
        try:
            # 1. 讀取化學式
            mol = Chem.MolFromSmiles(smiles_input)
            mol = Chem.AddHs(mol) # 加氫原子
            
            # 2. AI 運算 3D 座標
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol) 
            
            # 3. 計算藥物屬性
            mol_wt = Descriptors.MolWt(mol)
            mol_logp = Descriptors.MolLogP(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            
            # --- 顯示結果 (兩欄排版) ---
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📊 預測數據")
                st.metric("分子量 (MW)", f"{mol_wt:.2f} g/mol")
                st.metric("親脂性 (LogP)", f"{mol_logp:.2f}")
                st.metric("氫鍵給體數", num_h_donors)
                st.metric("氫鍵受體數", num_h_acceptors)
                
                if mol_logp < 5 and mol_wt < 500:
                    st.success("✅ 符合 Lipinski 五規則 (類藥性高)")
                else:
                    st.warning("⚠️ 違反部分類藥性規則")

            with col2:
                st.subheader("🧬 3D 分子結構 (可旋轉)")
                # 繪製 3D 圖
                view = py3Dmol.view(width=800, height=500)
                pdb_block = Chem.MolToPDBBlock(mol)
                view.addModel(pdb_block, 'pdb')
                view.setStyle({'stick': {}}) # 棒狀模型
                view.setBackgroundColor('white')
                view.zoomTo()
                showmol(view, height=500, width=800)
                
        except Exception as e:
            st.error(f"❌ 無法識別結構，請確認 SMILES 格式正確。\n錯誤訊息: {e}")

else:
    st.info("👈 請在左側輸入化學式，並點擊「開始運算」")
