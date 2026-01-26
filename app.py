import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol
import pubchempy as pcp # 引入新朋友：PubChem 資料庫工具

# --- 網頁設定 ---
st.set_page_config(page_title="BrainX AI Drug Discovery", page_icon="💊", layout="wide")

st.title("🧬 BrainX 藥物結構 AI 運算平台 (Pro版)")
st.markdown("""
**升級功能：** 現在支援直接輸入 **藥物英文名稱** (如 Aspirin) 或 **SMILES 代碼**。
AI 將自動聯網搜尋結構，並進行 3D 建模與屬性預測。
""")

# --- 側邊欄：輸入區 ---
st.sidebar.header("🧪 藥物搜尋")
# 預設提示文字改得更直覺
user_input = st.sidebar.text_input("輸入藥名 (英文) 或 SMILES", "Aspirin")

def get_smiles_from_name(input_text):
    """嘗試將輸入文字轉換為 SMILES"""
    # 1. 先檢查是不是合法的 SMILES (如果 RDKit 讀得懂，就直接回傳)
    mol = Chem.MolFromSmiles(input_text)
    if mol:
        return input_text, "SMILES 代碼"
    
    # 2. 如果不是 SMILES，就當作藥名，去 PubChem 查
    try:
        compounds = pcp.get_compounds(input_text, 'name')
        if compounds:
            return compounds[0].canonical_smiles, "PubChem 資料庫"
    except:
        pass
    
    return None, None

if st.sidebar.button("🚀 開始運算 (Run AI)"):
    if not user_input:
        st.warning("請輸入藥物名稱或代碼！")
    else:
        with st.spinner(f"🔍 正在分析 '{user_input}' 的結構資料..."):
            # 取得 SMILES
            smiles_code, source = get_smiles_from_name(user_input)
            
            if not smiles_code:
                st.error(f"❌ 找不到 '{user_input}' 的結構資料。\n請確認拼字正確 (建議使用英文藥名) 或改用 SMILES。")
            else:
                try:
                    # 顯示它找到了什麼
                    st.info(f"✅ 識別成功！來源：{source}")
                    st.code(smiles_code, language="text") # 秀出轉換後的 SMILES 給江董看，證明有在算
                    
                    # --- 以下是原本的運算邏輯 (完全沒變) ---
                    mol = Chem.MolFromSmiles(smiles_code)
                    mol = Chem.AddHs(mol) 
                    AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(mol) 
                    
                    mol_wt = Descriptors.MolWt(mol)
                    mol_logp = Descriptors.MolLogP(mol)
                    num_h_donors = Descriptors.NumHDonors(mol)
                    num_h_acceptors = Descriptors.NumHAcceptors(mol)
                    
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
                        st.subheader(f"🧬 {user_input} 的 3D 結構")
                        view = py3Dmol.view(width=800, height=500)
                        pdb_block = Chem.MolToPDBBlock(mol)
                        view.addModel(pdb_block, 'pdb')
                        view.setStyle({'stick': {}})
                        view.setBackgroundColor('white')
                        view.zoomTo()
                        showmol(view, height=500, width=800)
                        
                except Exception as e:
                    st.error(f"❌ 運算發生錯誤: {e}")

else:
    st.info("👈 請在左側輸入 'Aspirin', 'Panadol' 或其他藥名。")
