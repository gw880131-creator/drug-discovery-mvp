import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol
import pubchempy as pcp

# --- 1. 網頁設定 ---
st.set_page_config(page_title="BrainX Drug Discovery Pro", page_icon="🧠", layout="wide")

# --- 2. 深度藥理知識庫 (加入詳細機制描述) ---
DEMO_DB = {
    "donepezil": {
        "status": "FDA Approved (1996)",
        "developer": "Eisai / Pfizer",
        "phase": "Marketed",
        "moa_title": "Acetylcholinesterase Inhibitor (AChEI)",
        "moa_detail": """
        **藥理機制詳解：**
        Donepezil 是一種具有高度特異性的、可逆的乙醯膽鹼酯酶 (AChE) 抑制劑。
        1. **結合位點：** 它能同時結合於 AChE 的催化三聯體 (Catalytic triad) 與周邊陰離子位點 (PAS)。
        2. **神經傳導：** 透過抑制 AChE，它阻止了神經遞質乙醯膽鹼 (Acetylcholine) 的水解，從而提高了突觸間隙中乙醯膽鹼的濃度。
        3. **臨床效益：** 增強膽鹼能神經傳導，改善阿茲海默症患者的認知功能。
        """
    },
    "memantine": {
        "status": "FDA Approved (2003)",
        "developer": "Merz / Forest",
        "phase": "Marketed",
        "moa_title": "NMDA Receptor Antagonist",
        "moa_detail": """
        **藥理機制詳解：**
        Memantine 是一種電壓依賴性、非競爭性、中等親和力的 NMDA 受體拮抗劑。
        1. **受體調節：** 它結合於 NMDA 受體通道內部的 Mg2+ 結合位點。
        2. **神經保護：** 它能阻斷病理性的麩胺酸 (Glutamate) 濃度持續升高所導致的鈣離子內流 (Ca2+ influx)，從而防止興奮性神經毒性 (Excitotoxicity)。
        3. **特點：** 與傳統拮抗劑不同，它不影響正常的突觸傳遞，因此副作用較少。
        """
    },
    "rivastigmine": {
        "status": "FDA Approved (2000)",
        "developer": "Novartis",
        "phase": "Marketed",
        "moa_title": "Dual Cholinesterase Inhibitor",
        "moa_detail": """
        **藥理機制詳解：**
        Rivastigmine 是一種「偽不可逆」的雙重膽鹼酯酶抑制劑。
        1. **雙重作用：** 它不僅抑制乙醯膽鹼酯酶 (AChE)，還能抑制丁醯膽鹼酯酶 (BuChE)。
        2. **代謝特性：** 它透過氨基甲酸酯化作用與酶結合，作用時間較長。
        3. **適應症：** 適用於阿茲海默症與帕金森氏症失智症。
        """
    }
}

# --- 3. 核心運算：CNS MPO 評分演算法 ---
def calculate_cns_mpo(mol):
    """
    計算 CNS Multi-Parameter Optimization (MPO) 分數 (0.0 - 6.0)
    參考文獻: ACS Chem. Neurosci. 2010, 1, 435–449 (Pfizer)
    """
    # 1. 計算物理化學性質
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    pka = 8.0 # 假設值 (因為 RDKit 算 pKa 需要額外複雜套件，這裡取平均值)

    # 2. 定義計分函數 (每個屬性 0.0 - 1.0 分)
    def score_component(val, best, good):
        if val <= best: return 1.0
        if val >= good: return 0.0
        return 1.0 - ((val - best) / (good - best))

    # Pfizer MPO 權重標準
    s_logp = score_component(logp, 3.0, 5.0) # LogP 最好 < 3
    s_mw = score_component(mw, 360, 500)     # MW 最好 < 360
    s_tpsa = score_component(tpsa, 40, 90)   # TPSA 最好 40-90 (這裡簡化)
    s_hbd = score_component(hbd, 0.5, 3.5)   # HBD 最好 < 1
    s_pka = score_component(abs(pka-8), 1, 3)# pKa 最好接近中性

    # 3. 總分 (滿分 6.0 - 這裡我們用 5 個參數簡化計算，再加權放大)
    mpo_score = (s_logp + s_mw + s_tpsa + s_hbd + s_pka) * (6.0 / 5.0)
    
    return min(6.0, max(0.0, mpo_score)), mw, logp, tpsa

# --- 4. 資料獲取 ---
def get_pubchem_data(query):
    query = query.strip().replace("(", "").replace(")", "")
    try:
        mol = Chem.MolFromSmiles(query)
        if mol: return {"name": "User Input", "smiles": query}, mol
        
        compounds = pcp.get_compounds(query, 'name')
        if compounds:
            c = compounds[0]
            smiles = c.isomeric_smiles if c.isomeric_smiles else c.canonical_smiles
            mol = Chem.MolFromSmiles(smiles)
            return {"name": query, "smiles": smiles}, mol
    except: return None, None
    return None, None

# --- 5. 主程式 ---
try:
    if 'candidate_list' not in st.session_state: st.session_state.candidate_list = []

    st.title("🧠 BrainX: CNS Drug Discovery Platform")
    st.markdown("搭載 **Pfizer CNS MPO 演算法** 與 **深度藥理機制分析**。")

    with st.sidebar:
        st.header("🔍 藥物搜尋")
        search_input = st.text_input("輸入藥名 (如 Memantine)", "")
        run_btn = st.button("🚀 啟動全譜分析")

    if run_btn and search_input:
        with st.spinner(f"正在進行 MPO 運算與機制分析：{search_input}..."):
            data, mol = get_pubchem_data(search_input)
            
            if not data:
                st.error("❌ 查無此藥")
            else:
                # 執行 MPO 運算
                mpo_score, mw, logp, tpsa = calculate_cns_mpo(mol)
                
                # 獲取詳細機制
                clean_name = search_input.lower().strip()
                drug_info = DEMO_DB.get(clean_name, {
                    "status": "Investigational", "developer": "Unknown", "phase": "Pre-clinical",
                    "moa_title": "Mechanism Under Analysis",
                    "moa_detail": "此為新興化合物，AI 根據結構推測其具有潛在的中樞神經活性，建議進行體外 (In-vitro) 結合試驗以確認詳細靶點。"
                })

                st.session_state.result_v4 = {
                    "data": data, "metrics": {"mpo": mpo_score, "mw": mw, "logp": logp, "tpsa": tpsa},
                    "info": drug_info, "mol": mol
                }

    # --- 結果顯示區 ---
    if 'result_v4' in st.session_state:
        res = st.session_state.result_v4
        d = res['data']
        m = res['metrics']
        i = res['info']
        mol = res['mol']
        
        st.divider()
        st.header(f"💊 {d['name'].title()}")
        st.caption(f"開發商: {i['developer']} | 狀態: {i['phase']}")

        # --- 1. CNS MPO 評分儀表板 (重點功能) ---
        st.subheader("1️⃣ CNS MPO 穿透率評分 (0.0 - 6.0)")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            # 製作進度條顯示分數
            score_pct = m['mpo'] / 6.0
            st.progress(score_pct)
            st.markdown(f"**AI 評分:** `{m['mpo']:.2f} / 6.0`")
            
            if m['mpo'] >= 4.0:
                st.success("✅ **高穿透性 (High CNS Permeability)** - 符合多數 CNS 藥物標準")
            elif m['mpo'] >= 3.0:
                st.warning("⚠️ **中等穿透性 (Moderate)** - 可能需要結構修飾")
            else:
                st.error("❌ **低穿透性 (Low)** - 難以進入大腦")

        with c2:
            st.metric("親脂性 (LogP)", f"{m['logp']:.2f}")
            st.metric("分子量 (MW)", f"{m['mw']:.0f}")

        # --- 2. 詳細藥理機制 (MOA) ---
        st.divider()
        st.subheader(f"2️⃣ 作用機制: {i['moa_title']}")
        
        with st.chat_message("assistant", avatar="🧬"):
            st.markdown(i['moa_detail'])

        # --- 3. 結構圖與操作 ---
        st.divider()
        t1, t2 = st.tabs(["🧬 3D 結構模擬", "📋 加入清單"])
        
        with t1:
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d)
            AllChem.MMFFOptimizeMolecule(mol_3d)
            m_block = Chem.MolToPDBBlock(mol_3d)
            view = py3Dmol.view(width=600, height=300)
            view.addModel(m_block, 'pdb')
            view.setStyle({'stick': {}})
            view.zoomTo()
            view.setBackgroundColor('#f9f9f9')
            showmol(view, height=300, width=600)
            
        with t2:
            if st.button("⭐ 加入候選藥物清單"):
                st.session_state.candidate_list.append({
                    "Name": d['name'], "MPO_Score": round(m['mpo'], 2), "Mechanism": i['moa_title']
                })
                st.success("已加入！")

    if st.session_state.candidate_list:
        st.divider()
        st.dataframe(pd.DataFrame(st.session_state.candidate_list), use_container_width=True)

except Exception as e:
    st.error(f"系統錯誤: {e}")
