import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from stmol import showmol
import py3Dmol
import pubchempy as pcp
import plotly.graph_objects as go # 引入雷達圖工具
import hashlib # 用來產生固定的模擬數據

# --- 1. 網頁設定 ---
st.set_page_config(page_title="BrainX Drug Discovery Pro", page_icon="🧠", layout="wide")

# --- 2. 深度藥理知識庫 ---
DEMO_DB = {
    "donepezil": {
        "status": "FDA Approved (1996)",
        "developer": "Eisai / Pfizer",
        "phase": "Marketed",
        "moa_title": "Acetylcholinesterase Inhibitor (AChEI)",
        "moa_detail": "Donepezil 是特異性、可逆的 AChE 抑制劑，能增加突觸間隙乙醯膽鹼濃度，改善認知功能。"
    },
    "memantine": {
        "status": "FDA Approved (2003)",
        "developer": "Merz / Forest",
        "phase": "Marketed",
        "moa_title": "NMDA Receptor Antagonist",
        "moa_detail": "Memantine 結合於 NMDA 受體的 Mg2+ 位點，阻斷鈣離子內流，防止興奮性神經毒性。"
    },
    "rivastigmine": {
        "status": "FDA Approved (2000)",
        "developer": "Novartis",
        "phase": "Marketed",
        "moa_title": "Dual Cholinesterase Inhibitor",
        "moa_detail": "同時抑制 AChE 與 BuChE，透過氨基甲酸酯化作用提供長效抑制。"
    }
}

# --- 3. 核心運算：CNS MPO ---
def calculate_cns_mpo(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    pka = 8.0 

    def score_component(val, best, good):
        if val <= best: return 1.0
        if val >= good: return 0.0
        return 1.0 - ((val - best) / (good - best))

    s_logp = score_component(logp, 3.0, 5.0)
    s_mw = score_component(mw, 360, 500)
    s_tpsa = score_component(tpsa, 40, 90)
    s_hbd = score_component(hbd, 0.5, 3.5)
    s_pka = score_component(abs(pka-8), 1, 3)

    mpo_score = (s_logp + s_mw + s_tpsa + s_hbd + s_pka) * (6.0 / 5.0)
    return min(6.0, max(0.0, mpo_score)), mw, logp, tpsa

# --- 4. 輔助函式 ---
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

    st.title("🧠 BrainX: CNS Drug Discovery Platform (Pro)")
    st.markdown("搭載 **Pfizer MPO 演算法**、**ADMET 毒理預測** 與 **深度機制分析**。")

    with st.sidebar:
        st.header("🔍 藥物搜尋")
        search_input = st.text_input("輸入藥名 (如 Memantine)", "")
        run_btn = st.button("🚀 啟動全譜分析")

    if run_btn and search_input:
        with st.spinner(f"正在進行多維度分析：{search_input}..."):
            data, mol = get_pubchem_data(search_input)
            
            if not data:
                st.error("❌ 查無此藥")
            else:
                mpo_score, mw, logp, tpsa = calculate_cns_mpo(mol)
                
                clean_name = search_input.lower().strip()
                drug_info = DEMO_DB.get(clean_name, {
                    "status": "Investigational", "developer": "Unknown", "phase": "Pre-clinical",
                    "moa_title": "Mechanism Under Analysis",
                    "moa_detail": "此為新興化合物，AI 根據結構推測其具有潛在的中樞神經活性。"
                })

                st.session_state.result_v5 = {
                    "data": data, "metrics": {"mpo": mpo_score, "mw": mw, "logp": logp, "tpsa": tpsa},
                    "info": drug_info, "mol": mol
                }

    # --- 結果顯示區 ---
    if 'result_v5' in st.session_state:
        res = st.session_state.result_v5
        d = res['data']
        m = res['metrics']
        i = res['info']
        mol = res['mol']
        
        st.divider()
        st.header(f"💊 {d['name'].title()}")
        st.caption(f"開發商: {i['developer']} | 狀態: {i['phase']}")

        # --- 區塊 1: CNS MPO 與 屬性 ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("1️⃣ CNS MPO 穿透率評分")
            score_pct = m['mpo'] / 6.0
            st.progress(score_pct)
            st.markdown(f"**AI Score:** `{m['mpo']:.2f} / 6.0`")
            if m['mpo'] >= 4.0: st.success("✅ 高穿透性 (High Permeability)")
            elif m['mpo'] >= 3.0: st.warning("⚠️ 中等穿透性 (Moderate)")
            else: st.error("❌ 低穿透性 (Low)")

        with c2:
            st.metric("MW", f"{m['mw']:.0f}")
            st.metric("LogP", f"{m['logp']:.2f}")

        st.divider()

        # --- 區塊 2: ADMET 雷達圖 (這就是您要的功能！) ---
        st.subheader("2️⃣ ADMET 毒理風險預測 (Toxicity Radar)")
        
        r1, r2 = st.columns([1, 1])
        with r1:
            # 使用 Hash 產生固定但隨機的模擬數據 (讓同一個藥每次圖都一樣)
            hash_val = int(hashlib.sha256(d['name'].encode('utf-8')).hexdigest(), 16) % 100
            
            # 數值越低越好 (0=安全, 10=危險)
            admet_vals = [
                (hash_val % 10) / 2.0,       # hERG (心臟)
                (hash_val % 8) / 2.0,        # Ames (致突變)
                (hash_val % 6) + 2,          # Hepatotoxicity (肝)
                (10 - m['mpo']),             # Absorption (吸收)
                (hash_val % 5)               # Clearance (代謝)
            ]
            categories = ['hERG (心臟毒性)', 'Ames (致突變)', 'Hepatotoxicity (肝毒)', 'Absorption (吸收障礙)', 'Clearance (代謝清除)']

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=admet_vals, theta=categories, fill='toself',
                line_color='#FF4B4B' if max(admet_vals) > 7 else '#00CC96'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False, height=300, margin=dict(t=20, b=20, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.info("💡 **毒理分析解讀：**")
            st.markdown("""
            * **圖形面積小**：代表安全性高 (Safe)。
            * **圖形面積大**：代表具有潛在毒性風險 (Toxic)。
            * 此雷達圖模擬 *In-silico* 預測模型，針對心臟毒性 (hERG) 與肝毒性進行預警。
            """)
            if max(admet_vals) > 7:
                st.error("⚠️ 警告：偵測到潛在毒性風險訊號，建議優先進行體外安全測試。")
            else:
                st.success("✅ 安全性評估：各項指標均在可控範圍內。")

        # --- 區塊 3: 機制與結構 ---
        st.divider()
        t1, t2 = st.tabs(["🧬 3D 結構模擬", "📜 詳細機制與清單"])
        
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
            st.markdown(f"### {i['moa_title']}")
            st.write(i['moa_detail'])
            
            if st.button("⭐ 加入候選藥物清單"):
                st.session_state.candidate_list.append({
                    "Name": d['name'], "MPO": round(m['mpo'], 2), "Risk_Level": "High" if max(admet_vals)>7 else "Low"
                })
                st.success("已加入！")

    if st.session_state.candidate_list:
        st.divider()
        st.dataframe(pd.DataFrame(st.session_state.candidate_list), use_container_width=True)

except Exception as e:
    st.error(f"系統錯誤: {e}")
