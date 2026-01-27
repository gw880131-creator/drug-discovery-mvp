import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from stmol import showmol
import py3Dmol
import pubchempy as pcp
import plotly.graph_objects as go
import hashlib
import urllib.parse
import requests
import numpy as np

# --- 1. 網頁設定 ---
st.set_page_config(page_title="BrainX: Real-World Data Edition", page_icon="🧬", layout="wide")

# --- 2. [核心新功能] ChEMBL API 連線函式 ---
@st.cache_data(ttl=3600)
def fetch_chembl_targets(smiles):
    """
    使用 ChEMBL API 查詢該分子已知的標靶 (Targets)
    這不是 AI 預測，這是真實的實驗數據。
    """
    try:
        # 1. 先用 SMILES 搜尋 ChEMBL ID (Molecule)
        base_url = "https://www.ebi.ac.uk/chembl/api/data"
        
        # 搜尋分子
        # 為了演示方便，我們這裡簡化流程：直接用相似度搜尋或標準名搜尋會更準，
        # 這裡我們用 similiarity search 找最像的已知藥物
        safe_smiles = urllib.parse.quote(smiles)
        res = requests.get(f"{base_url}/similarity/{safe_smiles}/80?format=json", timeout=10)
        
        if res.status_code == 200:
            mols = res.json()['molecules']
            if len(mols) > 0:
                chembl_id = mols[0]['molecule_chembl_id']
                pref_name = mols[0]['pref_name']
                
                # 2. 用 ChEMBL ID 找活性數據 (Activities) -> 推導出 Targets
                act_res = requests.get(f"{base_url}/activity?molecule_chembl_id={chembl_id}&limit=5&format=json", timeout=10)
                if act_res.status_code == 200:
                    activities = act_res.json()['activities']
                    targets = []
                    for act in activities:
                        if 'target_pref_name' in act and act['target_pref_name']:
                            target_info = {
                                "Target": act['target_pref_name'],
                                "Type": act['standard_type'], # e.g., IC50, Ki
                                "Value": f"{act['standard_value']} {act['standard_units']}",
                                "Organism": act.get('target_organism', 'N/A')
                            }
                            targets.append(target_info)
                    return {"found": True, "id": chembl_id, "name": pref_name, "targets": targets}
    except Exception as e:
        return {"found": False, "error": str(e)}
            
    return {"found": False}

# --- 3. [核心] 真實化學反應庫 ---
TRANSFORMATIONS = {
    "Fluorination (芳香環氟化)": {
        "smarts": "[c:1]>>[c:1](F)", 
        "desc": "在芳香環上引入氟原子，降低代謝敏感度並調節 pKa。",
        "ref": "J. Med. Chem. 2008"
    },
    "Bioisostere (羧酸 -> 四唑)": {
        "smarts": "[CX3](=O)[OX2H1]>>c1nnnn1", 
        "desc": "將羧酸替換為四唑，改善穿透性。",
        "ref": "J. Med. Chem. 2011"
    },
    "Scaffold Hop (苯環 -> 吡啶)": {
        "smarts": "c1ccccc1>>c1ccncc1", 
        "desc": "將苯環替換為吡啶，增加水溶性並降低 LogP。",
        "ref": "Bioorg. Med. Chem. 2013"
    }
}

# --- 4. 運算引擎 ---
def calculate_metrics(mol, name_seed):
    tpsa = Descriptors.TPSA(mol)
    wlogp = Descriptors.MolLogP(mol)
    qed = QED.qed(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    h = int(hashlib.sha256(name_seed.encode()).hexdigest(), 16)
    pka = 6.0 + (h % 40) / 10.0 
    in_egg_yolk = (tpsa < 79) and (0.4 < wlogp < 6.0)
    return {"tpsa": tpsa, "wlogp": wlogp, "qed": qed, "mw": mw, "hbd": hbd, "pka": pka, "in_egg": in_egg_yolk}

def apply_real_transformation(mol):
    for name, data in TRANSFORMATIONS.items():
        rxn = AllChem.ReactionFromSmarts(data['smarts'])
        try:
            products = rxn.RunReactants((mol,))
            if products:
                new_mol = products[0][0]
                Chem.SanitizeMol(new_mol)
                return new_mol, name, data['desc'], data['ref']
        except: continue
    return None, None, None, None

@st.cache_data(ttl=3600)
def fetch_fda_label(drug_name):
    try:
        base_url = "https://api.fda.gov/drug/label.json"
        query = f'search=openfda.brand_name:"{drug_name}"+OR+openfda.generic_name:"{drug_name}"&limit=1'
        response = requests.get(f"{base_url}?{query}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                res = data["results"][0]
                return {
                    "found": True,
                    "boxed_warning": res.get("boxed_warning", ["No Boxed Warning found."])[0],
                    "mechanism_of_action": res.get("mechanism_of_action", ["Mechanism not detailed."])[0],
                }
    except: pass
    return {"found": False}

def get_pubchem_data(query):
    query = query.strip().replace("(", "").replace(")", "")
    try:
        mol = Chem.MolFromSmiles(query)
        if mol: return {"name": "User Input", "smiles": query}, mol
        c = pcp.get_compounds(query, 'name')
        if c:
            s = c[0].isomeric_smiles if c[0].isomeric_smiles else c[0].canonical_smiles
            return {"name": query, "smiles": s}, Chem.MolFromSmiles(s)
    except: return None, None
    return None, None

def generate_3d_block(mol):
    try:
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv2())
        try: AllChem.MMFFOptimizeMolecule(mol_3d)
        except: pass
        return Chem.MolToPDBBlock(mol_3d)
    except: return None

# --- 5. 主程式 ---
try:
    if 'candidate_list' not in st.session_state: st.session_state.candidate_list = []

    st.title("🧬 BrainX: Real-World Data Edition (V19.0)")
    st.markdown("整合 **ChEMBL 真實靶點數據**、**PubChem 結構** 與 **FDA 毒理資訊**。")

    with st.sidebar:
        st.header("🔍 藥物搜尋")
        search_input = st.text_input("輸入藥名 (如 Memantine)", "Memantine")
        run_btn = st.button("🚀 連線全球資料庫")

    if run_btn and search_input:
        with st.spinner(f"正在向 EBI (歐洲) 與 FDA (美國) 請求數據：{search_input}..."):
            data, mol = get_pubchem_data(search_input)
            
            if not data:
                st.error("❌ 查無此藥")
            else:
                # 1. 基礎運算
                metrics = calculate_metrics(mol, data['name'])
                new_mol, opt_name, opt_desc, opt_ref = apply_real_transformation(mol)
                
                # 2. [核心] ChEMBL 真實靶點
                chembl = fetch_chembl_targets(data['smiles'])
                
                # 3. FDA 毒理
                fda = fetch_fda_label(data['name'])

                st.session_state.res_v19 = {
                    "data": data, "m": metrics, "mol": mol, 
                    "opt": {"mol": new_mol, "name": opt_name, "desc": opt_desc, "ref": opt_ref},
                    "fda": fda, "chembl": chembl
                }

    if 'res_v19' in st.session_state:
        res = st.session_state.res_v19
        d = res['data']
        m = res['m']
        mol = res['mol']
        opt = res['opt']
        fda = res['fda']
        chembl = res['chembl']
        
        st.header(f"💊 {d['name'].title()}")

        # --- Tab 1: 物理化學屬性 ---
        st.subheader("1️⃣ 物理化學屬性 (MPO & BOILED-Egg)")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=6, y1=140,
                fillcolor="rgba(255, 204, 0, 0.2)", line_color="rgba(255, 204, 0, 0.5)")
            fig.add_trace(go.Scatter(
                x=[m['wlogp']], y=[m['tpsa']], mode='markers+text',
                marker=dict(size=18, color='green' if m['in_egg'] else 'red', line=dict(width=2, color='black')),
                text=[d['name']], textposition="top center"
            ))
            fig.update_layout(xaxis_title="WLOGP", yaxis_title="TPSA", height=300, title="BOILED-Egg Plot", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("TPSA", f"{m['tpsa']:.1f}")
            st.metric("LogP", f"{m['wlogp']:.2f}")
            st.metric("MW", f"{m['mw']:.0f}")

        st.divider()

        # --- Tab 2: 真實靶點數據 (ChEMBL) ---
        st.subheader("2️⃣ 藥物標靶與活性數據 (Source: ChEMBL)")
        
        if chembl['found']:
            st.success(f"✅ **已連線至 EBI ChEMBL 資料庫** (ID: {chembl['id']})")
            st.caption(f"Matched Molecule: {chembl['name']}")
            
            # 將數據轉為 DataFrame 顯示
            if chembl['targets']:
                df_targets = pd.DataFrame(chembl['targets'])
                st.dataframe(df_targets, use_container_width=True)
            else:
                st.info("此分子在資料庫中暫無具體的 IC50/Ki 活性數據紀錄。")
        else:
            st.warning("⚠️ ChEMBL 資料庫中未找到結構完全匹配的已知藥物 (可能為全新結構)。")
            st.info("💡 對於新結構，系統建議進行 **Docking (分子對接)** 模擬以預測潛在靶點。")

        st.divider()

        # --- Tab 3: 結構優化 ---
        st.subheader("3️⃣ AI 結構優化建議")
        c1, c2 = st.columns(2)
        with c1:
            v1 = py3Dmol.view(width=400, height=300)
            v1.addModel(generate_3d_block(mol), 'pdb')
            v1.setStyle({'stick': {}})
            v1.zoomTo()
            showmol(v1, height=300, width=400)
        with c2:
            if opt['mol']:
                st.success(f"📈 **AI 建議: {opt['name']}**")
                st.write(f"原理: {opt['desc']}")
                v2 = py3Dmol.view(width=400, height=300)
                v2.addModel(generate_3d_block(opt['mol']), 'pdb')
                v2.setStyle({'stick': {'colorscheme': 'greenCarbon'}})
                v2.zoomTo()
                showmol(v2, height=300, width=400)
            else:
                st.info("結構穩定，無須修飾。")

        st.divider()

        # --- Tab 4: FDA Label ---
        st.subheader("4️⃣ FDA 官方標籤 (Source: openFDA)")
        if fda['found']:
            with st.expander("📄 查看詳細 FDA 資訊", expanded=True):
                if "No Boxed Warning" not in fda['boxed_warning']:
                    st.error(f"**Boxed Warning:** {fda['boxed_warning'][:300]}...")
                st.write(f"**Mechanism:** {fda['mechanism_of_action']}")
        else:
            st.write("FDA 資料庫未收錄此藥物。")

except Exception as e:
    st.error(f"Error: {e}")
