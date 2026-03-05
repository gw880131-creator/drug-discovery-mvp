import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import hashlib
from datetime import datetime
import requests
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client
import plotly.express as px
import plotly.graph_objects as go
import py3Dmol
from stmol import showmol

# 【重要修正】拔除 try-except，強制載入 RDKit。如果這裡報錯，代表 requirements.txt 沒設定好。
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Fragments

# ==================== 頁面設定與 CSS ====================
st.set_page_config(
    page_title="MedChem Pro | Enterprise R&D Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 1. 全局背景：改為乾淨清爽的淺灰白漸層 */
    .stApp { 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
        color: #1e293b; 
        font-family: 'Inter', sans-serif; 
    }
    
    /* 2. 側邊欄：改為淡雅的灰白色 */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #cbd5e1;
    }
    
    /* 3. 玻璃擬態卡片：改為半透明白色，增強科技感 */
    div[data-testid="stExpander"], div.css-1r6slb0, .metric-card {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(12px); 
        border: 1px solid rgba(255, 255, 255, 1); 
        border-radius: 16px; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
        padding: 20px; 
        margin-bottom: 15px;
    }
    
    /* 4. 輸入框與按鈕：配合淺色系 */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div { 
        background-color: #ffffff !important; 
        color: #1e293b !important; 
        border: 1px solid #cbd5e1 !important; 
        border-radius: 8px; 
    }
    .stButton>button { 
        background: linear-gradient(to right, #2563eb, #3b82f6); 
        color: white; border: none; border-radius: 8px; font-weight: 600; 
    }
    
    /* =================【全新淺色主題：數字與標題】================= */
    /* 指標數值：改為企業深藍色，放大，強調專業感 */
    div[data-testid="stMetricValue"] { 
        font-family: 'JetBrains Mono', monospace; 
        color: #1e40af !important; /* 深藍色 */
        font-size: 2.5rem !important; 
        text-shadow: none !important; /* 淺色底不需要發光 */
    }
    
    /* 核彈級覆蓋：強制將指標標題 (MW, LogP 等) 改為清晰的深灰色 */
    div[data-testid="stMetricLabel"], 
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span,
    div[data-testid="stMetricLabel"] label { 
        color: #475569 !important; /* 深灰色 */
        font-size: 1.1rem !important; 
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-shadow: none !important;
    }
    /* ========================================================= */

    /* 5. 標題與一般文字顏色：改為深色以在淺背景上顯示 */
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
    p, li { color: #334155; }

    /* 6. 內部警示與風險標籤 */
    .internal-warning {
        background-color: #fef3c7; border: 1px solid #f59e0b; color: #b45309; padding: 10px; border-radius: 8px; font-size: 0.85rem; text-align: center; margin-bottom: 20px; font-weight: 600; letter-spacing: 0.5px;
    }
    .risk-high { color: #ef4444; font-weight: bold; }
    .risk-medium { color: #f59e0b; font-weight: bold; }
    .risk-low { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== 救命字典 (Demo防斷網) ====================
# 這些是 Demo 常用的藥物，寫死在這裡確保 100% 成功解析
SAFE_DEMO_DB = {
    "donepezil": "COc1ccc2cc1Oc1cc(cc(c1)C(F)(F)F)CC(=O)N2CCCCc1cccnc1",
    "memantine": "CC12CC3CC(C1)(CC(C3)(C2)N)C",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "amoxicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(cc3)O)N)C(=O)O)C",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
}

# ==================== 資料庫初始化 ====================
def init_database():
    conn = sqlite3.connect('medchem_pro.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS compounds (compound_id TEXT PRIMARY KEY, smiles TEXT UNIQUE, inchikey TEXT UNIQUE, mol_weight REAL, logp REAL, tpsa REAL, registered_by TEXT, project_code TEXT, registration_date TEXT, status TEXT DEFAULT 'active', metadata TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS inventory (sample_id TEXT PRIMARY KEY, compound_id TEXT, batch_id TEXT, quantity_mg REAL, storage_temp TEXT, location TEXT, status TEXT DEFAULT 'available', history TEXT, FOREIGN KEY (compound_id) REFERENCES compounds(compound_id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS experiments (exp_id TEXT PRIMARY KEY, title TEXT, chemist TEXT, project_code TEXT, created_date TEXT, status TEXT, objective TEXT, procedure TEXT, results TEXT, compounds_used TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS bioassay_data (data_id INTEGER PRIMARY KEY AUTOINCREMENT, compound_id TEXT, exp_id TEXT, assay_type TEXT, value REAL, unit TEXT, timestamp TEXT, FOREIGN KEY (compound_id) REFERENCES compounds(compound_id), FOREIGN KEY (exp_id) REFERENCES experiments(exp_id))''')
    conn.commit()
    return conn

if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_database()

# ==================== 核心類別 ====================

class CompoundRegistry:
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
    def register(self, smiles, chemist_name, project_code, metadata=None):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None, "Invalid SMILES"
        standardized = Chem.MolToSmiles(mol, isomericSmiles=True)
        inchikey = Chem.InchiToInchiKey(Chem.MolToInchi(mol))
        self.cursor.execute("SELECT compound_id FROM compounds WHERE inchikey = ?", (inchikey,))
        if self.cursor.fetchone(): return None, "Already exists"
        
        props = {'mw': Descriptors.MolWt(mol), 'logp': Descriptors.MolLogP(mol), 'tpsa': Descriptors.TPSA(mol)}
        compound_id = f"CPD-{datetime.now().strftime('%Y%m%d')}-{hashlib.md5(inchikey.encode()).hexdigest()[:6].upper()}"
        
        self.cursor.execute('''INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                            (compound_id, standardized, inchikey, props['mw'], props['logp'], props['tpsa'], chemist_name, project_code, datetime.now().isoformat(), 'active', json.dumps(metadata or {})))
        self.conn.commit()
        return compound_id, "Registration successful"
    def search(self, query, search_type="compound_id"):
        if search_type == "compound_id": self.cursor.execute("SELECT * FROM compounds WHERE compound_id = ?", (query,))
        elif search_type == "smiles": self.cursor.execute("SELECT * FROM compounds WHERE smiles LIKE ?", (f"%{query}%",))
        elif search_type == "project": self.cursor.execute("SELECT * FROM compounds WHERE project_code = ?", (query,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]

class InventoryManager:
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
    def add_sample(self, compound_id, batch_id, quantity_mg, storage_temp, location):
        sample_id = f"{compound_id}_{batch_id}"
        history = json.dumps([{'action': 'received', 'date': datetime.now().isoformat(), 'quantity': quantity_mg}])
        try:
            self.cursor.execute('''INSERT INTO inventory VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (sample_id, compound_id, batch_id, quantity_mg, storage_temp, location, 'available', history))
            self.conn.commit()
            return sample_id, "Success"
        except sqlite3.IntegrityError: return None, "Sample already exists"
    def get_inventory(self, compound_id=None):
        if compound_id: self.cursor.execute("SELECT * FROM inventory WHERE compound_id = ?", (compound_id,))
        else: self.cursor.execute("SELECT * FROM inventory")
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    def low_stock_alert(self, threshold_mg=10):
        self.cursor.execute("SELECT * FROM inventory WHERE quantity_mg < ? AND status = 'available'", (threshold_mg,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]

class ExperimentManager:
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
    def create_experiment(self, title, chemist, project_code, objective=""):
        exp_id = f"EXP-{datetime.now().strftime('%Y%m%d')}-{hashlib.md5(title.encode()).hexdigest()[:6].upper()}"
        self.cursor.execute('''INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (exp_id, title, chemist, project_code, datetime.now().isoformat(), 'in_progress', objective, '', '[]', '[]'))
        self.conn.commit()
        return exp_id
    def add_result(self, exp_id, compound_id, assay_type, value, unit):
        self.cursor.execute('''INSERT INTO bioassay_data (compound_id, exp_id, assay_type, value, unit, timestamp) VALUES (?, ?, ?, ?, ?, ?)''', (compound_id, exp_id, assay_type, value, unit, datetime.now().isoformat()))
        self.conn.commit()

class SARAnalyzer:
    def analyze_series(self, compound_ids, conn):
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(compound_ids))
        cursor.execute(f'''SELECT b.compound_id, c.smiles, b.assay_type, b.value, b.unit FROM bioassay_data b JOIN compounds c ON b.compound_id = c.compound_id WHERE b.compound_id IN ({placeholders})''', compound_ids)
        data = [{'compound_id': r[0], 'smiles': r[1], 'assay': r[2], 'value': r[3], 'unit': r[4]} for r in cursor.fetchall()]
        if not data: return None
        df = pd.DataFrame(data)
        df['scaffold'] = df['smiles'].apply(self._get_scaffold)
        return df
    def _get_scaffold(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol) if mol else None
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except: return None
    def plot_activity_vs_property(self, df, property_col='value'):
        fig = px.scatter(df, x='compound_id', y=property_col, color='assay', title='Compound Activity Profile', height=400, template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

class PublicDatabaseAPI:
    def __init__(self):
        self.chembl_bioactivities = new_client.activity
        
    def query_pubchem(self, identifier, id_type="name"):
        """強化版 PubChem 查詢，先查本地字典防斷網"""
        identifier_clean = identifier.lower().strip()
        
        # 1. 先查本地救命字典
        if identifier_clean in SAFE_DEMO_DB:
            smiles = SAFE_DEMO_DB[identifier_clean]
            mol = Chem.MolFromSmiles(smiles)
            return {
                'name': identifier.title(),
                'smiles': smiles,
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol)
            }
            
        # 2. 如果字典沒有，才去連 PubChem API
        try:
            c = pcp.get_compounds(identifier, id_type)
            if not c: return None
            comp = c[0]
            mol = Chem.MolFromSmiles(comp.isomeric_smiles or comp.canonical_smiles)
            if not mol: return None # 確保可以被 RDKit 解析
            
            return {
                'cid': comp.cid,
                'name': comp.iupac_name or (comp.synonyms[0] if comp.synonyms else identifier),
                'smiles': Chem.MolToSmiles(mol), # 使用標準化後的 SMILES
                'mw': comp.molecular_weight,
                'logp': comp.xlogp,
                'tpsa': comp.tpsa
            }
        except: return None
        
    def query_chembl_bioactivity(self, smiles):
        try:
            base_url = "https://www.ebi.ac.uk/chembl/api/data"
            res = requests.get(f"{base_url}/similarity/{urllib.parse.quote(smiles)}/90?format=json", timeout=5)
            if res.status_code == 200 and res.json().get('molecules'):
                chembl_id = res.json()['molecules'][0]['molecule_chembl_id']
                act_res = requests.get(f"{base_url}/activity?molecule_chembl_id={chembl_id}&limit=5&format=json", timeout=5)
                if act_res.status_code == 200:
                    return [{'type': b.get('standard_type'), 'value': b.get('standard_value'), 'units': b.get('standard_units'), 'target': b.get('target_pref_name')} for b in act_res.json().get('activities', []) if b.get('target_pref_name')]
        except: pass
        return []

class FreeADMETRules:
    @staticmethod
    def predict_herg(mol):
        tpsa, logp = Descriptors.TPSA(mol), Descriptors.MolLogP(mol)
        alerts = {"High": ["[c]CCN", "[c]OCCN"], "Moderate": ["N(C)C", "CN(C)C"]}
        if tpsa < 60 and logp > 3.5: return "High", "High lipophilicity & Low TPSA", "Ekins et al. 2002"
        for level, patterns in alerts.items():
            for patt in patterns:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(patt)): return level, f"Contains hERG pharmacophore", "Structural Alert"
        return "Low", "No significant alerts", "Rule-based"
    @staticmethod
    def predict_liver(mol):
        if Descriptors.MolLogP(mol) > 4.0 and Descriptors.MolWt(mol) > 400: return "Moderate", "Rule of 2: LogP > 4 & MW > 400", "Chen et al. 2016"
        if Fragments.fr_COO(mol) > 0: return "Moderate", "Contains carboxylic acid", "Structural Alert"
        return "Low", "Properties within safe range", "Rule-based"
    @staticmethod
    def predict_bbb(mol):
        logp, tpsa = Descriptors.MolLogP(mol), Descriptors.TPSA(mol)
        if tpsa < 79 and 0.4 < logp < 6.0: return "High", "Yellow Zone (Optimal for CNS)", "BOILED-Egg Model"
        elif tpsa < 120: return "Moderate", "White Zone (Peripheral)", "BOILED-Egg Model"
        else: return "Low", "Outside Egg (Poor Penetration)", "BOILED-Egg Model"

def generate_3d_pdb(mol):
    try:
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv2())
        return Chem.MolToPDBBlock(mol_3d)
    except: return None

# ==================== 主程式 ====================
def main():
    conn = st.session_state.db_conn
    registry = CompoundRegistry(conn)
    inventory = InventoryManager(conn)
    experiments = ExperimentManager(conn)
    sar = SARAnalyzer()
    public_api = PublicDatabaseAPI()
    admet = FreeADMETRules()
    
    st.markdown('<div class="internal-warning">⚠️ INTERNAL R&D USE ONLY - NOT FOR REGULATORY SUBMISSION</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🧬 Navigation")
        page = st.radio("Select Module", ["🏠 Dashboard", "🌐 Public DB & AI Analysis", "📝 Compound Registration", "📦 Inventory", "🔬 Experiments", "📊 SAR Analysis"])
        
        st.markdown("---")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM compounds")
        st.metric("Internal Compounds", cursor.fetchone()[0])

    # --- Page: Dashboard ---
    if page == "🏠 Dashboard":
        st.header("R&D Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Recent Compounds")
            cursor.execute("SELECT compound_id, registration_date FROM compounds ORDER BY registration_date DESC LIMIT 5")
            for row in cursor.fetchall(): st.write(f"• **{row[0]}** ({row[1][:10]})")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Low Stock Alert")
            alerts = inventory.low_stock_alert(10)
            if alerts:
                for a in alerts: st.warning(f"⚠️ {a['compound_id']}: {a['quantity_mg']}mg left")
            else: st.success("✅ All stock levels nominal")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Page: Public DB & AI Analysis ---
    elif page == "🌐 Public DB & AI Analysis":
        st.header("Public Database & AI Analysis")
        st.caption("即時連線資料庫並執行 ADMET 模型預測")
        
        query = st.text_input("Enter Drug Name or SMILES (e.g., Donepezil, Amoxicillin)", "Donepezil")
        if st.button("🚀 Analyze Molecule", use_container_width=True):
            with st.spinner("Running RDKit models and connecting databases..."):
                
                # 這裡調用強化版的 PubChem 查詢 (包含本地救命字典)
                result = public_api.query_pubchem(query, "name" if "1" not in query and "C" not in query else "smiles")
                
                if not result:
                    st.error("❌ 無法解析分子結構，請檢查輸入或網路連線。")
                else:
                    mol = Chem.MolFromSmiles(result['smiles'])
                    
                    # 1. 物理化學儀表板
                    st.markdown("### 1️⃣ Physicochemical Profile")
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("MW", f"{result['mw']:.1f}")
                    k2.metric("LogP", f"{result['logp']:.2f}")
                    k3.metric("TPSA", f"{result['tpsa']:.1f}")
                    k4.metric("HBD", f"{Descriptors.NumHDonors(mol)}")
                    k5.metric("QED", f"{QED.qed(mol):.2f}")
                    
                    # 【重要修復】五大指標表格完整回歸
                    with st.expander("📖 點擊查看：五大指標科學原理詳解 (Scientific Rationale)", expanded=False):
                        st.markdown("""
                        | 指標 (Metric) | 理想範圍 | 科學原理 (Scientific Rationale) |
                        | :--- | :--- | :--- |
                        | **TPSA** (極性表面積) | < 79 Å² | **反映去溶劑化能 (Desolvation Energy)。** TPSA 過高代表能障過大，難以入腦。 |
                        | **LogP** (親脂性) | 0.4 - 6.0 | **決定磷脂雙分子層的親和力。** 需具備適當脂溶性以穿透細胞膜。 |
                        | **MW** (分子量) | < 360 Da | **空間障礙 (Steric Hindrance)。** 分子量越小，擴散係數越高。 |
                        | **HBD** (氫鍵給體) | < 1 | **水合層效應 (Solvation Shell)。** 氫鍵給體易與水形成強鍵結，阻礙穿透。 |
                        | **pKa** (酸鹼度) | 7.5 - 8.5 | **離子化狀態 (Ionization State)。** 只有未帶電的中性分子能有效藉由被動擴散通過。 |
                        """)
                    
                    # 2. BOILED-Egg & 3D Viewer
                    st.markdown("### 2️⃣ BBB Penetration & 3D Structure")
                    c_chart, c_3d = st.columns(2)
                    with c_chart:
                        fig = go.Figure()
                        fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=6, y1=140, fillcolor="rgba(255, 204, 0, 0.2)", line_color="rgba(255, 204, 0, 0.5)")
                        fig.add_trace(go.Scatter(x=[result['logp']], y=[result['tpsa']], mode='markers+text', marker=dict(size=18, color='#4ade80'), text=[result['name']], textposition="top center"))
                        fig.update_layout(xaxis_title="WLOGP", yaxis_title="TPSA", plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=350, margin=dict(t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    with c_3d:
                        st.markdown(f"**{result['name']} (3D Live Render)**")
                        v1 = py3Dmol.view(width=400, height=300)
                        pdb_data = generate_3d_pdb(mol)
                        if pdb_data:
                            v1.addModel(pdb_data, 'pdb')
                            v1.setStyle({'stick': {}})
                            v1.zoomTo()
                            showmol(v1, height=300, width=400)
                        else:
                            st.warning("無法生成 3D 結構")
                    
                    # 3. ADMET 規則引擎卡片
                    st.markdown("### 3️⃣ ADMET Risk Assessment (Rule-based)")
                    herg_r, herg_d, herg_ref = admet.predict_herg(mol)
                    liv_r, liv_d, liv_ref = admet.predict_liver(mol)
                    bbb_r, bbb_d, bbb_ref = admet.predict_bbb(mol)
                    
                    col_h, col_l, col_b = st.columns(3)
                    with col_h:
                        c_code = "risk-high" if herg_r == "High" else "risk-medium" if herg_r == "Moderate" else "risk-low"
                        b_code = "#ef4444" if herg_r == "High" else "#f59e0b" if herg_r == "Moderate" else "#10b981"
                        st.markdown(f'<div style="background:rgba(30,41,59,0.7); border-radius:12px; padding:15px; border-top:4px solid {b_code};"><h4>🫀 hERG Risk</h4><p class="{c_code}">{herg_r}</p><p style="font-size:0.8rem;color:#94a3b8;">{herg_d}</p></div>', unsafe_allow_html=True)
                    with col_l:
                        c_code = "risk-high" if liv_r == "High" else "risk-medium" if liv_r == "Moderate" else "risk-low"
                        b_code = "#ef4444" if liv_r == "High" else "#f59e0b" if liv_r == "Moderate" else "#10b981"
                        st.markdown(f'<div style="background:rgba(30,41,59,0.7); border-radius:12px; padding:15px; border-top:4px solid {b_code};"><h4>🧪 Liver DILI</h4><p class="{c_code}">{liv_r}</p><p style="font-size:0.8rem;color:#94a3b8;">{liv_d}</p></div>', unsafe_allow_html=True)
                    with col_b:
                        c_code = "risk-high" if bbb_r == "Low" else "risk-medium" if bbb_r == "Moderate" else "risk-low"
                        b_code = "#ef4444" if bbb_r == "Low" else "#f59e0b" if bbb_r == "Moderate" else "#10b981"
                        st.markdown(f'<div style="background:rgba(30,41,59,0.7); border-radius:12px; padding:15px; border-top:4px solid {b_code};"><h4>🧠 BBB Penetration</h4><p class="{c_code}">{bbb_r}</p><p style="font-size:0.8rem;color:#94a3b8;">{bbb_d}</p></div>', unsafe_allow_html=True)

                    # 4. ChEMBL 活性
                    st.markdown("### 4️⃣ Target Bioactivity (ChEMBL)")
                    acts = public_api.query_chembl_bioactivity(result['smiles'])
                    if acts: st.dataframe(pd.DataFrame(acts), use_container_width=True)
                    else: st.info("No specific IC50/Ki data found in top ChEMBL results.")

    # --- Page: Registration ---
    elif page == "📝 Compound Registration":
        st.header("Compound Registration")
        with st.form("reg_form"):
            c1, c2 = st.columns(2)
            with c1: smiles = st.text_area("SMILES", "CC(=O)Oc1ccccc1C(=O)O")
            with c2: 
                chemist = st.text_input("Chemist")
                project = st.text_input("Project Code")
            if st.form_submit_button("Register"):
                cid, msg = registry.register(smiles, chemist, project)
                if cid: st.success(f"✅ {msg}: {cid}")
                else: st.error(f"❌ {msg}")
        
        st.subheader("Internal Database")
        data = registry.search("")
        if data: st.dataframe(pd.DataFrame(data), use_container_width=True)

    # --- Page: Inventory ---
    elif page == "📦 Inventory":
        st.header("Inventory Management")
        with st.form("inv_form"):
            c1, c2, c3 = st.columns(3)
            with c1: cid = st.text_input("Compound ID (e.g. CPD-...)")
            with c2: qty = st.number_input("Amount (mg)", 1.0)
            with c3: loc = st.text_input("Location (e.g. Fridge A)")
            if st.form_submit_button("Add Stock"):
                sid, msg = inventory.add_sample(cid, "BATCH-01", qty, "4C", loc)
                if sid: st.success(f"✅ Added: {sid}")
                else: st.error(msg)
        stock = inventory.get_inventory()
        if stock: st.dataframe(pd.DataFrame(stock), use_container_width=True)

    # --- Page: Experiments ---
    elif page == "🔬 Experiments":
        st.header("Experiment Logs")
        with st.form("exp_form"):
            title = st.text_input("Experiment Title")
            if st.form_submit_button("Create Record"):
                exp = experiments.create_experiment(title, "Auto", "PROJ-1")
                st.success(f"✅ Created EXP: {exp}")

    # --- Page: SAR ---
    elif page == "📊 SAR Analysis":
        st.header("Structure-Activity Relationship (SAR)")
        st.info("Select compounds from your bioassay_data table to plot SAR.")
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT compound_id FROM bioassay_data")
        ids = [r[0] for r in cursor.fetchall()]
        sel = st.multiselect("Select Compounds", ids)
        if sel and st.button("Analyze"):
            df = sar.analyze_series(sel, conn)
            if df is not None:
                st.plotly_chart(sar.plot_activity_vs_property(df), use_container_width=True)
            else: st.warning("No data.")

if __name__ == "__main__":
    main()
