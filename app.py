import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import sqlite3
import json
import hashlib
from datetime import datetime
import requests
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client
import plotly.express as px
import plotly.graph_objects as go

# ==================== 頁面設定 ====================
st.set_page_config(
    page_title="MedChem Pro | Enterprise R&D Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS 樣式 ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
        color: #e2e8f0; 
        font-family: 'Inter', sans-serif; 
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px); 
        border: 1px solid rgba(148, 163, 184, 0.1); 
        border-radius: 16px; 
        padding: 20px;
        margin: 10px 0;
    }
    
    .internal-warning {
        background-color: rgba(245, 158, 11, 0.15); 
        border: 1px solid #f59e0b; 
        color: #fbbf24; 
        padding: 15px; 
        border-radius: 8px; 
        font-size: 0.9rem; 
        text-align: center;
        margin-bottom: 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 資料庫初始化 ====================
def init_database():
    """初始化 SQLite 資料庫"""
    conn = sqlite3.connect('medchem_pro.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # 化合物註冊表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS compounds (
            compound_id TEXT PRIMARY KEY,
            smiles TEXT UNIQUE,
            inchikey TEXT UNIQUE,
            mol_weight REAL,
            logp REAL,
            tpsa REAL,
            registered_by TEXT,
            project_code TEXT,
            registration_date TEXT,
            status TEXT DEFAULT 'active',
            metadata TEXT
        )
    ''')
    
    # 庫存表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            sample_id TEXT PRIMARY KEY,
            compound_id TEXT,
            batch_id TEXT,
            quantity_mg REAL,
            storage_temp TEXT,
            location TEXT,
            status TEXT DEFAULT 'available',
            history TEXT,
            FOREIGN KEY (compound_id) REFERENCES compounds(compound_id)
        )
    ''')
    
    # 實驗記錄表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            exp_id TEXT PRIMARY KEY,
            title TEXT,
            chemist TEXT,
            project_code TEXT,
            created_date TEXT,
            status TEXT,
            objective TEXT,
            procedure TEXT,
            results TEXT,
            compounds_used TEXT
        )
    ''')
    
    # SAR 數據表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bioassay_data (
            data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            compound_id TEXT,
            exp_id TEXT,
            assay_type TEXT,
            value REAL,
            unit TEXT,
            timestamp TEXT,
            FOREIGN KEY (compound_id) REFERENCES compounds(compound_id),
            FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
        )
    ''')
    
    conn.commit()
    return conn

# 初始化資料庫連接
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_database()

# ==================== 核心類別 ====================

class CompoundRegistry:
    """化合物註冊系統"""
    
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
    
    def register(self, smiles, chemist_name, project_code, metadata=None):
        """註冊新化合物"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None, "Invalid SMILES"
        
        # 標準化
        standardized = Chem.MolToSmiles(mol, isomericSmiles=True)
        inchikey = Chem.InchiToInchiKey(Chem.MolToInchi(mol))
        
        # 檢查重複
        self.cursor.execute("SELECT compound_id FROM compounds WHERE inchikey = ?", (inchikey,))
        existing = self.cursor.fetchone()
        if existing:
            return existing[0], "Already exists"
        
        # 計算屬性
        props = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol)
        }
        
        # 生成 ID
        compound_id = f"CPD-{datetime.now().strftime('%Y%m%d')}-{hashlib.md5(inchikey.encode()).hexdigest()[:6].upper()}"
        
        # 存入資料庫
        self.cursor.execute('''
            INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            compound_id, standardized, inchikey, props['mw'], props['logp'], 
            props['tpsa'], chemist_name, project_code, 
            datetime.now().isoformat(), 'active', json.dumps(metadata or {})
        ))
        self.conn.commit()
        
        return compound_id, "Registration successful"
    
    def search(self, query, search_type="compound_id"):
        """搜尋化合物"""
        if search_type == "compound_id":
            self.cursor.execute("SELECT * FROM compounds WHERE compound_id = ?", (query,))
        elif search_type == "smiles":
            self.cursor.execute("SELECT * FROM compounds WHERE smiles LIKE ?", (f"%{query}%",))
        elif search_type == "project":
            self.cursor.execute("SELECT * FROM compounds WHERE project_code = ?", (query,))
        
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def substructure_search(self, query_smiles):
        """子結構搜尋"""
        query_mol = Chem.MolFromSmiles(query_smiles)
        if not query_mol:
            return []
        
        self.cursor.execute("SELECT compound_id, smiles FROM compounds WHERE status = 'active'")
        matches = []
        for cid, smiles in self.cursor.fetchall():
            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.HasSubstructMatch(query_mol):
                matches.append(cid)
        return matches

class InventoryManager:
    """庫存管理系統"""
    
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
    
    def add_sample(self, compound_id, batch_id, quantity_mg, storage_temp, location):
        """新增樣品"""
        sample_id = f"{compound_id}_{batch_id}"
        
        history = json.dumps([{
            'action': 'received',
            'date': datetime.now().isoformat(),
            'quantity': quantity_mg
        }])
        
        try:
            self.cursor.execute('''
                INSERT INTO inventory VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (sample_id, compound_id, batch_id, quantity_mg, storage_temp, location, 'available', history))
            self.conn.commit()
            return sample_id, "Success"
        except sqlite3.IntegrityError:
            return None, "Sample already exists"
    
    def checkout(self, sample_id, amount_mg, user, experiment_id):
        """領用樣品"""
        self.cursor.execute("SELECT quantity_mg, history FROM inventory WHERE sample_id = ?", (sample_id,))
        row = self.cursor.fetchone()
        
        if not row:
            return False, "Sample not found"
        
        current_qty, history_str = row
        if current_qty < amount_mg:
            return False, "Insufficient quantity"
        
        new_qty = current_qty - amount_mg
        history = json.loads(history_str)
        history.append({
            'action': 'checkout',
            'date': datetime.now().isoformat(),
            'quantity': -amount_mg,
            'user': user,
            'experiment': experiment_id
        })
        
        status = 'depleted' if new_qty == 0 else 'available'
        
        self.cursor.execute('''
            UPDATE inventory SET quantity_mg = ?, history = ?, status = ? WHERE sample_id = ?
        ''', (new_qty, json.dumps(history), status, sample_id))
        self.conn.commit()
        
        return True, "Checkout successful"
    
    def get_inventory(self, compound_id=None):
        """查詢庫存"""
        if compound_id:
            self.cursor.execute("SELECT * FROM inventory WHERE compound_id = ?", (compound_id,))
        else:
            self.cursor.execute("SELECT * FROM inventory")
        
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def low_stock_alert(self, threshold_mg=10):
        """低庫存警示"""
        self.cursor.execute("SELECT * FROM inventory WHERE quantity_mg < ? AND status = 'available'", (threshold_mg,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]

class ExperimentManager:
    """實驗記錄管理"""
    
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
    
    def create_experiment(self, title, chemist, project_code, objective=""):
        """建立新實驗"""
        exp_id = f"EXP-{datetime.now().strftime('%Y%m%d')}-{hashlib.md5(title.encode()).hexdigest()[:6].upper()}"
        
        self.cursor.execute('''
            INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            exp_id, title, chemist, project_code, datetime.now().isoformat(),
            'in_progress', objective, '', '[]', '[]'
        ))
        self.conn.commit()
        return exp_id
    
    def add_result(self, exp_id, compound_id, assay_type, value, unit):
        """新增實驗結果"""
        self.cursor.execute('''
            INSERT INTO bioassay_data (compound_id, exp_id, assay_type, value, unit, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (compound_id, exp_id, assay_type, value, unit, datetime.now().isoformat()))
        self.conn.commit()
    
    def get_experiment(self, exp_id):
        """取得實驗詳情"""
        self.cursor.execute("SELECT * FROM experiments WHERE exp_id = ?", (exp_id,))
        row = self.cursor.fetchone()
        if row:
            columns = [desc[0] for desc in self.cursor.description]
            return dict(zip(columns, row))
        return None

class SARAnalyzer:
    """SAR 分析工具"""
    
    def analyze_series(self, compound_ids, conn):
        """分析化合物系列"""
        cursor = conn.cursor()
        
        # 取得所有數據
        placeholders = ','.join('?' * len(compound_ids))
        cursor.execute(f'''
            SELECT b.compound_id, c.smiles, b.assay_type, b.value, b.unit
            FROM bioassay_data b
            JOIN compounds c ON b.compound_id = c.compound_id
            WHERE b.compound_id IN ({placeholders})
        ''', compound_ids)
        
        data = []
        for row in cursor.fetchall():
            data.append({
                'compound_id': row[0],
                'smiles': row[1],
                'assay': row[2],
                'value': row[3],
                'unit': row[4]
            })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # 提取骨架
        df['scaffold'] = df['smiles'].apply(self._get_scaffold)
        
        # 計算 R-group
        df['r_groups'] = df.apply(lambda x: self._get_r_groups(x['smiles'], x['scaffold']), axis=1)
        
        return df
    
    def _get_scaffold(self, smiles):
        """取得分子骨架"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            return None
    
    def _get_r_groups(self, smiles, scaffold_smiles):
        """識別 R-group（簡化版）"""
        if not scaffold_smiles:
            return []
        # 這裡可以加入更複雜的 R-group 分解邏輯
        return []
    
    def plot_activity_vs_property(self, df, property_col='value'):
        """繪製活性圖"""
        fig = px.scatter(
            df, 
            x='compound_id', 
            y=property_col,
            color='assay',
            title='Compound Activity Profile',
            height=400
        )
        return fig

# ==================== 公開資料庫 API ====================

class PublicDatabaseAPI:
    """整合 PubChem, ChEMBL"""
    
    def __init__(self):
        self.chembl_targets = new_client.target
        self.chembl_compounds = new_client.molecule
        self.chembl_bioactivities = new_client.activity
    
    def query_pubchem(self, identifier, id_type="name"):
        """查詢 PubChem"""
        try:
            if id_type == "smiles":
                c = pcp.get_compounds(identifier, "smiles")
            elif id_type == "inchikey":
                c = pcp.get_compounds(identifier, "inchikey")
            else:
                c = pcp.get_compounds(identifier, id_type)
            
            if not c:
                return None
            
            comp = c[0]
            return {
                'cid': comp.cid,
                'name': comp.iupac_name or (comp.synonyms[0] if comp.synonyms else "Unknown"),
                'smiles': comp.isomeric_smiles or comp.canonical_smiles,
                'inchikey': comp.inchikey,
                'mw': comp.molecular_weight,
                'logp': comp.xlogp,
                'tpsa': comp.tpsa,
                'synonyms': comp.synonyms[:5] if comp.synonyms else []
            }
        except Exception as e:
            st.error(f"PubChem error: {e}")
            return None
    
    def query_chembl_bioactivity(self, chembl_id):
        """查詢 ChEMBL 生物活性"""
        try:
            bioacts = self.chembl_bioactivities.filter(
                molecule_chembl_id=chembl_id,
                type__in=["IC50", "Ki", "Kd", "EC50"]
            ).only("type", "standard_value", "standard_units", "target_chembl_id")[:20]
            
            return [{
                'type': b.get('type'),
                'value': b.get('standard_value'),
                'units': b.get('standard_units'),
                'target': b.get('target_chembl_id')
            } for b in bioacts]
        except Exception as e:
            st.error(f"ChEMBL error: {e}")
            return []

# ==================== ADMET 規則引擎 ====================

class FreeADMETRules:
    """免費 ADMET 預測規則"""
    
    @staticmethod
    def predict_herg(mol):
        """hERG 心臟毒性預測"""
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        
        alerts = {
            "High": ["[c]CCN", "[c]OCCN"],
            "Moderate": ["N(C)C", "CN(C)C"]
        }
        
        if tpsa < 60 and logp > 3.5:
            return "High", "High lipophilicity & Low TPSA", "Ekins et al. 2002"
        
        for level, patterns in alerts.items():
            for patt in patterns:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(patt)):
                    return level, f"Contains hERG pharmacophore ({patt})", "Structural Alert"
        
        return "Low", "No significant alerts", "Rule-based"
    
    @staticmethod
    def predict_liver(mol):
        """肝臟毒性預測"""
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        if logp > 4.0 and mw > 400:
            return "Moderate", "Rule of 2: LogP > 4 & MW > 400", "Chen et al. 2016"
        
        from rdkit.Chem import Fragments
        if Fragments.fr_COO(mol) > 0:
            return "Moderate", "Contains carboxylic acid", "Structural Alert"
        
        return "Low", "Properties within safe range", "Rule-based"
    
    @staticmethod
    def predict_bbb(mol):
        """血腦屏障穿透性"""
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        if tpsa < 79 and 0.4 < logp < 6.0:
            return "High", "Yellow Zone (Optimal for CNS)", "BOILED-Egg Model"
        elif tpsa < 120:
            return "Moderate", "White Zone (Peripheral)", "BOILED-Egg Model"
        else:
            return "Low", "Outside Egg (Poor Penetration)", "BOILED-Egg Model"

# ==================== 主程式 ====================

def main():
    # 初始化管理器
    conn = st.session_state.db_conn
    registry = CompoundRegistry(conn)
    inventory = InventoryManager(conn)
    experiments = ExperimentManager(conn)
    sar = SARAnalyzer()
    public_api = PublicDatabaseAPI()
    admet = FreeADMETRules()
    
    # 頁面標題
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>🧬 MedChem Pro <span style="color: #3b82f6;">Enterprise</span></h1>
        <p>Integrated R&D Platform | Compound Registry | Inventory | SAR Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="internal-warning">⚠️ INTERNAL R&D USE ONLY - NOT FOR REGULATORY SUBMISSION</div>', unsafe_allow_html=True)
    
    # 側邊欄導航
    with st.sidebar:
        st.header("🔍 Navigation")
        
        page = st.radio(
            "Select Module",
            ["🏠 Dashboard", "📝 Compound Registration", "📦 Inventory", "🔬 Experiments", "📊 SAR Analysis", "🌐 Public Database"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # 顯示統計
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM compounds")
        compound_count = cursor.fetchone()[0]
        st.metric("Total Compounds", compound_count)
        
        cursor.execute("SELECT COUNT(*) FROM inventory WHERE status = 'available'")
        sample_count = cursor.fetchone()[0]
        st.metric("Available Samples", sample_count)
    
    # ==================== Dashboard ====================
    if page == "🏠 Dashboard":
        st.header("Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Recent Compounds")
            cursor.execute("SELECT compound_id, registration_date FROM compounds ORDER BY registration_date DESC LIMIT 5")
            for row in cursor.fetchall():
                st.write(f"• {row[0]} ({row[1][:10]})")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Low Stock Alert")
            alerts = inventory.low_stock_alert(10)
            if alerts:
                for alert in alerts:
                    st.warning(f"⚠️ {alert['compound_id']}: {alert['quantity_mg']}mg")
            else:
                st.success("No low stock items")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Active Experiments")
            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'in_progress'")
            exp_count = cursor.fetchone()[0]
            st.metric("In Progress", exp_count)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== Compound Registration ====================
    elif page == "📝 Compound Registration":
        st.header("Compound Registration")
        
        tab1, tab2 = st.tabs(["Register New", "Search"])
        
        with tab1:
            with st.form("registration_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    smiles = st.text_area("SMILES", height=100)
                    chemist = st.text_input("Chemist Name")
                
                with col2:
                    project = st.text_input("Project Code")
                    notes = st.text_area("Notes", height=100)
                
                submitted = st.form_submit_button("Register Compound", use_container_width=True)
                
                if submitted and smiles:
                    with st.spinner("Registering..."):
                        compound_id, msg = registry.register(smiles, chemist, project, {'notes': notes})
                        
                        if compound_id:
                            st.success(f"✅ Registered: {compound_id}")
                            
                            # 顯示分子預覽
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                img = Draw.MolToImage(mol, size=(300, 300))
                                st.image(img, caption="2D Structure")
                        else:
                            st.error(f"❌ {msg}")
        
        with tab2:
            search_col1, search_col2 = st.columns([2, 1])
            with search_col1:
                search_query = st.text_input("Search by ID, SMILES, or Project")
            with search_col2:
                search_type = st.selectbox("Type", ["compound_id", "smiles", "project"])
            
            if st.button("Search"):
                results = registry.search(search_query, search_type)
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else:
                    st.info("No compounds found")
    
    # ==================== Inventory ====================
    elif page == "📦 Inventory":
        st.header("Inventory Management")
        
        tab1, tab2, tab3 = st.tabs(["Add Sample", "Checkout", "View Stock"])
        
        with tab1:
            with st.form("add_sample_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    compound_id = st.text_input("Compound ID")
                    batch_id = st.text_input("Batch ID")
                    quantity = st.number_input("Quantity (mg)", min_value=0.1, value=10.0)
                
                with col2:
                    storage_temp = st.selectbox("Storage Temp", ["RT", "4C", "-20C", "-80C"])
                    location = st.text_input("Location (e.g., A101-B3)")
                
                if st.form_submit_button("Add to Inventory"):
                    sample_id, msg = inventory.add_sample(compound_id, batch_id, quantity, storage_temp, location)
                    if sample_id:
                        st.success(f"✅ Sample added: {sample_id}")
                    else:
                        st.error(f"❌ {msg}")
        
        with tab2:
            with st.form("checkout_form"):
                sample_id = st.text_input("Sample ID")
                amount = st.number_input("Amount to checkout (mg)", min_value=0.1)
                user = st.text_input("User Name")
                exp_id = st.text_input("Experiment ID")
                
                if st.form_submit_button("Checkout"):
                    success, msg = inventory.checkout(sample_id, amount, user, exp_id)
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
        
        with tab3:
            compound_filter = st.text_input("Filter by Compound ID (optional)")
            stock_data = inventory.get_inventory(compound_filter if compound_filter else None)
            
            if stock_data:
                df = pd.DataFrame(stock_data)
                st.dataframe(df, use_container_width=True)
                
                # 庫存分布圖
                st.subheader("Storage Distribution")
                temp_dist = df['storage_temp'].value_counts()
                st.bar_chart(temp_dist)
            else:
                st.info("No inventory data")
    
    # ==================== Experiments ====================
    elif page == "🔬 Experiments":
        st.header("Experiment Management")
        
        tab1, tab2 = st.tabs(["New Experiment", "Add Results"])
        
        with tab1:
            with st.form("exp_form"):
                title = st.text_input("Experiment Title")
                chemist = st.text_input("Chemist")
                project = st.text_input("Project Code")
                objective = st.text_area("Objective")
                
                if st.form_submit_button("Create Experiment"):
                    exp_id = experiments.create_experiment(title, chemist, project, objective)
                    st.success(f"✅ Created: {exp_id}")
        
        with tab2:
            with st.form("result_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    exp_id = st.text_input("Experiment ID")
                    compound_id = st.text_input("Compound ID")
                
                with col2:
                    assay_type = st.selectbox("Assay Type", ["IC50", "Ki", "Kd", "EC50", "MIC", "Other"])
                    value = st.number_input("Value", format="%.2e")
                    unit = st.text_input("Unit", "nM")
                
                if st.form_submit_button("Add Result"):
                    experiments.add_result(exp_id, compound_id, assay_type, value, unit)
                    st.success("✅ Result added")
    
    # ==================== SAR Analysis ====================
    elif page == "📊 SAR Analysis":
        st.header("SAR Analysis")
        
        # 選擇化合物系列
        cursor.execute("SELECT DISTINCT compound_id FROM bioassay_data")
        available_ids = [row[0] for row in cursor.fetchall()]
        
        selected_ids = st.multiselect("Select Compounds for Analysis", available_ids)
        
        if selected_ids and st.button("Analyze"):
            with st.spinner("Analyzing..."):
                df = sar.analyze_series(selected_ids, conn)
                
                if df is not None:
                    st.subheader("Activity Data")
                    st.dataframe(df, use_container_width=True)
                    
                    # 視覺化
                    fig = sar.plot_activity_vs_property(df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 骨架分析
                    if 'scaffold' in df.columns:
                        st.subheader("Scaffold Distribution")
                        scaffold_counts = df['scaffold'].value_counts()
                        st.write(scaffold_counts)
                else:
                    st.info("No bioactivity data found for selected compounds")
    
    # ==================== Public Database ====================
    elif page == "🌐 Public Database":
        st.header("Public Database Query")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_input("Enter Name, SMILES, or InChIKey")
        with col2:
            query_type = st.selectbox("Query Type", ["name", "smiles", "inchikey"])
        
        if st.button("Query PubChem"):
            with st.spinner("Querying..."):
                result = public_api.query_pubchem(query, query_type)
                
                if result:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.subheader("Compound Info")
                        info_df = pd.DataFrame([
                            ["CID", result['cid']],
                            ["Name", result['name']],
                            ["SMILES", result['smiles']],
                            ["MW", result['mw']],
                            ["LogP", result['logp']],
                            ["TPSA", result['tpsa']]
                        ], columns=["Property", "Value"])
                        st.dataframe(info_df, use_container_width=True, hide_index=True)
                    
                    with col_b:
                        # 顯示結構
                        img_url = f"https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid={result['cid']}&width=300&height=300"
                        st.image(img_url, caption="2D Structure")
                    
                    # ADMET 預測
                    st.subheader("ADMET Prediction")
                    mol = Chem.MolFromSmiles(result['smiles'])
                    if mol:
                        col_h, col_l, col_b = st.columns(3)
                        
                        with col_h:
                            risk, desc, ref = admet.predict_herg(mol)
                            color = "red" if risk == "High" else "orange" if risk == "Moderate" else "green"
                            st.markdown(f"**hERG Risk:** :{color}[{risk}]")
                            st.caption(desc)
                        
                        with col_l:
                            risk, desc, ref = admet.predict_liver(mol)
                            color = "red" if risk == "High" else "orange" if risk == "Moderate" else "green"
                            st.markdown(f"**Liver Risk:** :{color}[{risk}]")
                            st.caption(desc)
                        
                        with col_b:
                            risk, desc, ref = admet.predict_bbb(mol)
                            color = "red" if risk == "High" else "orange" if risk == "Moderate" else "green"
                            st.markdown(f"**BBB:** :{color}[{risk}]")
                            st.caption(desc)
                else:
                    st.error("Compound not found")

if __name__ == "__main__":
    main()
