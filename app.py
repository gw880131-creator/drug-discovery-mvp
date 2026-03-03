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
        """查詢庫存
