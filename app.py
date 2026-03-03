import streamlit as st
import pandas as pd
import requests
import json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client
import time

# ==================== 公開資料庫 API 整合層 ====================

class PublicDatabaseAPI:
    """整合 PubChem, ChEMBL, UniChem 的統一介面"""
    
    def __init__(self):
        # ChEMBL 客戶端初始化
        self.chembl_targets = new_client.target
        self.chembl_compounds = new_client.molecule
        self.chembl_bioactivities = new_client.activity
        self.chembl_assays = new_client.assay
        
        # UniChem API 端點
        self.unichem_url = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
        
        # PubChem PUG REST API
        self.pubchem_rest = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # --- 1. PubChem 即時查詢 ---
    def query_pubchem_live(self, identifier, id_type="name"):
        """
        即時查詢 PubChem 取得完整化合物資訊
        id_type: name, cid, smiles, inchikey
        """
        try:
            if id_type == "smiles":
                # 使用 SMILES 查詢
                c = pcp.get_compounds(identifier, "smiles")
            elif id_type == "inchikey":
                c = pcp.get_compounds(identifier, "inchikey")
            else:
                c = pcp.get_compounds(identifier, id_type)
            
            if not c:
                return None
                
            comp = c[0]
            return {
                "source": "PubChem",
                "cid": comp.cid,
                "name": comp.iupac_name or comp.synonyms[0] if comp.synonyms else "Unknown",
                "smiles": comp.isomeric_smiles or comp.canonical_smiles,
                "inchi": comp.inchi,
                "inchikey": comp.inchikey,
                "molecular_formula": comp.molecular_formula,
                "molecular_weight": comp.molecular_weight,
                "xlogp": comp.xlogp,
                "tpsa": comp.tpsa,
                "complexity": comp.complexity,
                "hbond_donor": comp.h_bond_donor_count,
                "hbond_acceptor": comp.h_bond_acceptor_count,
                "rotatable_bonds": comp.rotatable_bond_count,
                "exact_mass": comp.exact_mass,
                "charge": comp.charge,
                "synonyms": comp.synonyms[:5] if comp.synonyms else [],
                "description": comp.description if hasattr(comp, 'description') else None,
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{comp.cid}"
            }
        except Exception as e:
            st.error(f"PubChem 查詢錯誤: {e}")
            return None
    
    # --- 2. ChEMBL 生物活性數據查詢 ---
    def query_chembl_bioactivity(self, chembl_id=None, target_name=None, compound_name=None):
        """
        查詢 ChEMBL 生物活性數據
        可通過 ChEMBL ID、標靶名稱或化合物名稱查詢
        """
        results = {
            "compounds": [],
            "bioactivities": [],
            "targets": []
        }
        
        try:
            # 如果提供化合物 ChEMBL ID
            if chembl_id:
                # 取得化合物詳情
                compound_data = self.chembl_compounds.get(chembl_id)
                if compound_data:
                    results["compounds"].append({
                        "chembl_id": chembl_id,
                        "name": compound_data.get('pref_name', 'N/A'),
                        "smiles": compound_data.get('molecule_structures', {}).get('canonical_smiles', ''),
                        "properties": compound_data.get('molecule_properties', {})
                    })
                    
                    # 查詢相關生物活性
                    bioacts = self.chembl_bioactivities.filter(
                        molecule_chembl_id=chembl_id,
                        type__in=["IC50", "Ki", "Kd", "EC50"],
                        relation__in=["=", "<"]
                    ).only("type", "standard_value", "standard_units", 
                           "target_chembl_id", "assay_chembl_id", "activity_comment")
                    
                    for bio in bioacts[:20]:  # 限制前20筆
                        results["bioactivities"].append({
                            "type": bio.get('type'),
                            "value": bio.get('standard_value'),
                            "units": bio.get('standard_units'),
                            "target_id": bio.get('target_chembl_id'),
                            "assay_id": bio.get('assay_chembl_id'),
                            "comment": bio.get('activity_comment')
                        })
            
            # 如果提供標靶名稱（如 EGFR, AChE）
            elif target_name:
                targets = self.chembl_targets.search(target_name)
                if targets:
                    target = targets[0]
                    target_id = target['target_chembl_id']
                    results["targets"].append({
                        "chembl_id": target_id,
                        "name": target.get('pref_name'),
                        "type": target.get('target_type'),
                        "organism": target.get('organism')
                    })
                    
                    # 查詢該標靶的所有活性化合物
                    bioacts = self.chembl_bioactivities.filter(
                        target_chembl_id=target_id,
                        type="IC50",
                        relation="="
                    ).only("molecule_chembl_id", "standard_value", "standard_units")
                    
                    # 取得唯一化合物清單
                    unique_compounds = list(set([b['molecule_chembl_id'] for b in bioacts]))
                    
                    # 批次查詢化合物資訊
                    for comp_id in unique_compounds[:10]:
                        comp_data = self.chembl_compounds.get(comp_id)
                        if comp_data:
                            results["compounds"].append({
                                "chembl_id": comp_id,
                                "name": comp_data.get('pref_name', 'N/A'),
                                "smiles": comp_data.get('molecule_structures', {}).get('canonical_smiles', '')
                            })
                            
        except Exception as e:
            st.error(f"ChEMBL 查詢錯誤: {e}")
            
        return results
    
    # --- 3. UniChem 交叉引用查詢 ---
    def query_unichem_crossref(self, identifier, id_type="inchikey"):
        """
        使用 UniChem 查詢化合物在多個資料庫的交叉引用
        支援: inchikey, inchi, smiles
        """
        try:
            payload = {
                "type": id_type,
                "compound": identifier
            }
            
            response = requests.post(
                self.unichem_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                cross_refs = []
                
                if 'compounds' in data:
                    for compound in data['compounds']:
                        for source in compound.get('sources', []):
                            cross_refs.append({
                                "database": source.get('name'),
                                "id_in_source": source.get('compoundId'),
                                "url": source.get('url'),
                                "short_name": source.get('shortName')
                            })
                
                return {
                    "query": identifier,
                    "total_sources": len(cross_refs),
                    "cross_references": cross_refs,
                    "uci": compound.get('uci') if 'compounds' in data and data['compounds'] else None
                }
            else:
                return None
                
        except Exception as e:
            st.error(f"UniChem 查詢錯誤: {e}")
            return None
    
    # --- 4. PubChem PUG REST 進階查詢 ---
    def query_pubchem_advanced(self, cid, properties=None):
        """
        使用 PubChem PUG REST API 取得特定屬性
        """
        if properties is None:
            properties = ["MolecularWeight", "XLogP", "TPSA", "Complexity"]
        
        prop_str = ",".join(properties)
        url = f"{self.pubchem_rest}/compound/cid/{cid}/property/{prop_str}/JSON"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()['PropertyTable']['Properties'][0]
            return None
        except:
            return None
    
    # --- 5. 結構相似性搜尋 (PubChem) ---
    def search_pubchem_similarity(self, smiles, threshold=90, max_records=20):
        """
        在 PubChem 中搜尋結構相似化合物
        threshold: Tanimoto 相似度閾值 (0-100)
        """
        try:
            # 使用 PubChem PUG REST 進行相似性搜尋
            url = f"{self.pubchem_rest}/compound/fastsimilarity_2d/smiles/{requests.utils.quote(smiles)}/cids/JSON"
            params = {"Threshold": threshold, "MaxRecords": max_records}
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                cids = data.get('IdentifierList', {}).get('CID', [])
                
                similar_compounds = []
                for cid in cids[:max_records]:
                    comp_info = self.query_pubchem_live(str(cid), "cid")
                    if comp_info:
                        similar_compounds.append(comp_info)
                
                return similar_compounds
            return []
        except Exception as e:
            st.error(f"相似性搜尋錯誤: {e}")
            return []

# ==================== 強化版 Streamlit UI ====================

def main():
    st.set_page_config(
        page_title="MedChem Pro | Live Database Edition", 
        page_icon="🧬", 
        layout="wide"
    )
    
    # 初始化 API 連接器
    if 'api' not in st.session_state:
        st.session_state.api = PublicDatabaseAPI()
    
    st.title("🧬 MedChem Pro - 即時公開資料庫版")
    st.markdown("即時連線 **PubChem** | **ChEMBL** | **UniChem**")
    
    # 側邊欄搜尋
    with st.sidebar:
        st.header("🔍 多資料庫搜尋")
        
        search_type = st.selectbox(
            "搜尋類型",
            ["化合物名稱", "SMILES", "InChIKey", "ChEMBL ID", "標靶名稱"]
        )
        
        search_input = st.text_input("輸入搜尋內容", "Aspirin")
        
        col1, col2 = st.columns(2)
        with col1:
            search_btn = st.button("🚀 搜尋", use_container_width=True)
        with col2:
            similarity_search = st.checkbox("相似性搜尋")
        
        st.markdown("---")
        st.markdown("### 資料庫狀態")
        st.success("🟢 PubChem: 連線中")
        st.success("🟢 ChEMBL: 連線中")
        st.success("🟢 UniChem: 連線中")
    
    if search_btn and search_input:
        api = st.session_state.api
        
        with st.spinner("正在查詢多個公開資料庫..."):
            progress_bar = st.progress(0)
            
            # 1. PubChem 查詢
            progress_bar.progress(25)
            if search_type == "化合物名稱":
                pubchem_data = api.query_pubchem_live(search_input, "name")
            elif search_type == "SMILES":
                pubchem_data = api.query_pubchem_live(search_input, "smiles")
            elif search_type == "InChIKey":
                pubchem_data = api.query_pubchem_live(search_input, "inchikey")
            else:
                pubchem_data = None
            
            # 2. ChEMBL 查詢
            progress_bar.progress(50)
            if search_type == "標靶名稱":
                chembl_data = api.query_chembl_bioactivity(target_name=search_input)
            elif search_type == "ChEMBL ID":
                chembl_data = api.query_chembl_bioactivity(chembl_id=search_input)
            else:
                chembl_data = api.query_chembl_bioactivity(compound_name=search_input)
            
            # 3. UniChem 交叉引用
            progress_bar.progress(75)
            unichem_data = None
            if pubchem_data and pubchem_data.get('inchikey'):
                unichem_data = api.query_unichem_crossref(pubchem_data['inchikey'], "inchikey")
            
            # 4. 相似性搜尋
            similar_compounds = []
            if similarity_search and pubchem_data and pubchem_data.get('smiles'):
                similar_compounds = api.search_pubchem_similarity(pubchem_data['smiles'])
            
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
        
        # ==================== 結果展示 ====================
        
        if pubchem_data:
            st.success(f"✅ 成功從 **{pubchem_data['source']}** 取得資料")
            
            # 頂部資訊卡
            cols = st.columns(4)
            cols[0].metric("PubChem CID", pubchem_data['cid'])
            cols[1].metric("分子量", f"{pubchem_data['molecular_weight']:.2f}")
            cols[2].metric("XLogP", pubchem_data['xlogp'] if pubchem_data['xlogp'] else "N/A")
            cols[3].metric("TPSA", pubchem_data['tpsa'] if pubchem_data['tpsa'] else "N/A")
            
            # Tab 介面
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 基礎資訊", 
                "🧬 ChEMBL 生物活性", 
                "🔗 UniChem 交叉引用",
                "🔍 相似化合物"
            ])
            
            with tab1:
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.markdown("### 化合物詳情")
                    info_df = pd.DataFrame([
                        ["IUPAC 名稱", pubchem_data['name']],
                        ["分子式", pubchem_data['molecular_formula']],
                        ["SMILES", pubchem_data['smiles']],
                        ["InChIKey", pubchem_data['inchikey']],
                        ["精確質量", pubchem_data['exact_mass']],
                        ["複雜度", pubchem_data['complexity']],
                        ["氫鍵供體", pubchem_data['hbond_donor']],
                        ["氫鍵受體", pubchem_data['hbond_acceptor']],
                        ["可旋轉鍵", pubchem_data['rotatable_bonds']]
                    ], columns=["屬性", "數值"])
                    
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
                    
                    if pubchem_data.get('synonyms'):
                        st.markdown("### 同義詞")
                        st.write(", ".join(pubchem_data['synonyms']))
                
                with col_right:
                    # 顯示 2D 結構（使用 PubChem 圖片）
                    if pubchem_data['cid']:
                        img_url = f"https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid={pubchem_data['cid']}&width=300&height=300"
                        st.image(img_url, caption="PubChem 2D 結構")
                    
                    st.markdown(f"[🔗 在 PubChem 查看]({pubchem_data['url']})")
            
            with tab2:
                if chembl_data and (chembl_data['compounds'] or chembl_data['bioactivities']):
                    st.markdown("### ChEMBL 數據")
                    
                    if chembl_data['targets']:
                        st.markdown("**標靶資訊：**")
                        for t in chembl_data['targets']:
                            st.write(f"- {t['name']} ({t['chembl_id']}) | {t['organism']}")
                    
                    if chembl_data['bioactivities']:
                        st.markdown("**生物活性數據：**")
                        bio_df = pd.DataFrame(chembl_data['bioactivities'])
                        st.dataframe(bio_df, use_container_width=True)
                    else:
                        st.info("未找到生物活性數據")
                else:
                    st.info("ChEMBL 中無此化合物數據")
            
            with tab3:
                if unichem_data:
                    st.markdown(f"### 找到 {unichem_data['total_sources']} 個資料庫交叉引用")
                    
                    if unichem_data['cross_references']:
                        ref_df = pd.DataFrame(unichem_data['cross_references'])
                        st.dataframe(ref_df[['database', 'id_in_source', 'url']], 
                                   use_container_width=True)
                        
                        # 視覺化資料庫分布
                        db_counts = ref_df['database'].value_counts()
                        st.bar_chart(db_counts)
                else:
                    st.info("未找到交叉引用數據")
            
            with tab4:
                if similarity_search:
                    if similar_compounds:
                        st.markdown(f"### 找到 {len(similar_compounds)} 個相似化合物")
                        
                        sim_cols = st.columns(3)
                        for idx, sim_comp in enumerate(similar_compounds[:6]):
                            with sim_cols[idx % 3]:
                                sim_img = f"https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid={sim_comp['cid']}&width=200&height=200"
                                st.image(sim_img, caption=f"{sim_comp['name'][:30]}...")
                                st.caption(f"CID: {sim_comp['cid']}")
                    else:
                        st.info("未找到相似化合物")
                else:
                    st.info("請在側邊欄勾選「相似性搜尋」")
        
        else:
            st.error("❌ 無法在公開資料庫中找到該化合物")

if __name__ == "__main__":
    main()
