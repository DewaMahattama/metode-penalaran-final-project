import streamlit as st
import pandas as pd
import os
from app.cbr import CarCBREngine

db_path = 'dataset/used_car_dataset.csv'
KURS_INR_TO_IDR = 195

st.set_page_config(page_title="Kelompok 5 - CBR", layout="wide")

def format_to_idr(rupee_val):
    idr_val = rupee_val * KURS_INR_TO_IDR
    return f"Rp {idr_val:,.0f}".replace(",", ".")

st.title("Sistem CBR — Rekomendasi Mobil Bekas")
st.caption("Case-Based Reasoning • B-Tree Indexing • Flat Feature-Value Representation • 9.583 data")

if not os.path.exists(db_path):
    st.error(f"Dataset file not found at: {db_path}.")
else:
    if 'engine' not in st.session_state:
        st.session_state.engine = CarCBREngine(db_path)
    
    tab1, tab2, tab3, tab4 = st.tabs(["New Query", "CBR Results", "Indexing Process", "Case Dataset"])

    if 'results' not in st.session_state:
        st.session_state.results = None

    with tab1:
        st.subheader("Enter Car Specifications")
        df_db = st.session_state.engine.cases_df
        available_brands = sorted(df_db['Brand'].unique().tolist())
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            brand = st.selectbox("BRAND", available_brands)
            df_filtered_by_brand = df_db[df_db['Brand'] == brand]
            km_driven = st.number_input("KM DRIVEN", min_value=0, value=45000, step=5000)
            
        with col_f2:
            available_models = sorted(df_filtered_by_brand['model'].unique().tolist())
            model = st.selectbox("MODEL", available_models)
            df_filtered_by_model = df_filtered_by_brand[df_filtered_by_brand['model'] == model]
            available_trans = sorted(df_filtered_by_model['Transmission'].unique().tolist())
            transmission = st.selectbox("TRANSMISSION", available_trans)
            
        with col_f3:
            year = st.number_input(
                "MANUFACTURING YEAR", 
                min_value=2000, 
                max_value=2026, 
                value=2018, 
                step=1
            )
            
            owner_options = ["First Owner", "Second Owner", "Third Owner"]
            owner = st.selectbox("OWNERSHIP STATUS", owner_options)
            
        col_f4, col_f5 = st.columns(2)
        with col_f4:
            available_fuels = sorted(df_filtered_by_model['FuelType'].unique().tolist())
            fuel_type = st.selectbox("FUEL TYPE", available_fuels)
        with col_f5:
            offered_price_idr = st.number_input("SELLER PRICE (Rp)", min_value=0, value=120000000, step=5000000)
            offered_price_inr = offered_price_idr / KURS_INR_TO_IDR

        st.markdown("---")
        st.subheader("Feature Similarity Weights")
        w_year = st.slider("Manufacturing Year (%)", 0, 100, 30)
        w_km = st.slider("Km Driven (%)", 0, 100, 40)
        w_owner = st.slider("Ownership (%)", 0, 100, 30)

        total_w = w_year + w_km + w_owner
        if total_w > 0:
            st.session_state.engine.weights = {
                "Year": w_year / total_w,
                "kmDriven": w_km / total_w,
                "Owner": w_owner / total_w
            }

        if st.button("Start", type="primary"):
            owner_int = 1 if "First" in owner else 2 if "Second" in owner else 3
            query_mobil = {
                "Brand": brand, "Model": model, "Year": int(year), 
                "kmDriven": km_driven, "Transmission": transmission, 
                "FuelType": fuel_type, "Owner": 1 if "First" in owner else 2
            }
            
            top_matches = st.session_state.engine.retrieve(query_mobil, k=5)
            
            if top_matches:
                final_price_inr = st.session_state.engine.reuse_and_revise(query_mobil, top_matches)
                st.session_state.results = {
                    "query": query_mobil,
                    "top_matches": top_matches,
                    "final_price": final_price_inr,
                    "offered_price": offered_price_inr
                }

                kasus_siap_simpan = query_mobil.copy()
                kasus_siap_simpan["AskPrice"] = final_price_inr
                log_retain = st.session_state.engine.retain(kasus_siap_simpan, highest_similarity=top_matches[0]['similarity'])
                
                st.success(f"Analysis Completed! {log_retain}")
            else:
                st.error("No matching historical cases found within the B-Tree hierarchy levels.")

    with tab2:
        if st.session_state.results is None:
            st.info("Please execute a query in the 'New Query' tab first.")
        else:
            res = st.session_state.results
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            p_diff = ((res['offered_price'] - res['final_price']) / res['final_price']) * 100
            
            with m_col1:
                st.metric(label="Recommended Price (System)", value=format_to_idr(res['final_price']))
            with m_col2:
                range_min = format_to_idr(res['final_price'] * 0.9)
                range_max = format_to_idr(res['final_price'] * 1.1)
                st.metric(label="Fair Price Range", value=f"{range_min} - {range_max}")
            with m_col3:
                st.metric(label="Seller Price", value=format_to_idr(res['offered_price']))
            with m_col4:
                status_txt = f"Overpriced (+{p_diff:.0f}%)" if p_diff > 5 else "Fair Price" if p_diff >= -5 else f"Underpriced ({p_diff:.0f}%)"
                st.metric(label="Price Status", value=status_txt)

            st.markdown("<br>", unsafe_allow_html=True)
            cbr_left, cbr_right = st.columns([1, 1])

            with cbr_left:
                st.subheader("CBR Cycle Process")
                st.write("1. Retrieve (B-Tree & Heterogeneous Similarity)")
                st.write("2. Reuse (Weighted Price Average)")
                st.write("3. Revise (Rule-Based Specification Correction)")
                st.write("4. Retain (Automatic Recommendation Storage)")
                
                st.markdown("---")
                st.subheader("Price Adjustment (Revise)")
                st.text_input("Base Recommended Price", value=format_to_idr(res['final_price']), disabled=True)

            with cbr_right:
                st.subheader("Closest Cases (Top-5)")
                
                for idx, match in enumerate(res['top_matches']):
                    owner_text = "First Owner" if match['case']['Owner'] == 1 else "Second Owner"
                    st.markdown(f"""
                    <div style='background-color: #262626; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #00ffcc;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <b>{match['case']['Brand']} {match['case']['model']} ({int(match['case']['Year'])})</b>
                            <span style='background-color: #333; padding: 2px 8px; border-radius: 12px; font-size:12px; color:#00ffcc;'>{match['similarity']*100:.1f}% Match</span>
                        </div>
                        <small style='color: #aaa;'>{int(match['case']['kmDriven']):,} km • {match['case']['Transmission']} • {match['case']['FuelType']} • {owner_text}</small>
                        <div style='text-align: right; font-weight: bold; color: #ffffff;'>Conversion: {format_to_idr(match['case']['AskPrice'])}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab3:
            st.subheader("Multi-level B-Tree Indexing Hierarchy Path Visualization")
            st.write("Step-by-step data reduction based on hierarchy index rules:")
            
            if st.session_state.results is not None:
                q = st.session_state.results['query']
                df_total = st.session_state.engine.cases_df
                total_awal = len(df_total)
                
                l1_df = df_total[df_total['Transmission'].str.lower() == q['Transmission'].lower()]
                total_l1 = len(l1_df)

                l2_df = l1_df[l1_df['FuelType'].str.lower() == q['FuelType'].lower()]
                total_l2 = len(l2_df)
                
                l2_df_numeric = l2_df.copy()
                if not l2_df_numeric.empty:
                    l2_df_numeric['Owner_Num'] = l2_df_numeric['Owner'].astype(str).str.lower().apply(
                        lambda x: 1 if 'first' in x or '1' in x else 2 if 'second' in x or '2' in x else 3
                    )
                else:
                    l2_df_numeric['Owner_Num'] = 0

                l3_df = l2_df_numeric[
                    (l2_df_numeric['Year'] >= q['Year'] - 3) & (l2_df_numeric['Year'] <= q['Year'] + 3) &
                    (l2_df_numeric['Owner_Num'] >= q['Owner'] - 1) & (l2_df_numeric['Owner_Num'] <= q['Owner'] + 1)
                ]
                total_l3 = len(l3_df)

                idx_c1, idx_c2, idx_c3, idx_c4 = st.columns(4)
                with idx_c1:
                    st.metric(label="Total Database", value=f"{total_awal:,} Cases")
                with idx_c2:
                    st.metric(label="Level 1 (Transmission)", value=f"{total_l1:,} Passed", delta=f"-{total_awal - total_l1:,}")
                with idx_c3:
                    st.metric(label="Level 2 (Fuel Type)", value=f"{total_l2:,} Passed", delta=f"-{total_l1 - total_l2:,}")
                with idx_c4:
                    st.metric(label="Level 3 (Year & Owner)", value=f"{total_l3:,} Final", delta=f"-{total_l2 - total_l3:,}")
                
                st.markdown("---")
                st.markdown("#### Detailed Log Tracking:")
                st.info(f"[Level 1] Primary Index: Isolated variant types to '{q['Transmission']}', maintaining {total_l1:,} target matches.")
                st.info(f"[Level 2] Secondary Index: Sub-routed nodes to fuel types '{q['FuelType']}', maintaining {total_l2:,} target matches.")
                st.info(f"[Level 3] Tertiary Index: Constrained dimensional criteria bounds to safe intervals, matching {total_l3:,} items.")
                
                if total_l3 > 0:
                    st.success(f"Final subset formed with {total_l3:,} cases transferred to the engine for Weighted Heterogeneous Similarity calculation.")
                else:
                    st.warning("Automatic criteria relaxation triggered due to empty level 3 subset bounds. Fallback mechanisms initialized.")
                    
            else:
                st.warning("Indexing log trace data is empty. Please process a vehicle evaluation profile inside the 'New Query' tab.")
                
    with tab4:
        st.subheader("Knowledge Base Case Exploration")
        st.dataframe(st.session_state.engine.cases_df, use_container_width=True)