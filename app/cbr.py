import pandas as pd
import os
import re

class CarCBREngine:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.cases_df = self._load_dataset()
        self.weights = {
            "kmDriven": 0.4,
            "Year": 0.3,
            "Owner": 0.3
        }

    def _clean_numeric(self, val):
        if pd.isna(val):
            return 0
        cleaned = re.sub(r'[^\d]', '', str(val))
        return int(cleaned) if cleaned != '' else 0

    def _clean_owner(self, val):
        if pd.isna(val):
            return 1
        val_str = str(val).lower()
        if 'first' in val_str or '1' in val_str:
            return 1
        elif 'second' in val_str or '2' in val_str:
            return 2
        elif 'third' in val_str or '3' in val_str:
            return 3
        else:
            return 4

    def _load_dataset(self):
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            if 'kmDriven' in df.columns:
                df['kmDriven'] = df['kmDriven'].apply(self._clean_numeric)
            if 'AskPrice' in df.columns:
                df['AskPrice'] = df['AskPrice'].apply(self._clean_numeric)
            if 'Owner' in df.columns:
                df['Owner'] = df['Owner'].apply(self._clean_owner)
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2015).astype(int)
            return df
        else:
            columns = ["Brand", "model", "Year", "kmDriven", "Transmission", "FuelType", "Owner", "AskPrice"]
            return pd.DataFrame(columns=columns)

    def _tree_indexing_filter(self, query):
        df_local = self.cases_df.copy()
        if not df_local.empty and df_local['kmDriven'].dtype == 'object':
            df_local['kmDriven'] = df_local['kmDriven'].astype(str).str.replace(r'[^\d]', '', regex=True)
            df_local['kmDriven'] = pd.to_numeric(df_local['kmDriven'], errors='coerce').fillna(0).astype(int)

        q_brand = query['Brand'].lower()
        q_model = query['Model'].lower()
        q_trans = query['Transmission'].lower()
        q_fuel = query['FuelType'].lower()

        mask_brand = df_local['Brand'].str.lower() == q_brand
        mask_model = df_local['model'].str.lower() == q_model if 'model' in df_local.columns else df_local['Model'].str.lower() == q_model
        mask_trans = df_local['Transmission'].str.lower() == q_trans
        mask_fuel = df_local['FuelType'].str.lower() == q_fuel
        mask_year = (df_local['Year'] >= query['Year'] - 3) & (df_local['Year'] <= query['Year'] + 3)

        subset = df_local[mask_brand & mask_model & mask_trans & mask_fuel & mask_year]
        if not subset.empty:
            return subset

        subset = df_local[mask_brand & mask_model & mask_trans]
        if not subset.empty:
            return subset

        subset_brand_alt = df_local[mask_brand & mask_trans & mask_fuel]
        if not subset_brand_alt.empty:
            return subset_brand_alt

        subset_cross_brand = df_local[mask_trans & mask_fuel & mask_year]
        if not subset_cross_brand.empty:
            return subset_cross_brand

        return df_local

    def _calculate_similarity(self, query, row, max_km):
        try:
            q_km = int(query['kmDriven'])
            r_km = int(row['kmDriven'])
        except:
            q_km, r_km = 0, 0
            
        dist_km = abs(q_km - r_km) / max_km if max_km > 0 else 0
        sim_km = 1 - dist_km

        try:
            q_year = int(query['Year'])
            r_year = int(row['Year'])
        except:
            q_year, r_year = 2020, 2020
            
        max_year_diff = 25  
        dist_year = abs(q_year - r_year) / max_year_diff
        sim_year = 1 - dist_year

        r_owner_raw = str(row['Owner']).lower().strip()
        if 'first' in r_owner_raw or '1' in r_owner_raw:
            r_owner_int = 1
        elif 'second' in r_owner_raw or '2' in r_owner_raw:
            r_owner_int = 2
        elif 'third' in r_owner_raw or '3' in r_owner_raw:
            r_owner_int = 3
        else:
            try:
                r_owner_int = int(r_owner_raw)
            except:
                r_owner_int = 1
                
        q_owner_int = int(query['Owner'])
        sim_owner = 1 - (abs(q_owner_int - r_owner_int) / 3)

        total_similarity = (
            (sim_km * self.weights['kmDriven']) +
            (sim_year * self.weights['Year']) +
            (sim_owner * self.weights['Owner'])
        )
        return total_similarity

    def retrieve(self, query, k=3):
        subset_df = self._tree_indexing_filter(query)
        if subset_df.empty:
            return []

        max_km = pd.to_numeric(subset_df['kmDriven'], errors='coerce').fillna(0).max()
        if max_km == 0:
            max_km = 100000

        scored_cases = []
        for index, row in subset_df.iterrows():
            row_cleaned = row.copy()
            score = self._calculate_similarity(query, row_cleaned, max_km)
            
            r_owner_str = str(row_cleaned['Owner']).lower()
            if 'first' in r_owner_str or '1' in r_owner_str:
                row_cleaned['Owner'] = 1
            else:
                row_cleaned['Owner'] = 2
                
            scored_cases.append({
                "case": row_cleaned.to_dict(),
                "similarity": round(score, 4)
            })

        scored_cases.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_cases[:k]

    def reuse_and_revise(self, query, top_cases):
        if not top_cases:
            return 0

        total_weight = sum([c['similarity'] for c in top_cases])
        base_price = 0
        for c in top_cases:
            weight = c['similarity'] / total_weight if total_weight > 0 else 1 / len(top_cases)
            base_price += c['case']['AskPrice'] * weight

        best_match = top_cases[0]['case']
        revised_price = base_price

        year_diff = query['Year'] - best_match['Year']
        revised_price += (year_diff * (0.05 * base_price))

        km_diff = query['kmDriven'] - best_match['kmDriven']
        revised_price -= (km_diff / 10000) * (0.02 * base_price)

        try:
            q_owner = int(query['Owner'])
            b_owner = int(best_match['Owner'])
        except:
            q_owner, b_owner = 1, 1

        owner_diff = q_owner - b_owner
        revised_price -= (owner_diff * (0.03 * base_price))

        if revised_price < 10000:
            revised_price = 10000

        return round(revised_price, 2)

    def retain(self, new_case, highest_similarity=0.0):
        if highest_similarity >= 0.95:
            return "Retain skipped. An identical case with similarity above 95% already exists in the database."

        txt_case = {
            "Brand": new_case["Brand"],
            "model": new_case["Model"], 
            "Year": int(new_case["Year"]),
            "Age": 2026 - int(new_case["Year"]), 
            "kmDriven": f"{int(new_case['kmDriven']):,} km".replace(',', '.'), 
            "Transmission": new_case["Transmission"],
            "FuelType": new_case["FuelType"],
            "Owner": "first" if new_case["Owner"] == 1 else "second",
            "PostedDate": "Jun-26",
            "AdditionInfo": "Added by CBR System Retain (System Recommended Price)",
            "AskPrice": f"₹ {int(new_case['AskPrice']):,}".replace(',', '.') 
        }
        
        new_row_df = pd.DataFrame([txt_case])
        new_row_df.to_csv(self.dataset_path, mode='a', header=not os.path.exists(self.dataset_path), index=False)
        
        mem_case = {
            "Brand": new_case["Brand"],
            "model": new_case["Model"],
            "Year": int(new_case["Year"]),
            "Age": 2026 - int(new_case["Year"]),
            "kmDriven": int(new_case["kmDriven"]),
            "Transmission": new_case["Transmission"],
            "FuelType": new_case["FuelType"],
            "Owner": int(new_case["Owner"]), 
            "PostedDate": "Jun-26",
            "AdditionInfo": "Added by CBR System Retain",
            "AskPrice": float(new_case["AskPrice"]) 
        }
        
        self.cases_df = pd.concat([self.cases_df, pd.DataFrame([mem_case])], ignore_index=True)
        return "New case with recommended price successfully retained to the CSV dataset."