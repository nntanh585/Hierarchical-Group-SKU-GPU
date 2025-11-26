import re
import json
import pandas as pd

# Helper functions
def _read_json(json_file: str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def _clean_text(text: str) -> str:
    """Clean up, normalize to lowercase and remove special characters"""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text) # Remove HTML tags
    text = re.sub(r"\[.*?\]", " ", text) # Remove text in square brackets
    text = re.sub(r"[^\w\s]", " ", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

def _extract_adaptive_attributes(attributes_list: list) -> dict:
    """
    Extract all adaptive key-value pairs.
    """
    result_dict = {}
    if not attributes_list:
        return result_dict

    for attr in attributes_list:
        key = None
        value = None

        # Case 1: Structure 'required_attributes'
        if 'name' in attr and 'value' in attr:
            key_raw = attr.get('name')
            value_list = attr.get('value', [])
            if key_raw and value_list and value_list[0]:
                key = _clean_text(key_raw)
                value = _clean_text(value_list[0])
                unit = attr.get('unit')
                if unit:
                    value = f"{value} {unit}"

        # Case 2: Structure 'attribute_simples'
        elif 'attributes' in attr and 'attribute_values' in attr:
            key_raw = attr.get('attributes', {}).get('name')
            value_raw = attr.get('attribute_values', {}).get('name')
            if key_raw and value_raw:
                key = _clean_text(key_raw)
                value = _clean_text(value_raw)

        if key and value:
            result_dict[key] = value
        result_dict['quantity'] = attr.get('quantity', '')

    return result_dict

def _transform_data_from_json(json_file: str) -> pd.DataFrame:
    """
    Read JSON file and flatten the data.
    """
    full_json_data = _read_json(json_file)
    products_list = full_json_data.get('data', [])

    result_list = []

    for product_data in products_list:
        # 1. Parent information
        parent_id = product_data.get('id')
        parent_name = _clean_text(product_data.get('name'))
        parent_category = _clean_text(product_data.get('category', {}).get('name_en'))
        parent_short_desc = _clean_text(product_data.get('short_description'))

        # Safely handle jan_info
        jan_info = product_data.get('jan_info', {})
        jan_code = ""
        if isinstance(jan_info, dict):
            jan_code = str(jan_info.get('reason_no_code') or jan_info.get('code') or "")

        # 2. Loop through Children (SKUs)
        skus_data = product_data.get('product_skus', {}).get('data', [])
        if skus_data:
            for sku in skus_data:
                req_attrs_list = sku.get('required_attributes', [])
                simple_attrs_list = sku.get('attribute_simples', {}).get('data', [])
                quantity = sku.get('product_sku_detail', '').get('quantity', '')

                required_attributes = _extract_adaptive_attributes(req_attrs_list)
                attribute_simples = _extract_adaptive_attributes(simple_attrs_list)

                product_info = {
                    "product_id": parent_id,
                    "product_name": parent_name,
                    "category_name": parent_category,
                    "short_description": parent_short_desc,
                    "seller_sku": _clean_text(sku.get("seller_sku")),
                    "required_attributes": required_attributes,
                    "attribute_simples": attribute_simples,
                    "jan_infor": jan_code,
                    "quantity": quantity
                }
                result_list.append(product_info)

    return pd.DataFrame(result_list) if result_list else pd.DataFrame()

# Table generation functions
def _transform_data_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    output_rows = []
    master_ids = sorted(df['master_id'].unique())

    for mid in master_ids:
        master_group = df[df['master_id'] == mid]

        # Group by variant_id
        variants_in_master = {}
        for idx, row in master_group.iterrows():
            vid = row['variant_id']
            if vid not in variants_in_master:
                variants_in_master[vid] = []
            variants_in_master[vid].append(row)

        num_variant_groups = len(variants_in_master)
        master_name = master_group.iloc[0]['product_name']
        total_qty = master_group['quantity'].sum()
        description = master_group['short_description'].iloc[0]
        # --- Create master ---
        output_rows.append({
            'Product Master ID': f"ID{mid:03d}",
            'Product/SKU Name': master_name,
            'Description': description,
            'Group SKU': f"{num_variant_groups} Group SKU",
            'Quantity': total_qty,
        })

        # --- Create variants ---
        sorted_vids = sorted(variants_in_master.keys())
        for vid in sorted_vids:
            rows = variants_in_master[vid]

            # Sum the quantity of this variant_id group
            qty_sum = sum(r['quantity'] for r in rows)

            # Get representative information from the first line
            rep_row = rows[0]
            rep_name = rep_row['product_name']
            rep_sku = rep_row['seller_sku']
            rep_attrs = rep_row['attribute_simples']
            rep_attrs = ' - '.join([f'{k} {v}' for k, v in rep_attrs.items() if v != ''])
            description = rep_row['short_description']
            output_rows.append({
                'Product Master ID': f"ID{mid:03d}",
                'Product/SKU Name': f"{rep_name} - {rep_attrs}",
                'Description': description,    
                'Group SKU': rep_sku,
                'Quantity': qty_sum,
            })
        # Add an empty line after the end of a master loop
        output_rows.append({'Product Master ID': '', 'Product/SKU Name': '', 'Description': '', 'Group SKU': '', 'Quantity': ''})

    output_df = pd.DataFrame(output_rows)
    cols = ['Product Master ID', 'Product/SKU Name', 'Description', 'Group SKU',  'Quantity']
    output_df = output_df[cols]
    return output_df

# Text generation functions
def _get_master_text(product_row: pd.Series) -> str:
    """
    Create text string for MASTER clustering.
    Goal: Group variants (Color/Size) into the same group.
    Strategy: Only take common information (Name, Brand, Model, Category, JAN).
              EXCLUDE SKU and variant attributes.
    """
    name = product_row.get('product_name', '')
    category = product_row.get('category_name', '')
    short_description = product_row.get('short_description', '')
    jan_infor = product_row.get('jan_infor', '')

    # Get attributes to find Brand/Model, but not all
    attributes_dict = product_row.get('attributes', {})

    # Try to find Brand and Model in the messy attributes
    brand = ""
    model = ""

    # Common keywords to identify Brand/Model (cleaned text)
    for k, v in attributes_dict.items():
        if 'brand' in k or 'ブランド' in k: # Brand name
            brand = v
        if 'model' in k or '型番' in k: # Model number
            model = v

    # Prompt structure for Master (No SKU, no specific Color/Size)
    text = f"Product: {name}\nBrand: {brand}\nModel: {model}\nCategory: {category}\nJAN: {jan_infor}\nDescription: {short_description}"
    return text

def _get_variant_text(product_row: pd.Series) -> str:
    """
    Create text string for VARIANT clustering.
    Goal: Differentiate between SKUs within the same Master group.
    Strategy: Focus on SKU and ALL detailed attributes.
    """
    # Still need the name to maintain context (e.g., distinguish Shirt vs Pants if Master grouping is incorrect)
    # But the main weight will be on Details and SKU
    # name = product_row.get('product_name', '')
    size_mapping = {
        'xs': 'extra small', 's': 'small', 'm': 'medium', 'l': 'large',
        'xl': 'extra large', 'xxl': 'double extra large', '2xl': 'double extra large',
        'xxxl': 'triple extra large', '3xl': 'triple extra large',
        'f': 'freesize', 'free': 'freesize'
    }

    required_attributes = product_row.get('required_attributes', {})
    attribute_simples = product_row.get('attribute_simples', {})
    sku = product_row.get('seller_sku', '')
    # Remove unwanted keys (as requested by user)
    remove_keys = {'総個数', '総重量', '総容量'}
    filtered_required_attrs = {
        k: v for k, v in required_attributes.items() 
        if k not in remove_keys and v is not None and str(v).strip() != ''
    }

    filtered_simple_attrs = {
        k: size_mapping.get(str(v).strip().lower(), v) 
        for k, v in attribute_simples.items()
        if v is not None and str(v).strip() != ''       
    }
    
    # Create attribute details string
    details_required_attributes = " | ".join([f"{k}: {v}" for k, v in filtered_required_attrs.items()])
    details_attribute_simples = " | ".join([f"{k}: {" ".join([str(v).upper()] * 10)}" for k, v in filtered_simple_attrs.items()])
    details = f"{details_attribute_simples} | {details_required_attributes}"

    # Prompt structure for Variant (SKU and Details are most important)
    text = f"Seller SKU: {sku}\nDetails: {details}"
    return text
