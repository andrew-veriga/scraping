"""
Script to fix Excel file with incorrect ID formatting and merge attachment data.
This script:
1. Reads discord_messages_fresh_pics.xlsx (correct string IDs, old image URLs)
2. Reads discord_messages_names_gcs_pics.xlsx (incorrect numeric IDs, correct GCS URLs)
3. Matches rows using first 14 digits of Message ID + Content field
4. Copies Attachments field from GCS file to fresh file
5. Saves result as discord_messages_correct.xlsx with leading apostrophes
"""

import pandas as pd
import os
from typing import Dict, Tuple, Optional

def read_excel_files():
    """Read both Excel files and return DataFrames."""
    print("📖 Reading Excel files...")
    
    # Read the fresh file (correct string IDs, old image URLs)
    fresh_file = "data/discord_messages_fresh_pics.xlsx"
    if not os.path.exists(fresh_file):
        print(f"❌ File not found: {fresh_file}")
        return None, None
    
    fresh_df = pd.read_excel(fresh_file)
    print(f"✅ Fresh file loaded: {len(fresh_df)} rows")
    print(f"   Columns: {list(fresh_df.columns)}")
    
    # Read the GCS file (incorrect numeric IDs, correct GCS URLs)
    gcs_file = "data/discord_messages_names_gcs_pics.xlsx"
    if not os.path.exists(gcs_file):
        print(f"❌ File not found: {gcs_file}")
        return None, None
    
    gcs_df = pd.read_excel(gcs_file)
    print(f"✅ GCS file loaded: {len(gcs_df)} rows")
    print(f"   Columns: {list(gcs_df.columns)}")
    
    return fresh_df, gcs_df

def create_matching_key(message_id: str, content: str) -> str:
    """Create a matching key using first 14 digits of Message ID + Content."""
    # Convert to string and clean it (remove invisible characters, whitespace)
    id_str = str(message_id).strip()
    # Remove any non-digit characters and get first 14 digits
    digits_only = ''.join(filter(str.isdigit, id_str))
    first_14_digits = digits_only[:14] if len(digits_only) >= 14 else digits_only
    
    # Clean content for matching (remove extra whitespace)
    clean_content = str(content).strip() if pd.notna(content) else ""
    
    return f"{first_14_digits}|{clean_content}"

def match_and_merge_data(fresh_df: pd.DataFrame, gcs_df: pd.DataFrame) -> pd.DataFrame:
    """Match rows between dataframes and merge attachment data."""
    print("🔍 Creating matching keys...")
    
    # Create matching keys for both dataframes
    fresh_df['match_key'] = fresh_df.apply(
        lambda row: create_matching_key(row['Message ID'], row['Content']), axis=1
    )
    
    gcs_df['match_key'] = gcs_df.apply(
        lambda row: create_matching_key(row['Message ID'], row['Content']), axis=1
    )
    
    print(f"📊 Fresh file unique keys: {fresh_df['match_key'].nunique()}")
    print(f"📊 GCS file unique keys: {gcs_df['match_key'].nunique()}")
    
    # Show sample matching keys for debugging
    print("\n🔍 Sample matching keys from fresh file:")
    print(fresh_df['match_key'].head(5).tolist())
    print("\n🔍 Sample matching keys from GCS file:")
    print(gcs_df['match_key'].head(5).tolist())
    
    # Create a mapping from match_key to attachments
    gcs_attachments_map = gcs_df.set_index('match_key')['Attachments'].to_dict()
    
    print("🔄 Merging attachment data...")
    
    # Copy the fresh dataframe
    result_df = fresh_df.copy()
    
    # Update attachments based on matching keys
    matches_found = 0
    for idx, row in result_df.iterrows():
        match_key = row['match_key']
        if match_key in gcs_attachments_map:
            result_df.at[idx, 'Attachments'] = gcs_attachments_map[match_key]
            matches_found += 1
    
    print(f"✅ Matches found: {matches_found} out of {len(result_df)} rows")
    
    # Remove the temporary match_key column
    result_df = result_df.drop('match_key', axis=1)
    
    return result_df

def save_with_string_ids(df: pd.DataFrame, output_file: str):
    """Save DataFrame with leading apostrophes to preserve string format for IDs."""
    print(f"💾 Saving to {output_file}...")
    
    # Create a copy to avoid modifying the original
    save_df = df.copy()
    
    # Add leading apostrophe to Message ID column to preserve string format
    if 'Message ID' in save_df.columns:
        save_df['Message ID'] = "'" + save_df['Message ID'].astype(str)
    
    # Add leading apostrophe to other ID columns if they exist
    id_columns = ['Author ID', 'Referenced Message ID', 'Parent ID', 'Thread ID']
    for col in id_columns:
        if col in save_df.columns:
            save_df[col] = "'" + save_df[col].astype(str)
    
    # Save to Excel
    save_df.to_excel(output_file, index=False)
    print(f"✅ File saved successfully: {output_file}")
    print(f"   Total rows: {len(save_df)}")
    print(f"   Columns: {list(save_df.columns)}")

def main():
    """Main function to execute the fix process."""
    print("🚀 Starting Excel ID fix process...")
    
    # Read Excel files
    fresh_df, gcs_df = read_excel_files()
    if fresh_df is None or gcs_df is None:
        return
    
    # Show sample data
    print("\n📋 Sample data from fresh file:")
    print(fresh_df[['Message ID', 'Content', 'Attachments']].head(3))
    
    print("\n📋 Sample data from GCS file:")
    print(gcs_df[['Message ID', 'Content', 'Attachments']].head(3))
    
    # Match and merge data
    result_df = match_and_merge_data(fresh_df, gcs_df)
    
    # Show sample of merged data
    print("\n📋 Sample merged data:")
    print(result_df[['Message ID', 'Content', 'Attachments']].head(3))
    
    # Save result
    output_file = "data/discord_messages_correct.xlsx"
    save_with_string_ids(result_df, output_file)
    
    print("\n🎉 Process completed successfully!")

if __name__ == "__main__":
    main()
