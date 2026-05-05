import os
import pandas as pd

def process_and_filter(audio_dir: str, tsv_path: str, output_csv: str):
    """
    Read TSV with columns ['filename', 'event_labels'].
    Filter to only rows where all labels belong to a target set.
    Then check that audio file exists under audio_dir.
    Write out CSV with columns ['file_path', 'category'].
    If multiple labels, joins as comma string in category.
    """
    df = pd.read_csv(tsv_path, sep='\t', usecols=['filename', 'event_labels'])
    # target labels
    allowed = {'Running_water', 'Electric_shaver_toothbrush', 'Blender'}
    
    # Split multi-labels
    df['labels_list'] = df['event_labels'].str.split(',')
    # strip whitespace
    df['labels_list'] = df['labels_list'].apply(lambda lst: [lbl.strip() for lbl in lst])
    
    # Filter rows where all labels are in allowed
    # df = df[df['labels_list'].apply(lambda lst: all(lbl in allowed for lbl in lst))]
    
    # Build file_path and category
    rows = []
    for _, row in df.iterrows():
        fname = row['filename']
        full_path = os.path.join(audio_dir, fname)
        if os.path.exists(full_path):
            # join labels back
            cat = ','.join(row['labels_list'])
            rows.append((full_path, cat))
        else:
            print(f"Warning: file not found {full_path}")
    
    out_df = pd.DataFrame(rows, columns=['file_path', 'category'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved filtered CSV to {output_csv}")


def postprocess_csv(output_csv, correlation_file):
    """
    Given an output CSV, check if the 'category' column contains a comma-separated list of labels.
    If more than one of them appears in the 'Foreground' column of the correlation file, remove that row.
    """
    # Load files
    df = pd.read_csv(output_csv)
    corr_df = pd.read_csv(correlation_file)

    # Get foreground labels (lowercased)
    foregrounds = set(str(fg).strip().lower() for fg in corr_df['Foreground'] if pd.notna(fg))

    # Keep only rows with <= 1 foreground label in the 'category' column
    def is_valid(row):
        categories = [c.strip().lower() for c in str(row['category']).split(',') if c.strip()]
        fg_count = sum(1 for c in categories if c in foregrounds)
        return fg_count <= 1

    filtered_df = df[df.apply(is_valid, axis=1)]

    # Overwrite the original CSV
    filtered_df.to_csv(output_csv, index=False)
    print(f"✅ Postprocessing complete. Original: {len(df)}, Filtered: {len(filtered_df)}")

    def clean_output_csv(output_csv, correlation_file):
        """
        For each row in the output CSV, split the 'category' column into foreground and background labels.
        Uses the 'Foreground' column from the correlation file to identify foregrounds.
        Adds two new columns: 'foreground_label' and 'background_label'.
        """
        # Load files
        df = pd.read_csv(output_csv)
        corr_df = pd.read_csv(correlation_file)

        # Build set of lowercase foreground labels
        foregrounds = set(str(fg).strip().lower() for fg in corr_df['Foreground'] if pd.notna(fg))

        # Function to extract foreground/background labels
        def extract_labels(category_str):
            labels = [lbl.strip() for lbl in str(category_str).split(',') if lbl.strip()]
            fg = next((l for l in labels if l.strip().lower() in foregrounds), '')
            bg = next((l for l in labels if l.strip().lower() not in foregrounds), '') if len(labels) > 1 else ''
            return pd.Series([fg, bg])

        # Apply and assign new columns
        df[['foreground_label', 'background_label']] = df['category'].apply(extract_labels)

        # Save back to the same file (or change this path to save elsewhere)
        df.to_csv(output_csv, index=False)
        print(f"✅ Updated CSV with foreground/background labels saved to: {output_csv}")    

    clean_output_csv(output_csv, correlation_file)    


# Example usage:
if __name__ == '__main__':
    audio_dir = '/root/SC/Datasets/DESED/data/dataset/audio/train/weak/'
    tsv_path = '/root/SC/Datasets/DESED/data/dataset/metadata/train/weak.tsv'
    output_csv = '/root/SC/Preprocessed_CSVs/dsed_csv/all_dataset.csv'
    correlation_file = '/root/SC/dataset_preprocessing/dataset_correlation.csv'
    process_and_filter(audio_dir, tsv_path, output_csv)
    postprocess_csv(output_csv, correlation_file)
