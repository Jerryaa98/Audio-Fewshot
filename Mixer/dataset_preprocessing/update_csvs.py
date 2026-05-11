import os
import pandas as pd

def build_updated_mapping(correlation_path):
    df = pd.read_csv(correlation_path)
    mapping = {}

    for _, row in df.iterrows():
        for updated_name in row.dropna():  # Go through Foreground + Background_*
            components = updated_name.split('+')
            for label in components:
                mapping[label] = updated_name
    return mapping

def update_csv_files(csv_dir, correlation_path):
    label_mapping = build_updated_mapping(correlation_path)

    for fname in os.listdir(csv_dir):
        if fname.endswith('.csv'):
            full_path = os.path.join(csv_dir, fname)
            df = pd.read_csv(full_path)

            if 'category' not in df.columns:
                print(f"Skipping {fname}: No 'category' column")
                continue

            # Replace category using mapping if substring match
            df['category'] = df['category'].apply(
                lambda label: label_mapping.get(label, label)
            )

            df.to_csv(full_path, index=False)
            print(f"Updated {fname}")

# === Example usage ===
if __name__ == '__main__':
    correlation_file = '/root/SC/dataset_preprocessing/Sheet1-Table 1.csv'

    csv_directory = '/root/SC/Preprocessed_CSVs/esc50_csvs/'
    update_csv_files(csv_directory, correlation_file)

    csv_directory = '/root/SC/Preprocessed_CSVs/decase_csvs/'
    update_csv_files(csv_directory, correlation_file)
    
    csv_directory = '/root/SC/Preprocessed_CSVs/dsed_csv/'
    update_csv_files(csv_directory, correlation_file)

    csv_directory = '/root/SC/Preprocessed_CSVs/urbansound8k_csvs/'
    update_csv_files(csv_directory, correlation_file)

    csv_directory = '/root/SC/Preprocessed_CSVs/vocalsound_csvs/'
    update_csv_files(csv_directory, correlation_file)
