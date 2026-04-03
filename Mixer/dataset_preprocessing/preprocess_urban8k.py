import os
import pandas as pd

def process_urbansound8k_dataset(urbansound8k_folder, output_csv):
    # Paths to the audio and metadata
    audio_base_dir = os.path.join(urbansound8k_folder, 'audio')
    meta_file = os.path.join(urbansound8k_folder, 'metadata', 'UrbanSound8K.csv')

    if not os.path.exists(audio_base_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_base_dir}")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    # Read metadata
    meta_df = pd.read_csv(meta_file)

    # Validate required columns
    if 'slice_file_name' not in meta_df.columns or 'fold' not in meta_df.columns or 'class' not in meta_df.columns:
        raise ValueError("Metadata file must contain 'slice_file_name', 'fold', and 'class' columns")

    # Filter rows to only include the desired classes
    included_classes = {
        "air_conditioner",
        "children_playing",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "street_music"
    }
    meta_df = meta_df[meta_df['class'].isin(included_classes)]

    # Build full paths and extract class
    new_data = []
    for _, row in meta_df.iterrows():
        fold_folder = f"fold{row['fold']}"
        filename = row['slice_file_name']
        category = row['class']
        full_path = os.path.join(audio_base_dir, fold_folder, filename)
        if os.path.exists(full_path):
            new_data.append([full_path, category])
        else:
            print(f"Warning: Audio file not found: {full_path}")

    # Save new dataset
    new_df = pd.DataFrame(new_data, columns=['audio_path', 'category'])
    new_df.to_csv(output_csv, index=False)
    print(f"Processed dataset saved to: {output_csv}")

    def split_dataset_by_correlation(input_csv, correlation_csv, foreground_csv, background_csv):
        processed_df = pd.read_csv(input_csv)
        correlation_df = pd.read_csv(correlation_csv)

        foreground_categories = set(correlation_df['Foreground'].dropna())
        background_categories = set(correlation_df[['Background_1', 'Background_2', 'Background_3', 'Background_4']].stack().dropna())

        foreground_df = processed_df[processed_df['category'].isin(foreground_categories)]
        background_df = processed_df[processed_df['category'].isin(background_categories)]

        foreground_df.to_csv(foreground_csv, index=False)
        background_df.to_csv(background_csv, index=False)

        print(f"Foreground dataset saved to: {foreground_csv}")
        print(f"Background dataset saved to: {background_csv}")

    # Run correlation split
    correlation_csv = "/root/SC/dataset_preprocessing/dataset_correlation.csv"
    foreground_csv = "/root/SC/Preprocessed_CSVs/urbansound8k_csvs/foreground_dataset.csv"
    background_csv = "/root/SC/Preprocessed_CSVs/urbansound8k_csvs/background_dataset.csv"
    split_dataset_by_correlation(output_csv, correlation_csv, foreground_csv, background_csv)

# Example usage
if __name__ == "__main__":
    urbansound8k_folder = "/root/SC/Datasets/Urban8k"
    output_csv = "/root/SC/Preprocessed_CSVs/urbansound8k_csvs/urbansound8k_all_dataset.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    process_urbansound8k_dataset(urbansound8k_folder, output_csv)
