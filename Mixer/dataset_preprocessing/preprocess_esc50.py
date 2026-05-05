import os
import pandas as pd

def process_esc50_dataset(esc50_folder, output_csv):
    # Paths to the audio and meta directories
    audio_dir = os.path.join(esc50_folder, 'audio')
    meta_file = os.path.join(esc50_folder, 'meta', 'esc50.csv')

    # Check if the required directories and files exist
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Meta file not found: {meta_file}")

    # Load the meta CSV file
    meta_df = pd.read_csv(meta_file)

    # Ensure the required columns exist in the meta file
    if 'filename' not in meta_df.columns or 'category' not in meta_df.columns:
        raise ValueError("Meta file must contain 'filename' and 'category' columns")

    # Build the new CSV data
    new_data = []
    for _, row in meta_df.iterrows():
        audio_file_path = os.path.join(audio_dir, row['filename'])
        if os.path.exists(audio_file_path):
            new_data.append([audio_file_path, row['category']])
        else:
            print(f"Warning: Audio file not found: {audio_file_path}")

    # Create a new DataFrame and save it to the output CSV
    new_df = pd.DataFrame(new_data, columns=['audio_path', 'category'])
    new_df.to_csv(output_csv, index=False)
    print(f"Processed dataset saved to: {output_csv}")

    def split_dataset_by_correlation(input_csv, correlation_csv, foreground_csv, background_csv):
        # Load the processed dataset and the correlation CSV
        processed_df = pd.read_csv(input_csv)
        correlation_df = pd.read_csv(correlation_csv)

        # Extract foreground and background categories
        foreground_categories = set(correlation_df['Foreground'].dropna())
        background_categories = set(correlation_df[['Background_1', 'Background_2', 'Background_3', 'Background_4']].stack().dropna())

        # Filter the processed dataset into foreground and background
        foreground_df = processed_df[processed_df['category'].isin(foreground_categories)]
        background_df = processed_df[processed_df['category'].isin(background_categories)]

        # Save the filtered datasets to separate CSV files
        foreground_df.to_csv(foreground_csv, index=False)
        background_df.to_csv(background_csv, index=False)

        print(f"Foreground dataset saved to: {foreground_csv}")
        print(f"Background dataset saved to: {background_csv}")

    # Example usage
    correlation_csv = "/root/SC/dataset_preprocessing/dataset_correlation.csv"  # Replace with the path to your correlation CSV
    foreground_csv = "/root/SC/Preprocessed_CSVs/esc50_csvs/foreground_dataset.csv"  # Replace with the desired foreground CSV path
    background_csv = "/root/SC/Preprocessed_CSVs/esc50_csvs/background_dataset.csv"  # Replace with the desired background CSV path
    split_dataset_by_correlation(output_csv, correlation_csv, foreground_csv, background_csv)

# Example usage
if __name__ == "__main__":
    esc50_folder = "/root/SC/Datasets/ESC-50"  # Replace with the path to your ESC-50 folder
    output_csv = "/root/SC/Preprocessed_CSVs/esc50_csvs/esc-50-all-dataset.csv"  # Replace with the desired output CSV path
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    process_esc50_dataset(esc50_folder, output_csv)