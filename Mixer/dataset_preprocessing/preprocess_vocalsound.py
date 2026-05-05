import os
import csv
import pandas as pd

def generate_audio_csv(audio_dir: str, output_csv: str) -> None:
    """
    Walks through the audio directory, extracts .wav files with a specific naming pattern,
    and writes their paths and categories to a CSV file.

    Parameters:
    - audio_dir: Path to the directory containing audio files.
    - output_csv: Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    csv_rows = []

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('_')
                if len(parts) == 3 and parts[2].endswith('.wav'):
                    category = parts[2].replace('.wav', '')
                    file_path = os.path.join(root, file)
                    csv_rows.append([file_path, category])

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['audio_path', 'category'])
        writer.writerows(csv_rows)

    print(f"CSV file has been created at: {output_csv}")

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
    foreground_csv = "/root/SC/Preprocessed_CSVs/vocalsound_csvs/foreground_dataset.csv"
    background_csv = "/root/SC/Preprocessed_CSVs/vocalsound_csvs/background_dataset.csv"
    split_dataset_by_correlation(output_csv, correlation_csv, foreground_csv, background_csv)


if __name__ == '__main__':
    audio_dir = '/root/SC/Datasets/VocalSound/audio_16k/'
    output_csv = '/root/SC/Preprocessed_CSVs/vocalsound_csvs/vocalsound-all-dataset.csv'
    generate_audio_csv(audio_dir, output_csv)