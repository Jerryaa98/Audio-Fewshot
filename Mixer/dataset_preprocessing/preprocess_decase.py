import os
import pandas as pd
import re

def process_audio_directory(audio_dir, output_csv):
    rows = []
    for root, _, files in os.walk(audio_dir):
        for fname in files:
            if fname.lower().endswith('.wav'):
                path = os.path.join(root, fname)
                name, _ = os.path.splitext(fname)
                cls = re.sub(r'\d+$', '', name)
                rows.append((path, cls))
    df = pd.DataFrame(rows, columns=['audio_path', 'category'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("Saved full dataset:", output_csv)

def split_dataset_by_correlation(input_csv, correlation_csv,
                                 foreground_csv, background_csv,
                                 fg_filter=None):
    df = pd.read_csv(input_csv)
    corr = pd.read_csv(correlation_csv)
    fg_set = set(corr['Foreground'].dropna())
    bg_set = set(corr[['Background_1','Background_2','Background_3','Background_4']]
                 .stack().dropna())
    fg = df[df['category'].isin(fg_set)]
    bg = df[df['category'].isin(bg_set)]

    if fg_filter:
        fg = fg[fg['category'].isin(fg_filter)]

    for p in (foreground_csv, background_csv):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    fg.to_csv(foreground_csv, index=False)
    bg.to_csv(background_csv, index=False)
    print("Saved FG:", foreground_csv)
    print("Saved BG:", background_csv)

if __name__ == '__main__':
    # Modify these to your own paths:
    audio_dir = "/root/SC/Datasets/DECASE2016/dcase2016_task2_train_dev/dcase2016_task2_train"
    corr_csv = "/root/SC/dataset_preprocessing/dataset_correlation.csv"
    full_csv = "/root/SC/Preprocessed_CSVs/decase_csvs/decase_full_dataset.csv"
    fg_csv = "/root/SC/Preprocessed_CSVs/decase_csvs/foreground_dataset.csv"
    bg_csv = "/root/SC/Preprocessed_CSVs/decase_csvs/background_dataset.csv"
    # Ensure the output directories exist
    os.makedirs(os.path.dirname(full_csv), exist_ok=True)
    os.makedirs(os.path.dirname(fg_csv), exist_ok=True)
    os.makedirs(os.path.dirname(bg_csv), exist_ok=True)
    process_audio_directory(audio_dir, full_csv)
  # Only these six classes in foreground
    desired_fg = {'clearthroat', 'doorslam', 'keysDrop', 'pageturn', 'phone', 'drawer'}

    split_dataset_by_correlation(full_csv, corr_csv, fg_csv, bg_csv, fg_filter=desired_fg)