import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio
import torchaudio

def load_common_voice_dataset_from_tsv(tsv_path, clips_dir):
    """
    Load Common Voice dataset from local TSV.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[["path", "sentence"]].dropna()
    df["audio"] = df["path"].apply(lambda x: os.path.join(clips_dir, x))
    df = df.rename(columns={"sentence": "text"})
    return Dataset.from_pandas(df[["audio", "text"]])
