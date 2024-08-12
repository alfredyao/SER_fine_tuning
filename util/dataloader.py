
from tqdm import tqdm
import torchaudio
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def dfcreator(base_dirs):
  """
  This function create csv file for training and testing data
  """
  data = []

  for base_dir in base_dirs:
    for path in tqdm(Path(base_dir).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = str(path).split('/')[-2]

        try:

            s = torchaudio.load(path)
            data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:

            pass
  df = pd.DataFrame(data)

  df = df[df["path"].apply(lambda path: os.path.exists(path))]

  df = df.dropna(subset=["path"])



  df = df.sample(frac=1)
  df = df.reset_index(drop=True)

  return df

def save_traindata(save_path, df: pd.DataFrame):
  train_df, test_df = train_test_split(df, test_size=0.1, random_state=101, stratify=df["emotion"])

  train_df = train_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
  test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)


  print(train_df.shape)
  print(test_df.shape)


def save_testextdata(save_path, df: pd.DataFrame):

  train_df_extra = df.reset_index(drop=True)


  train_df_extra.to_csv(f"{save_path}/test_extra.csv", sep="\t", encoding="utf-8", index=False)



  print(train_df_extra.shape)



