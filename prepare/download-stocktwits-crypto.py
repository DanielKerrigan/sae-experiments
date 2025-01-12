import pandas as pd

from datasets import ClassLabel, Dataset, Features, Value

sheet_to_df = pd.read_excel(
    "https://huggingface.co/datasets/ElKulako/stocktwits-crypto/resolve/main/st-data-full.xlsx",
    sheet_name=None,
)

df = pd.concat([sheet_to_df["stocktwits_1"], sheet_to_df["stocktwits_2"]]).dropna()

class_index_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
class_names = list(class_index_map.values())

df["text"] = df["text"].astype("string")
df["label"] = df["label"].map(class_index_map).astype("string")

features = Features({"text": Value("string"), "label": ClassLabel(names=class_names)})

dataset = Dataset.from_pandas(df, features=features, preserve_index=False)

dataset.save_to_disk("../datasets/ElKulako/stocktwits-crypto")
