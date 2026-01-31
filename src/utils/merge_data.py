import pandas as pd
import glob


def data_combine(data_path=None, out=False):
    if data_path is None:
        data_path = glob.glob("../data/raw/*.csv")
    texts = []
    for file in data_path:
        df = pd.read_csv(file)
        cols = df.iloc[:, 1]
        cols = cols.dropna()
        cols = cols[cols.str.lower().str.strip() != "text"]
        cols = cols[cols.str.strip() != ""]

        texts.append(cols)

    all_texts = pd.concat(texts, ignore_index=True)

    combined = pd.DataFrame({'text': all_texts})
    if out:
        combined.to_csv("../data/combined.csv", index=False)

    return combined