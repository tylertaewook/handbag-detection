import pandas as pd
from urllib.request import urlretrieve


def load_img(n=22092, dirname="./dataset/"):
    df = pd.read_csv("gcmrg.csv")
    for row in df.head(n).itertuples():
        urlretrieve(row.img_url, dirname + row.id + ".png")


if __name__ == "__main__":
    load_img()
