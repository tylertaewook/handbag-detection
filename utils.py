import pandas as pd
import cv2
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve


def load_img(df, dirname="./dataset/"):
    """
        load dataset imgs from gcmrg.csv into folder ./dataset as .png
    """
    for row in df.head(len(df)).itertuples():
        urlretrieve(row.img_url, dirname + row.id + ".png")


def resTooutput():
    """
        converts res.json file produced by multiscale.py to output.json
        which is easier to verify each img url
    """
    gcmrg = pd.read_csv("./gcmrg.csv", index_col=[0])
    output = {}
    with open("./res_multiscale_ds.json") as json_file:
        data = json.load(json_file)

        for imageFile in data["DETECTED"]:
            imgDict = {}
            imgDict["score"] = data["DETECTED"][imageFile]
            imgDict["url"] = gcmrg["img_url"][imageFile.rsplit(".", 1)[0]]
            output[imageFile] = imgDict

    with open("./output.json", "w") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)


def verify():
    """
        iterates and displays every img in output.json to let user
        confirm whether it's correctly labelled or not
    """
    output = {}
    verified = {}
    with open("./output.json") as json_file:
        data = json.load(json_file)

        for img in data:
            imgObj = data[img]
            image_nparray = np.asarray(
                bytearray(requests.get(imgObj["url"]).content), dtype=np.uint8
            )
            image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
            cv2.imshow(img, image)
            ret = cv2.waitKey(0)
            if ret == 106:  # 'J'
                verified[img] = imgObj
                print(img, " is gucci")
            cv2.destroyAllWindows()

    with open("./verified-output.json", "w") as file:
        json.dump(verified, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # df = pd.read_csv("./gcmrg.csv", index_col=[0])
    # load_img(df)
    # resTooutput(df)
    verify()
