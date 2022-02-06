import pandas as pd
import json

if __name__ == "__main__":
    gcmrg = pd.read_csv("./gcmrg2.csv", index_col=[0])
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
