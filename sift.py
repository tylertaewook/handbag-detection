import numpy as np
import argparse
import imutils
import glob
import cv2
import time
import json
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument(
    "-i",
    "--images",
    required=True,
    help="Path to images where template will be matched",
)
ap.add_argument(
    "-s",
    "--threshold",
    required=True,
    help="Threshold TM_COEFF_NORMED score for determining GC or ETC",
)
ap.add_argument(
    "-v",
    "--visualize",
    help="Flag indicating whether or not to visualize each iteration",
)
args = vars(ap.parse_args())


def sift_matchtemp(dirname=None):
    template = cv2.imread(args["template"])
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    resultdict = {}

    imagesets = (
        glob.glob(args["images"] + "/" + dirname + "/*.jpg")
        if dirname
        else glob.glob(args["images"] + "/*.png")
    )

    for imagePath in tqdm(imagesets):
        image = cv2.imread(imagePath)
        imagename = imagePath.rsplit("/", 1)[-1]
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return resultdict


def write_json(res_pxg, res_etc):
    final = {}
    merged = {**res_pxg, **res_etc}  # ! syntax error when not on debugger
    detected = {}
    na = {}
    for key, value in merged.items():
        if value > float(args["threshold"]):
            detected[key] = value
        else:
            na[key] = value
    final["DETECTED"] = detected
    final["NA"] = na
    # final["pxg_MAXV"] = res_pxg
    # final["etc_MAXV"] = res_etc
    final["pxg_MEAN"] = sum(res_pxg.values()) / len(res_pxg)
    final["etc_MEAN"] = sum(res_etc.values()) / len(res_etc)
    final["pxg_MIN"] = min(list(res_pxg.values()))
    final["etc_MAX"] = max(list(res_etc.values()))

    with open("res_multiscale.json", "w") as file:
        json.dump(final, file, ensure_ascii=False, indent=4)


def write_json_ds(res):
    final = {}
    detected = {}
    na = {}
    for key, value in res.items():
        if value > float(args["threshold"]):
            detected[key] = value
        else:
            na[key] = value
    final["DETECTED"] = detected
    final["NA"] = na
    # final["pxg_MAXV"] = res_pxg
    # final["etc_MAXV"] = res_etc
    final["MEAN"] = sum(res.values()) / len(res)
    final["MEAN"] = sum(res.values()) / len(res)

    with open("res_multiscale_ds.json", "w") as file:
        json.dump(final, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    start_time = time.process_time()
    # res_gc = multiscale_matchtemp("gc")
    # res_etc = multiscale_matchtemp("pxg")
    # write_json(res_gc, res_etc)
    res = sift_matchtemp()
    write_json_ds(res)
    end_time = time.process_time()
    elapsed_time = end_time - start_time

    print("Done! Elapsed Time: ", elapsed_time)
