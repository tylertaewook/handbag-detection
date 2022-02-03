import numpy as np
import argparse
import imutils
import glob
import cv2
import time
import json
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--templates", required=True, help="Path to template image")
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
args = vars(ap.parse_args())


def multiscale_matchtemp(dirname=None):
    # cv2.imshow("Template", template)
    resultdict = {}

    imagesets = glob.glob(args["images"] + "/*.png")
    logosets = glob.glob(args["templates"] + "/*.png")

    for imagePath in tqdm(imagesets):
        imagename = imagePath.rsplit("/", 1)[-1]
        maxVal = foreachLogo(imagePath, logosets)

        # print(imagename, "overall maxVal: ", maxVal)
        if maxVal > float(args["threshold"]):
            resultdict[imagename] = maxVal
            continue

    return resultdict


def foreachLogo(imagePath, logos):
    for logoPath in logos:
        template = cv2.imread(logoPath)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]

        image = cv2.imread(imagePath)
        if image is None:
            return 0
        # Adding randn noise
        noise = np.zeros(image.shape, np.int32)
        cv2.randn(noise, 50, 10)
        image = cv2.add(image, noise, dtype=cv2.CV_8UC3)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # box smaller than resized img
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # template matching
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if args.get("--visualize", True):
            #     clone = np.dstack([edged, edged, edged])
            #     cv2.rectangle(
            #         clone,
            #         (maxLoc[0], maxLoc[1]),
            #         (maxLoc[0] + tW, maxLoc[1] + tH),
            #         (0, 0, 255),
            #         2,
            #     )
            #     cv2.imshow("Visualize", clone)
            #     cv2.waitKey(0)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
            # print(imagePath, maxVal)
            # early return when maxv above threshold found
            if maxVal > float(args["threshold"]):
                return maxVal
    return maxVal


def write_json_ds(res):
    final = {}
    detected = {}
    for key, value in res.items():
        if value > float(args["threshold"]):
            detected[key] = value
    final["DETECTED"] = detected
    final["MEAN"] = sum(res.values()) / len(res)

    with open("res_multiscale_ds.json", "w") as file:
        json.dump(final, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    start_time = time.process_time()
    res = multiscale_matchtemp()
    write_json_ds(res)
    end_time = time.process_time()
    elapsed_time = end_time - start_time

    print("Done! Elapsed Time: ", elapsed_time)
