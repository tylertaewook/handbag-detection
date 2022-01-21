import numpy as np
import argparse
import imutils
import glob
import cv2
import time
import json

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


def multiscale_matchtemp(dirname):
    template = cv2.imread(args["template"])
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    cv2.imshow("Template", template)
    resultdict = {}

    for imagePath in glob.glob(args["images"] + "/" + dirname + "/*.jpg"):
        image = cv2.imread(imagePath)
        imagename = imagePath.rsplit("/", 1)[-1]
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

            if args.get("visualize", False):
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(
                    clone,
                    (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH),
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
            # maxv above threshold found
            if maxVal > float(args["threshold"]):
                break
        (maxVal, maxLoc, r) = found
        # print(imagename, "maxVal: ", maxVal)
        resultdict[imagename] = maxVal
        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        # (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        # (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # # draw a bounding box around the detected result and display the image
        # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
    return resultdict


def write_json(res_pxg, res_etc):
    final = {}
    merged = {**res_pxg, **res_etc}  # ! syntax error when not on debugger
    merged = {}
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


if __name__ == "__main__":
    start_time = time.process_time()
    res_gc = multiscale_matchtemp("gc")
    res_etc = multiscale_matchtemp("pxg")
    write_json(res_gc, res_etc)
    end_time = time.process_time()
    elapsed_time = end_time - start_time

    print("Done! Elapsed Time: ", elapsed_time)
