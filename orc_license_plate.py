from automatic_license import py_image_search_ANPR
from imutils import paths
import argparse
import imutils
import cv2


def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, default="./images/biensoxe/",
                help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
                help="whether or to clear border pixels before OCR's img")
ap.add_argument("-p", "--psm", type=int, default=7,
                help="default PSM mode for OCR's img license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not to show additional visualizations")
args = vars(ap.parse_args())

anpr = py_image_search_ANPR(debug=args["debug"] > 0)

image_paths = sorted(list(paths.list_images(args["input"])))

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)

    (lp_text, lp_cnt) = anpr.find_and_ocr(image, psm=args["psm"],
                                        clear_border=args["clear_border"] > 0)
    print(args['psm'])
    print(args["clear_border"])
    print(lp_text)

    '''if lp_text is not None or lp_cnt is not None:
        box = cv2.boxPoints(cv2.minAreaRect(lp_cnt))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        (x, y, w, h)  = cv2.boundingRect(lp_cnt)
        cv2.putText(image, cleanup_text(lp_text), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        print("[INFO] {}".format(lp_text))
        cv2.imshow("Output ANPR", image)
        cv2.waitKey(0)

    else:
        print("False")'''
