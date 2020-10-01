from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import pytesseract


class py_image_search_ANPR:
    def __init__(self, minAR=2, maxAR=3, debug=False):
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug

    def debug_imshow(self, title, image, wait_key=False):
        if self.debug:
            cv2.imshow(title, image)

            if wait_key:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)
        self.debug_imshow("Blackhat", blackhat)

        square_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kern)
        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)

        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")
        self.debug_imshow("Scharr", grad_x)

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kern)
        thresh = cv2.threshold(grad_x, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)

        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, wait_key=True)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        return cnts

    def locate_license_plate(self, gray, candidates, clear_boder=False):
        lp_cnt = None
        roi = None
        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if self.minAR <= ar <= self.maxAR:
                lp_cnt = c
                license_plate = gray[y: y + h, x: x + w]
                roi = cv2.threshold(license_plate, 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                if clear_boder:
                    roi = clear_boder(roi)

                self.debug_imshow("License Plate", license_plate)
                self.debug_imshow("Roi", roi, wait_key=False)
                break
        return roi, lp_cnt

    def build_tesseract_option(self, psm=7):
        options = ""
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " -l eng"
        options += " --psm {}".format(psm)
        options += " -oem 3"

        return options

    def find_and_ocr(self, image, psm=7, clear_border=False):
        lp_text = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (lp, lp_cnt) = self.locate_license_plate(gray,
                                                 candidates, clear_boder=clear_border)

        if lp is not None:
            options = self.build_tesseract_option(psm=psm)
            lp_text = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License", lp)

        return lp_text, lp_cnt
