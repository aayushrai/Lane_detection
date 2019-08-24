import cv2
import numpy as np
from PIL import ImageGrab

img = ImageGrab.grab()
img = np.array(img)
r = cv2.selectROI(img)
coordinate = int(r[0]), int(r[1]), int(r[0] + r[2]), int(r[1] + r[3])
print(r)
cv2.destroyAllWindows()


class lane_detection:

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        canny = cv2.Canny(blur, 50, 100)
        return canny

    def roi_image(self, image, polygons):
        height = image.shape[0]
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        roi_image = cv2.bitwise_and(image, mask)
        return roi_image

    def display_line(self, lines, image):
        line_image = np.zeros_like(image)
        if lines.all():
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        return line_image

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_coordinate(image, left_fit_average)
        right_line = self.make_coordinate(image, right_fit_average)
        return np.array([left_line, right_line])

    def make_coordinate(self, image, line_parameters):
        try:
            slope, intercept = line_parameters
            y1 = image.shape[0]
            y2 = int(y1 * (3 / 5))
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return np.array([x1, y1, x2, y2])
        except Exception:
            print("no lane detected")


while True:
    image = ImageGrab.grab(coordinate)
    image = np.array(image)
    image = cv2.resize(image, (1120, 700))
    polygon = np.array([[(0, 620), (1400, 620), (680, 410)]])
    lane_image = image.copy()
    lane = lane_detection()
    cny = lane.canny(lane_image)
    cropped_image = lane.roi_image(cny.copy(), polygon)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_line = lane.average_slope_intercept(lane_image, lines)
    line_image = lane.display_line(averaged_line, lane_image)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.line(combo_image, (0, 620), (1400, 620), (0, 0, 255), 3)
    cv2.line(combo_image, (1400, 620), (680, 410), (0, 0, 255), 3)
    cv2.line(combo_image, (680, 410), (0, 620), (0, 0, 255), 3)
    cv2.imshow("combo", combo_image)
    cv2.imshow("canny", line_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
