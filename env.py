from PIL import Image
import numpy as np
import cv2
import pyastar2d


class BuildEnvironment:
    def __init__(self, img, res=0.05) -> None:
        temp = Image.open(img)
        self.grid = np.asarray(temp)[..., 0]
        self.grid = (1 - np.round(self.grid / 255))
        self.grid = self.grid.astype(np.uint8)
        self.res = res
        self.clicked_point = (0, 0)
        # cv2.namedWindow("map")
        # cv2.setMouseCallback("map", self.get_location)

        
    def show_env(self):
        cv2.imshow("map", self.grid * 255)
        cv2.waitKey(1)

    def shotdown(self):
        cv2.destroyAllWindows()

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.grid, (x, y), 80, (55), -1)
            print(x, y)

    def get_location(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicked_point = (x, y)


if __name__ == "__main__":
    env = BuildEnvironment("maps/map1.png")
    cv2.setMouseCallback("map", env.draw_circle)
    img = env.grid
    while True:
        cv2.imshow("map", env.grid * 255)
        k = cv2.waitKey(20) & 0XFF
        if k == 27:
            cv2.destroyAllWindows()
            break
