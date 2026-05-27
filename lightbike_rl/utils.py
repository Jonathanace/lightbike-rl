import numpy as np
import cv2
from lightbike_rl.constants import COLOR_MAP

pixel_size = 5
grid_color = (128, 128, 128)  # Gray
thickness = 1

def render(frame):

    height, width = frame.shape
    raw_image = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in COLOR_MAP.items():
        raw_image[frame == value] = color

    new_width = width * pixel_size
    new_height = height * pixel_size

    scaled_image = cv2.resize(
        raw_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )

    # Gray lines
    # for x in range(0, new_width, pixel_size):
    #     cv2.line(scaled_image, (x, 0), (x, new_height), grid_color, thickness)

    # for y in range(0, new_height, pixel_size):
    #     cv2.line(scaled_image, (0, y), (new_width, y), grid_color, thickness)

    cv2.imshow("Rendered Grid", scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


