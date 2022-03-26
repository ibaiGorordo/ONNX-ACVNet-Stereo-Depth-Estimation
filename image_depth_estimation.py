import cv2
import numpy as np
from imread_from_url import imread_from_url

from acvnet import ACVNet

# Initialize model
model_path='models/acvnet_maxdisp192_sceneflow_384x640/acvnet_maxdisp192_sceneflow_384x640.onnx'
depth_estimator = ACVNet(model_path)

# Load images
left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

# Estimate the depth
disparity_map = depth_estimator(left_img, right_img)

color_disparity = depth_estimator.draw_disparity()
combined_image = np.hstack((left_img, color_disparity))

cv2.imwrite("out.jpg", combined_image)

cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
cv2.imshow("Estimated disparity", combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
