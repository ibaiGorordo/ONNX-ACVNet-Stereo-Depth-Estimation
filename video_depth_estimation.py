import cv2
import pafy
import numpy as np
import glob
from acvnet import ACVNet, CameraConfig

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/Yui48w71SG0'
start_time = 0 # skip first {start_time} seconds
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Store baseline (m) and focal length (pixel)
# TODO: Fix with the values witht the correct configuration
input_width = 1280
camera_config = CameraConfig(0.12, 0.5*input_width/0.72) 
max_distance = 10

# Initialize model
model_path='models/acvnet_maxdisp192_sceneflow_480x640/acvnet_maxdisp192_sceneflow_480x640.onnx'
depth_estimator = ACVNet(model_path, camera_config, max_distance)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue

	# Extract the left and right images
	left_img  = frame[:,:frame.shape[1]//3]
	right_img = frame[:,frame.shape[1]//3:frame.shape[1]*2//3]
	color_real_depth = frame[:,frame.shape[1]*2//3:]

	# Estimate the depth
	disparity_map = depth_estimator(left_img, right_img)
	color_depth = depth_estimator.draw_depth()
	combined_image = np.hstack((left_img, color_real_depth, color_depth))

	cv2.imshow("Estimated depth", combined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()