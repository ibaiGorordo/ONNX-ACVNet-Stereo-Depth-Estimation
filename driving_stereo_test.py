import cv2
import numpy as np
import glob

from acvnet import ACVNet, CameraConfig

def get_driving_stereo_images(base_path, start_sample=0):

	# Get image list
	left_images = glob.glob(f'{base_path}/left/*.jpg')
	left_images.sort()
	right_images = glob.glob(f'{base_path}/right/*.jpg')
	right_images.sort()
	depth_images = glob.glob(f'{base_path}/depth/*.png')
	depth_images.sort()

	return left_images[start_sample:], right_images[start_sample:], depth_images[start_sample:]

# Store baseline (m) and focal length (pixel)
input_width = 480
camera_config = CameraConfig(0.546, 500/1720*input_width) # rough estimate from the original calibration
max_distance = 10

# Initialize model
model_path='models/acvnet_maxdisp192_sceneflow_320x480/acvnet_maxdisp192_sceneflow_320x480.onnx'
depth_estimator = ACVNet(model_path, camera_config)

# Get the driving stereo samples
driving_stereo_path = "drivingStereo"
start_sample = 700
left_images, right_images, depth_images = get_driving_stereo_images(driving_stereo_path, start_sample)
out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (881*2,400))

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
for left_path, right_path, depth_path in zip(left_images, right_images, depth_images):

	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/1000

	# Estimate the depth
	disparity_map = depth_estimator(left_img, right_img)
	color_depth = depth_estimator.draw_depth()

	# color_real_depth = depth_estimator.util_draw_depth(depth_img, (left_img.shape[1], left_img.shape[0]), max_distance)
	# combined_image = np.hstack((left_img, color_real_depth, color_depth))
	combined_image = np.hstack((left_img, color_depth))

	out.write(combined_image)
	cv2.imshow("Estimated depth", combined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

out.release()
cv2.destroyAllWindows()
