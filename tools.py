import cv2
import numpy as np
from PIL import Image
import copy


def colormap(rgb=True):
	color_list = np.array(
		[
			0.000, 0.000, 0.000,
			1.000, 1.000, 1.000,
			1.000, 0.498, 0.313,
			0.392, 0.581, 0.929,
			0.000, 0.447, 0.741,
			0.850, 0.325, 0.098,
			0.929, 0.694, 0.125,
			0.494, 0.184, 0.556,
			0.466, 0.674, 0.188,
			0.301, 0.745, 0.933,
			0.635, 0.078, 0.184,
			0.300, 0.300, 0.300,
			0.600, 0.600, 0.600,
			1.000, 0.000, 0.000,
			1.000, 0.500, 0.000,
			0.749, 0.749, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 1.000,
			0.667, 0.000, 1.000,
			0.333, 0.333, 0.000,
			0.333, 0.667, 0.000,
			0.333, 1.000, 0.000,
			0.667, 0.333, 0.000,
			0.667, 0.667, 0.000,
			0.667, 1.000, 0.000,
			1.000, 0.333, 0.000,
			1.000, 0.667, 0.000,
			1.000, 1.000, 0.000,
			0.000, 0.333, 0.500,
			0.000, 0.667, 0.500,
			0.000, 1.000, 0.500,
			0.333, 0.000, 0.500,
			0.333, 0.333, 0.500,
			0.333, 0.667, 0.500,
			0.333, 1.000, 0.500,
			0.667, 0.000, 0.500,
			0.667, 0.333, 0.500,
			0.667, 0.667, 0.500,
			0.667, 1.000, 0.500,
			1.000, 0.000, 0.500,
			1.000, 0.333, 0.500,
			1.000, 0.667, 0.500,
			1.000, 1.000, 0.500,
			0.000, 0.333, 1.000,
			0.000, 0.667, 1.000,
			0.000, 1.000, 1.000,
			0.333, 0.000, 1.000,
			0.333, 0.333, 1.000,
			0.333, 0.667, 1.000,
			0.333, 1.000, 1.000,
			0.667, 0.000, 1.000,
			0.667, 0.333, 1.000,
			0.667, 0.667, 1.000,
			0.667, 1.000, 1.000,
			1.000, 0.000, 1.000,
			1.000, 0.333, 1.000,
			1.000, 0.667, 1.000,
			0.167, 0.000, 0.000,
			0.333, 0.000, 0.000,
			0.500, 0.000, 0.000,
			0.667, 0.000, 0.000,
			0.833, 0.000, 0.000,
			1.000, 0.000, 0.000,
			0.000, 0.167, 0.000,
			0.000, 0.333, 0.000,
			0.000, 0.500, 0.000,
			0.000, 0.667, 0.000,
			0.000, 0.833, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 0.167,
			0.000, 0.000, 0.333,
			0.000, 0.000, 0.500,
			0.000, 0.000, 0.667,
			0.000, 0.000, 0.833,
			0.000, 0.000, 1.000,
			0.143, 0.143, 0.143,
			0.286, 0.286, 0.286,
			0.429, 0.429, 0.429,
			0.571, 0.571, 0.571,
			0.714, 0.714, 0.714,
			0.857, 0.857, 0.857
		]
	).astype(np.float32)
	color_list = color_list.reshape((-1, 3)) * 255
	if not rgb:
		color_list = color_list[:, ::-1]
	return color_list


color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def gauss_filter(kernel_size, sigma):
	max_idx = kernel_size // 2
	idx = np.linspace(-max_idx, max_idx, kernel_size)
	Y, X = np.meshgrid(idx, idx)
	gauss_filter = np.exp(-(X**2 + Y**2) / (2*sigma**2))
	gauss_filter /= np.sum(np.sum(gauss_filter))
	
	return gauss_filter


def vis_add_mask(image, mask, color, alpha, kernel_size):
	color = np.array(color)
	mask = mask.astype('float').copy()
	mask = (cv2.GaussianBlur(mask, (kernel_size, kernel_size), kernel_size) / 255.) * (alpha)

	for i in range(3):
		image[:, :, i] = image[:, :, i] * (1-alpha+mask) + color[i] * (alpha-mask)

	return image


def vis_add_mask_wo_blur(image, mask, color, alpha):
	color = np.array(color)
	mask = mask.astype('float').copy()
	for i in range(3):
		image[:, :, i] = image[:, :, i] * (1-alpha+mask) + color[i] * (alpha-mask)
	return image


def mask_painter(input_image, input_mask, background_alpha=0.7, background_blur_radius=7, contour_width=3, contour_color=3, contour_alpha=1):
	"""
	Input:
	input_image: numpy array
	input_mask: numpy array
	background_alpha: transparency of background, [0, 1], 1: all black, 0: do nothing
	background_blur_radius: radius of background blur, must be odd number
	contour_width: width of mask contour, must be odd number
	contour_color: color index (in color map) of mask contour, 0: black, 1: white, >1: others
	contour_alpha: transparency of mask contour, [0, 1], if 0: no contour highlighted

	Output:
	painted_image: numpy array
	"""
	assert input_image.shape[:2] == input_mask.shape, 'different shape'
	assert background_blur_radius % 2 * contour_width % 2 > 0, 'background_blur_radius and contour_width must be ODD'

	width, height = input_image.shape[0], input_image.shape[1]
	res = 1024
	ratio = min(1.0 * res / max(width, height), 1.0)  
	input_image = cv2.resize(input_image, (int(height*ratio), int(width*ratio)))
	input_mask = cv2.resize(input_mask, (int(height*ratio), int(width*ratio)))
	# 0: background, 1: foreground
	input_mask[input_mask>0] = 255

	# mask background
	painted_image = vis_add_mask(input_image, input_mask, color_list[0], background_alpha, background_blur_radius)	# black for background
	# mask contour
	contour_mask = input_mask.copy()
	contour_mask = cv2.Canny(contour_mask, 100, 200)	# contour extraction
	# widden contour
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (contour_width, contour_width))
	contour_mask = cv2.dilate(contour_mask, kernel)
	painted_image = vis_add_mask(painted_image, 255-contour_mask, color_list[contour_color], contour_alpha, contour_width)
	painted_image = cv2.resize(painted_image, (height, width))
	return painted_image


if __name__ == '__main__':
	
	background_alpha = 0.7  	# transparency of background 1: all black, 0: do nothing
	background_blur_radius = 35	# radius of background blur, must be odd number
	contour_width = 7       	# contour width, must be odd number
	contour_color = 3      		# id in color map, 0: black, 1: white, >1: others
	contour_alpha = 1       	# transparency of background, 0: no contour highlighted

	# load input image and mask
	input_image = np.array(Image.open('./test_img/painter_input_image.jpg').convert('RGB'))
	input_mask = np.array(Image.open('./test_img/painter_input_mask.jpg').convert('P'))
	
	# paint
	painted_image = mask_painter(input_image, input_mask, background_alpha, background_blur_radius, contour_width, contour_color, contour_alpha)

	# save
	painted_image = Image.fromarray(painted_image)
	painted_image.save('./test_img/painter_output_image.png')
