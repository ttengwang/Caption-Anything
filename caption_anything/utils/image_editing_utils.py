from PIL import Image, ImageDraw, ImageFont
import copy
import numpy as np
import cv2


def wrap_text(text, font, max_width):
    lines = []
    words = text.split(' ')
    current_line = ''

    for word in words:
        if font.getsize(current_line + word)[0] <= max_width:
            current_line += word + ' '
        else:
            lines.append(current_line)
            current_line = word + ' '

    lines.append(current_line)
    return lines


def create_bubble_frame(image, text, point, segmask, input_points=(), input_labels=(),
                        font_path='assets/times_with_simsun.ttf', font_size_ratio=0.033, point_size_ratio=0.01):
    # Load the image
    if input_points is None:
        input_points = []
    if input_labels is None:
        input_labels = []

    if type(image) == np.ndarray:
        image = Image.fromarray(image)

    image = copy.deepcopy(image)
    width, height = image.size

    # Calculate max_text_width and font_size based on image dimensions and total number of characters
    total_chars = len(text)
    max_text_width = int(0.4 * width)
    font_size = int(height * font_size_ratio)
    point_size = max(int(height * point_size_ratio), 1)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Wrap the text to fit within the max_text_width
    lines = wrap_text(text, font, max_text_width)
    text_width = max([font.getsize(line)[0] for line in lines])
    _, text_height = font.getsize(lines[0])
    text_height = text_height * len(lines)

    # Define bubble frame dimensions
    padding = 10
    bubble_width = text_width + 2 * padding
    bubble_height = text_height + 2 * padding

    # Create a new image for the bubble frame
    bubble = Image.new('RGBA', (bubble_width, bubble_height), (255, 248, 220, 0))

    # Draw the bubble frame on the new image
    draw = ImageDraw.Draw(bubble)
    # draw.rectangle([(0, 0), (bubble_width - 1, bubble_height - 1)], fill=(255, 255, 255, 0), outline=(255, 255, 255, 0), width=2)
    draw_rounded_rectangle(draw, (0, 0, bubble_width - 1, bubble_height - 1), point_size * 2,
                           fill=(255, 248, 220, 120), outline=None, width=2)
    # Draw the wrapped text line by line
    y_text = padding
    for line in lines:
        draw.text((padding, y_text), line, font=font, fill=(0, 0, 0, 255))
        y_text += font.getsize(line)[1]

    # Determine the point by the min area rect of mask
    try:
        ret, thresh = cv2.threshold(segmask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        min_area_rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(min_area_rect)
        sorted_points = box[np.argsort(box[:, 0])]
        right_most_points = sorted_points[-2:]
        right_down_most_point = right_most_points[np.argsort(right_most_points[:, 1])][-1]
        x, y = int(right_down_most_point[0]), int(right_down_most_point[1])
    except:
        x, y = point
    # Calculate the bubble frame position
    if x + bubble_width > width:
        x = width - bubble_width
    if y + bubble_height > height:
        y = height - bubble_height

    # Paste the bubble frame onto the image
    image.paste(bubble, (x, y), bubble)
    draw = ImageDraw.Draw(image)
    colors = [(0, 191, 255, 255), (255, 106, 106, 255)]
    for p, label in zip(input_points, input_labels):
        point_x, point_y = p[0], p[1]
        left = point_x - point_size
        top = point_y - point_size
        right = point_x + point_size
        bottom = point_y + point_size
        draw.ellipse((left, top, right, bottom), fill=colors[label])
    return image


def draw_rounded_rectangle(draw, xy, corner_radius, fill=None, outline=None, width=1):
    x1, y1, x2, y2 = xy

    draw.rectangle(
        (x1, y1 + corner_radius, x2, y2 - corner_radius),
        fill=fill,
        outline=outline,
        width=width
    )
    draw.rectangle(
        (x1 + corner_radius, y1, x2 - corner_radius, y2),
        fill=fill,
        outline=outline,
        width=width
    )

    draw.pieslice((x1, y1, x1 + corner_radius * 2, y1 + corner_radius * 2), 180, 270, fill=fill, outline=outline,
                  width=width)
    draw.pieslice((x2 - corner_radius * 2, y1, x2, y1 + corner_radius * 2), 270, 360, fill=fill, outline=outline,
                  width=width)
    draw.pieslice((x2 - corner_radius * 2, y2 - corner_radius * 2, x2, y2), 0, 90, fill=fill, outline=outline,
                  width=width)
    draw.pieslice((x1, y2 - corner_radius * 2, x1 + corner_radius * 2, y2), 90, 180, fill=fill, outline=outline,
                  width=width)
