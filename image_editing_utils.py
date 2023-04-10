from PIL import Image, ImageDraw, ImageFont
import copy

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

def create_bubble_frame(image, text, point, font_path='TlwgMono.ttf', font_size=60):
    # Load the image
    image = copy.deepcopy(image)
    width, height = image.size

    # Calculate max_text_width and font_size based on image dimensions and total number of characters
    total_chars = len(text)
    max_text_width = int(0.8 * width)
    font_size = font_size

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Wrap the text to fit within the max_text_width
    lines = wrap_text(text, font, max_text_width)
    text_width, text_height = font.getsize(lines[0])
    text_height = text_height * len(lines)

    # Define bubble frame dimensions
    padding = 10
    bubble_width = text_width + 2 * padding
    bubble_height = text_height + 2 * padding

    # Create a new image for the bubble frame
    bubble = Image.new('RGBA', (bubble_width, bubble_height), (255, 255, 255, 150))

    # Draw the bubble frame on the new image
    draw = ImageDraw.Draw(bubble)
    draw.rectangle([(0, 0), (bubble_width - 1, bubble_height - 1)], outline=(0, 0, 0, 200), width=2)

    # Draw the wrapped text line by line
    y_text = padding
    for line in lines:
        draw.text((padding, y_text), line, font=font, fill=(50, 50, 50, 255))
        y_text += font.getsize(line)[1]

    # Calculate the bubble frame position
    x, y = point
    if x + bubble_width > width:
        x = width - bubble_width
    if y + bubble_height > height:
        y = height - bubble_height

    # Paste the bubble frame onto the image
    image.paste(bubble, (x, y), bubble)
    return image