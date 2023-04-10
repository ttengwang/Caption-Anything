from PIL import Image, ImageDraw, ImageFont

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

def create_bubble_frame(image_path, text, point, font_path='TlwgMono.ttf'):
    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    # Calculate max_text_width and font_size based on image dimensions and total number of characters
    total_chars = len(text)
    max_text_width = int(0.8 * width)
    font_size = 0.033 * height

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

    # Save the result
    image.save("result/output.png")
# from PIL import Image, ImageDraw, ImageFont

# def wrap_text(text, font, max_width):
#     lines = []
#     words = text.split(' ')
#     current_line = ''

#     for word in words:
#         if font.getsize(current_line + word)[0] <= max_width:
#             current_line += word + ' '
#         else:
#             lines.append(current_line)
#             current_line = word + ' '

#     lines.append(current_line)
#     return lines

# def create_speech_bubble(text, font, max_text_width,padding=10):
    
#     lines = wrap_text(text, font, max_text_width)
#     text_width, text_height = font.getsize(lines[0])
#     text_height = text_height * len(lines)

#     bubble_width = text_width + 2 * padding
#     bubble_height = text_height + 2 * padding
#     arrow_height = text_height // 3

#     bubble = Image.new('RGBA', (bubble_width, bubble_height + arrow_height), (255, 255, 255, 0))

#     draw = ImageDraw.Draw(bubble)
#     draw.rectangle([(0, arrow_height), (bubble_width - 1, bubble_height - 1 + arrow_height)], fill=(255, 255, 255, 150), outline=(0, 0, 0, 200), width=2)
#     draw.polygon([(0, arrow_height), (arrow_height, 0), (arrow_height, 2 * arrow_height)], fill=(255, 255, 255, 150), outline=(0, 0, 0, 200), width=2)

#     y_text = padding + arrow_height
#     for line in lines:
#         draw.text((padding, y_text), line, font=font, fill=(50, 50, 50, 255))
#         y_text += font.getsize(line)[1]

#     return bubble

# def create_bubble_frame(image_path, text, point, font_path='arial.ttf'):
#     # Load the image
#     image = Image.open(image_path)
#     width, height = image.size

#     # Calculate max_text_width and font_size based on image dimensions and total number of characters
#     total_chars = len(text)
#     max_text_width = int(0.8 * width)
#     font_size = int(height / (2 * (total_chars // 20 + 1)))

#     # Load the font
#     font = ImageFont.truetype(font_path, font_size)

#     # Create the speech bubble
#     speech_bubble = create_speech_bubble(text, font, max_text_width)

#     # Calculate the bubble frame position
#     x, y = point
#     bubble_width, bubble_height = speech_bubble.size
#     if x + bubble_width > width:
#         x = width - bubble_width
#     if y + bubble_height > height:
#         y = height - bubble_height

#     # Paste the speech bubble onto the image
#     image.paste(speech_bubble, (x, y), speech_bubble)

#     # Save the result
#     image.save("result/output.png")




if __name__ == "__main__":
    image_path = "test_img/img13.jpg"
    text = "raw_caption: there is a dog running in the grass with a frisbee in its mouth," + "caption: A playful dog joyfully carries a frisbee through the grass." +"wiki: Dog: A dog is a domesticated mammal and a common household pet. They are known for their loyalty and companionship to humans. Dogs come in a variety of breeds, each with their own unique characteristics and traits. They are highly intelligent and can be trained to perform a variety of tasks, such as assisting people with disabilities, herding livestock, and providing security. Dogs require regular exercise and proper nutrition to maintain their health and well-being. They are social animals and thrive on human interaction and attention. In the sentence provided, a dog is seen running in the grass with a frisbee in its mouth, which is a common activity for dogs who enjoy playing fetch."
    point = (100, 100)  # Coordinate point (x, y)
    font_path = "TlwgMono.ttf"  # Optional, you can use the default 'arial.ttf'
    create_bubble_frame(image_path, text, point, font_path)