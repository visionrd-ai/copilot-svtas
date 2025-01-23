from PIL import Image, ImageDraw, ImageFont
import os
import colorsys

def generate_colors(num_colors):
    """Generate `num_colors` distinct colors using HSL color space."""
    hues = [i / num_colors for i in range(num_colors)]  # Evenly spaced hues
    colors = [colorsys.hls_to_rgb(h, 0.5, 1.0) for h in hues]  # Full saturation and lightness
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

def create_a4_colored_pngs(output_dir, num_images):
    """Create A4-sized PNGs with distinct colors."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # A4 size in pixels at 300 DPI (2480 x 3508)
    a4_width, a4_height = 2480, 3508

    # Generate distinct colors
    colors = generate_colors(num_images)
    import cv2 
    # Load a default font
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=128)
    for i, color in enumerate(colors):
        # Create an A4 image with the specified color
        img = Image.new("RGB", (a4_width, a4_height), color)

        # Draw the index on the image for identification
        draw = ImageDraw.Draw(img)
        text = f"Branch {i}"
        text_size =[draw.textlength(text, font=font)]*2  # Measure text size with the specified font
        draw.text(
            ((a4_width - text_size[0]) / 2, (a4_height - text_size[1]) / 2),
            text,
            fill=(0, 0, 0),
            font=font,
        )

        # Save the image as a PNG
        img.save(os.path.join(output_dir, f"colored_image_{i + 1}.png"))

# Specify output directory and number of images
output_directory = "a4_colored_images"
number_of_images = 16

# Generate the images
create_a4_colored_pngs(output_directory, number_of_images)
