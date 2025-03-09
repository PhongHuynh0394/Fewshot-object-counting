import numpy as np
from matplotlib import cm
from PIL import Image, ImageOps


def draw_density_overlay(image: Image, density_map: Image, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay the density map onto the original image using a fixed color.

    Args:
        image (PIL.Image): Original input image.
        density_map (PIL.Image): Predicted density map (grayscale).
        alpha (float): Transparency level for the density map overlay.
        color (tuple): RGB color to use for the density map.

    Returns:
        PIL.Image: The overlayed image.
    """
    # Ensure the density map is grayscale
    density_map = density_map.convert("L")

    color_layer = ImageOps.colorize(density_map, black="black", white=color)

    density_map_color = color_layer.resize(image.size, Image.LANCZOS)

    overlay_image = Image.blend(image.convert("RGB"), density_map_color, alpha=alpha)

    return overlay_image, density_map_color
