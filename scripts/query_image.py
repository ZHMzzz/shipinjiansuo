"""Decode the visible image, including orientation and transparent backgrounds."""
import io
from PIL import Image, ImageOps


def decode_query_image(raw):
    with Image.open(io.BytesIO(raw)) as source:
        source = ImageOps.exif_transpose(source)
        if 'A' in source.getbands() or 'transparency' in source.info:
            rgba = source.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            background.alpha_composite(rgba)
            image = background.convert('RGB')
        else:
            image = source.convert('RGB')
    image.thumbnail((1280, 1280))
    return image
