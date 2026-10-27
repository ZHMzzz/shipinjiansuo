import io
from PIL import Image
from scripts.query_image import decode_query_image


def test_hidden_rgb_pixels_do_not_become_visible_background():
    image=Image.new('RGBA',(20,20),(71,112,76,0));image.putpixel((10,10),(4,5,6,255))
    data=io.BytesIO();image.save(data,'PNG');result=decode_query_image(data.getvalue())
    assert result.getpixel((0,0))==(255,255,255)
    assert result.getpixel((10,10))==(4,5,6)


def test_opaque_pixels_preserved():
    data=io.BytesIO();Image.new('RGB',(20,20),(18,22,44)).save(data,'PNG')
    assert decode_query_image(data.getvalue()).getpixel((0,0))==(18,22,44)
