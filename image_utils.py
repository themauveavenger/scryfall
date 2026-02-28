from io import BytesIO
from PIL import Image, ImageEnhance
from requests import get

def store_upscaled_image(image_url: str, image_path: str):
    resp = get(image_url)

    with Image.open(BytesIO(resp.content)) as img:
        new_size = (img.width * 4, img.height * 4)

        upscaled = img.resize(new_size, resample=Image.Resampling.LANCZOS)

        color_enhance = ImageEnhance.Color(upscaled)
        upscaled = color_enhance.enhance(1.2)

        sharpness_enhance = ImageEnhance.Sharpness(upscaled)
        upscaled = sharpness_enhance.enhance(1.2)

        upscaled.save(image_path)

def upscale_image(image_path: str, new_name: str, size_multiplier = 4):
    with Image.open(image_path) as img:
        new_size = (img.width * size_multiplier, img.height * size_multiplier)

        upscaled = img.resize(new_size, resample=Image.Resampling.LANCZOS)

        color_enhance = ImageEnhance.Color(upscaled)
        upscaled = color_enhance.enhance(1.2)

        sharpness_enhance = ImageEnhance.Sharpness(upscaled)
        upscaled = sharpness_enhance.enhance(1.2)

        upscaled.save(new_name)

