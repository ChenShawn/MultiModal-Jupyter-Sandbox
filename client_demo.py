import random
import requests
import base64
from PIL import Image
from io import BytesIO

code_1 = """
from PIL import Image
import pytesseract
import cv2
import easyocr
import matplotlib.pyplot as plt

x = [1,2,3]
y = [4,8,6]
print(f' [DEBUG] {x=}')

plt.plot(x, y)
plt.title('Debug Plot')
plt.show()

import time
while True:
    time.sleep(1)
"""

code_2 = """
import numpy as np
x = np.array(x)
y = np.array(y)
z = x + y
print(' [DEBUG 123]')
plt.plot(x, z)
plt.title('Debug Plot 2')
plt.show()

print(' [DEBUG 234] {z=}')

plt.plot(x, -z)
plt.title('DEBUG PLOT 3')
plt.show()

print(' [DEBUG 345]')
"""

code_3 = """
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('highlighted_space.jpg').convert('RGB')
img_crop = img.crop((0, 0, 400, 600))  # Crop the image to a 100x100 square
plt.imshow(img_crop)
plt.axis('off')
plt.show()

# 中文注释
print('打印中文输出')

img_crop2 = img.crop((400, 600, 800, 1200))  # Crop another part of the image
plt.imshow(img_crop2)
plt.axis('off')
plt.show()
"""

INITIALIZATION_CODE_TEMPLATE = """
from PIL import Image
import base64
from io import BytesIO

_img_base64 = "{base64_image}"
image = Image.open(BytesIO(base64.b64decode(_img_base64)))

# dsadsfarg

import matplotlib.pyplot as plt
plt.imshow(image)
plt.title('中文测试')
plt.axis('off')
plt.show()
"""

def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


test_sid = 'debug_jupyter_250805'
test_timeout = 5

res1 = requests.post(
    'http://127.0.0.1:18902/jupyter_sandbox',
    json={
        "session_id": test_sid,
        "code": code_1,
        "timeout": test_timeout,
    }
).json()

print(res1)
exit()

target_image = Image.open('highlighted_space.jpg').convert('RGB')
target_image_base64 = image_to_base64(target_image)
code_string = INITIALIZATION_CODE_TEMPLATE.format(base64_image=target_image_base64)
res2 = requests.post(
    # 'http://10.39.10.230:12345/jupyter_sandbox',
    'http://127.0.0.1:18902/jupyter_sandbox',
    json={
        "session_id": test_sid,
        "code": code_string,
        "timeout": test_timeout,
    }
).json()
print(f' [DEBUG #222] {res2.keys()=}')
print(f' [DEBUG #222] {res2["status"]=}')
print(f' [DEBUG #222] {res2["execution_time"]=}')
result_dict = res2['output']
# for k, v in result_dict.items():
#     print(f' [DEBUG #222] {k=}, {len(v)=}')

print(f' [stdout] {result_dict["stdout"]=}')
print(f' [stderr] {result_dict["stderr"]=}')
print(f' [images] {len(result_dict["images"])=}')

for idx, img in enumerate(result_dict['images']):
    img_pil = base64_to_image(img)
    img_pil.save(f'./debug_output/{test_sid}-{idx}.png', format='PNG')

print(' Done!!')

