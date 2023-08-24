import os
import requests
import io
from PIL import Image
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from urllib.parse import urlparse

PATH = "C:\\Custom\\PythonLearning\\chromedriver.exe"
START_INDEX = 139000
MAX_IMAGES = 500

def save_metadata(metadata, file_path):
    with open(file_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item))
            f.write('\n')

def get_images_from_utsavfashion(wd, delay):
    def scroll_down(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    url = "https://www.google.com/search?q=A-Line+kurta&tbm=isch&ved=2ahUKEwiOvKbOzNyAAxX4XUEAHTArCaQQ2-cCegQIABAA&oq=A-Line+kurta&gs_lcp=CgNpbWcQAzIFCAAQgAQyBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB46BAgjECc6CAgAEIAEELEDOgUIABCxAzoHCAAQigUQQ1DsFFiLUGC1VGgAcAB4AIABbIgBgAaSAQQxMi4xmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=IVraZM6nHvi7hbIPsNakoAo&bih=739&biw=1536&rlz=1C1ONGR_enIE1057IE1060"
    wd.get(url)

    image_urls = set()
    metadata = []

    while len(metadata) < MAX_IMAGES:
        scroll_down(wd)

        products = wd.find_elements(By.CLASS_NAME, "Q4LuWd")

        for i, product in enumerate(products):
            try:
                # Get the image URL
                WebDriverWait(wd, 10).until(EC.visibility_of(product))
                image_url = product.get_attribute("src")

                if image_url in image_urls:
                    continue

                if image_url and 'http' in image_url:
                    # Validate the image URL
                    response = requests.head(image_url)
                    if response.status_code != 200:
                        continue

                    image_urls.add(image_url)
                    metadata.append({
                        "image_url": image_url,
                        "image_path": f"images/train/{START_INDEX + len(metadata)}.jpg",
                        "brand": "N/A",
                        "product_title": "N/A",
                        "class_label": "women_a-line_kurta"
                    })
                    print(f"Found {len(metadata)}/{MAX_IMAGES} images")

                    if len(metadata) >= MAX_IMAGES:
                        break
            except Exception as e:
                print("Error:", str(e))
                continue

        if len(metadata) >= MAX_IMAGES:
            break

    return metadata

def download_image(download_path, url, file_name):
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc == "":
            # Relative URL, prepend the base URL
            base_url = "https://www.google.co.in/"
            url = f"{base_url}{url}"
        
        response = requests.get(url, timeout=10)  # Set timeout for the request
        response.raise_for_status()  # Raise an exception if the request was unsuccessful
        image_content = response.content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = os.path.join(download_path, file_name)  # Use os.path.join to create the full file path

        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        print("Success")
    except (requests.exceptions.RequestException, IOError) as e:
        print('FAILED -', e)

wd = webdriver.Chrome(PATH)
metadata = get_images_from_utsavfashion(wd, 1)

for i, url in enumerate(metadata):
    download_image("images/aline/", url["image_url"], str(START_INDEX + i) + ".jpg")

save_metadata(metadata, "metadata.json")

wd.quit()
