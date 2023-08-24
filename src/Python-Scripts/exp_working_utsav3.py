import os
import requests
import io
from PIL import Image
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By

PATH = "C:\\Custom\\PythonLearning\\chromedriver.exe"
START_INDEX = 9191

def save_metadata(metadata, file_path):
    with open(file_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item))
            f.write('\n')

def get_images_from_utsavfashion(wd, delay, max_images):
    def scroll_down(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    url = "https://www.utsavfashion.com/men/turban?p=7"
    wd.get(url)

    image_urls = set()
    images_collected = 0

    while images_collected < max_images:
        scroll_down(wd)

        products = wd.find_elements(By.CLASS_NAME, "img-responsive.product-image-photo")

        for i, product in enumerate(products):
            try:
                # Get the image URL
                image_url = product.get_attribute("src")

                if image_url in image_urls:
                    continue

                if image_url and 'http' in image_url:
                    image_urls.add(image_url)
                    images_collected += 1
                    print(f"Found {images_collected}/{max_images} images")

                    if images_collected >= max_images:
                        break

                    metadata.append({
                        "image_url": image_url,
                        "image_path": f"images/train/{START_INDEX + images_collected - 1}.jpeg",
                        "brand": "N/A",
                        "product_title": "N/A",
                        "class_label": "men_pagdi"
                    })
            except Exception as e:
                print("Error:", str(e))
                continue

        if images_collected >= max_images:
            break

    return metadata[:max_images]

def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = os.path.join(download_path, file_name)  # Use os.path.join to create the full file path

        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        print("Success")
    except Exception as e:
        print('FAILED -', e)

wd = webdriver.Chrome(PATH)
metadata = []
urls = get_images_from_utsavfashion(wd, 1, 24)

for i, url in enumerate(urls):
    download_image("images/train/", url["image_url"], str(START_INDEX + i) + ".jpg")

save_metadata(metadata, "metadata.json")

wd.quit()
