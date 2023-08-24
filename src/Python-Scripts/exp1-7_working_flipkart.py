import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time
import json

PATH = "C:\\Custom\\PythonLearning\\chromedriver.exe"
START_INDEX = 91166

def save_metadata(metadata, file_path):
    with open(file_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item))
            f.write('\n')

def get_images_from_flipkart(wd, delay, max_images):
    def scroll_down(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    url = "https://www.flipkart.com/clothing-and-accessories/clothing-accessories/pr?sid=clo%2Cqd8&q=pagdi+for+men&otracker=categorytree&p%5B%5D=facets.ideal_for%255B%255D%3DMen&page=1"
    wd.get(url)

    image_urls = set()
    images_collected = 0

    while images_collected < max_images:
        scroll_down(wd)

        thumbnails = wd.find_elements(By.CLASS_NAME, "_2r_T1I")

        for img in thumbnails:
            try:
                img.click()
                time.sleep(delay)
                # Switch to the new window
                wd.switch_to.window(wd.window_handles[-1])
            except:
                continue

            images = wd.find_elements(By.CLASS_NAME, "_396cs4._2amPTt._3qGmMb")
            for image in images:
                if image.get_attribute('src') in image_urls:
                    continue

                if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                    image_urls.add(image.get_attribute('src'))
                    images_collected += 1
                    print(f"Found {images_collected}/{max_images} images")

                    if images_collected >= max_images:
                        break

                    metadata.append({
                        "image_url": image.get_attribute('src'),
                        "image_path": f"images/train/{START_INDEX + images_collected - 1}.jpeg",
                        "brand": "N/A",
                        "product_title": "N/A",
                        "class_label": "men_pagdi"
                    })

            # Close the current window
            wd.close()
            # Switch back to the original window
            wd.switch_to.window(wd.window_handles[0])

        # Check if "Next" button is available and click it
        try:
            next_button = wd.find_element(By.XPATH, "//a[@class='_1LKTO3']/span[contains(text(), 'Next')]")
            if not next_button.is_enabled():
                break

            next_button.click()
            time.sleep(delay)
        except:
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
urls = get_images_from_flipkart(wd, 1, 650)

for i, url in enumerate(urls):
    download_image("images/train/", url["image_url"], str(START_INDEX + i) + ".jpg")

save_metadata(metadata, "metadata.json")

wd.quit()
