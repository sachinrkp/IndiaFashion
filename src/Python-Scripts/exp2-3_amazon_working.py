import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time
import json

PATH = "C:\\Custom\\PythonLearning\\chromedriver.exe"
START_INDEX = 91563

def save_metadata(metadata, file_path):
    with open(file_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item))
            f.write('\n')

def get_images_from_amazon(wd, delay, max_images):
    def scroll_down(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    url = "https://www.amazon.in/s?k=Ethnic+pagdi+for+groom&crid=2EJUJRORVYPQC&sprefix=ethnic+pagdi+for+groom%2Caps%2C179&ref=nb_sb_noss"
    wd.get(url)

    image_urls = set()
    images_collected = 0
    images_downloaded = 0

    # Get the session ID of the original window
    original_window = wd.current_window_handle

    while images_collected < max_images:
        scroll_down(wd)

        thumbnails = wd.find_elements(By.CSS_SELECTOR, ".s-image")

        for i, img in enumerate(thumbnails):
            try:
                img.click()
                time.sleep(delay)
                # Switch to the new window
                wd.switch_to.window(wd.window_handles[-1])
            except Exception as e:
                print("Error switching to new window:", str(e))
                continue

            images = wd.find_elements(By.CSS_SELECTOR, ".a-dynamic-image.a-stretch-horizontal")
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

            wd.close()  # Close the current window

            # Switch back to the original window using the session ID
            try:
                wd.switch_to.window(original_window)
            except Exception as e:
                print("Error switching back to the original window:", str(e))
                break

        if images_collected >= max_images or images_collected >= 56:
            break

        # Stop loading more images if no new images have been collected
        if images_downloaded == images_collected:
            print("No more images to load.")
            break
        else:
            images_downloaded = images_collected

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
urls = get_images_from_amazon(wd, 1, 56)

for i, url in enumerate(urls):
    download_image("images/train/", url["image_url"], str(START_INDEX + i) + ".jpg")
    #metadata.append(url)

save_metadata(metadata, "metadata.json")

wd.quit()
