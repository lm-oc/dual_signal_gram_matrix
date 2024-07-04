from collections import defaultdict
import glob
import hashlib
import os
import random
import requests
import shutil
import sys
import time

#import cv2
from skimage import io

# Mapping: (class, Name, groups)
STYLE_MAPPING = [
    ('Bokeh', 'Bokeh', ['1543486@N25']),
    ('Bright', 'Bright', ['799643@N24']),
    ('Depth_of_Field', 'Depth_of_Field', ['75418467@N00', '407825@N20']),
    ('Detailed', 'Detailed', ['1670588@N24', '1131378@N23']),
    ('Ethereal', 'Ethereal', ['907784@N22']),
    ('Geometric_Composition', 'Geometric_Composition', ['46353124@N00']),
    ('Hazy', 'Hazy', ['38694591@N00']),
    ('HDR', 'HDR', ['99275357@N00']),
    ('Horror', 'Horror', ['29561404@N00']),
    ('Long_Exposure', 'Long_Exposure', ['52240257802@N01']),
    ('Macro', 'Macro', ['52241335207@N01']),
    ('Melancholy', 'Melancholy', ['70495179@N00']),
    ('Minimal', 'Minimal', ['42097308@N00']),
    ('Noir', 'Noir', ['42109523@N00']),
    ('Romantic', 'Romantic', ['54284561@N00']),
    ('Serene', 'Serene', ['1081625@N25']),
    ('Pastel', 'Pastel', ['1055565@N24', '1371818@N25']),
    ('Sunny', 'Sunny', ['1242213@N23']),
    ('Texture', 'Texture', ['70176273@N00']),
    ('Vintage', 'Vintage', ['1222306@N25', "1176551@N24"]),
]


def main():
    if len(sys.argv) != 5:
        print('Usage: python assemble_data.py image_path train_file test_file images_per_style')
        return

    image_path = os.path.abspath(sys.argv[1])
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    images_per_style = int(sys.argv[4])

    url_file = os.path.join(os.path.dirname(__file__), 'flickr_style_url.txt')
    img_info_file = os.path.join(os.path.dirname(__file__), 'flickr_style_img_info.txt')

    collect_image_style_url(url_file, images_per_style)
    fetch_images(url_file, img_info_file, image_path)
    generate_train_test_dataset(img_info_file, train_file, test_file, train_ratio=0.8)


def collect_image_style_url(url_file, photos_per_style):
    if os.path.exists(url_file):
        print('[Skip] Url file exists: {}'.format(url_file))
        return

    with open(url_file, 'w') as f:
        for class_id, style, groups in STYLE_MAPPING:
            print('Get_photos_for_style: {}'.format(style))
            urls = get_image_url_from_group(groups, photos_per_style)
            for url in urls:
                print('{} {}'.format(url, class_id), file=f)

    print('[Done] Url file saves to: {}'.format(url_file))


# def get_image_url_from_group(groups, num_images):
#     params = {
#         'api_key': "6a92790b1c2a301c6e7ddef645dca1f53ea97ac2",
#         'per_page': 500,  # 500 is the maximum allowed
#         'content_type': 1,  # only photos
#     }
#
#     image_urls = []
#     for page in range(10):
#         params['page'] = page
#
#         for group in groups:
#             params['group_id'] = group
#
#             url = ('https://api.flickr.com/services/rest/?'
#                    'method=flickr.photos.search&format=json&nojsoncallback=1'
#                    '&api_key={api_key}&content_type={content_type}'
#                    '&group_id={group_id}&page={page}&per_page={per_page}')
#             url = url.format(**params)
#
#             # Make the request and ensure it succeeds.
#             try:
#                 page_data = requests.get(url).json()
#             except:
#                 print(requests.get(url))
#                 raise
#             if page_data['stat'] != 'ok':
#                 raise Exception("Something is wrong: API returned {}".format(page_data['stat']))
#
#             for photo_item in page_data['photos']['photo']:
#                 image_urls.append(_get_image_url(photo_item))
#
#             if len(image_urls) >= num_images:
#                 return image_urls[:num_images]
#
#     raise Exception('Not enough images, only find {}'.format(len(image_urls)))

# def get_image_url_from_group(groups, num_images):
#     params = {
#         'api_key': "d31c7cb60c57aa7483c5c80919df5371",
#         'per_page': 500,  # 500 is the maximum allowed
#         'content_type': 1,  # only photos
#     }
#
#     image_urls = []
#     # Calculate the number of pages needed
#     pages_needed = num_images // 500 + (1 if num_images % 500 else 0)
#     for page in range(pages_needed):
#         params['page'] = page + 1  # Flickr pages start at 1
#
#         for group in groups:
#             params['group_id'] = group
#
#             url = ('https://api.flickr.com/services/rest/?'
#                    'method=flickr.photos.search&format=json&nojsoncallback=1'
#                    '&api_key={api_key}&content_type={content_type}'
#                    '&group_id={group_id}&page={page}&per_page={per_page}')
#             url = url.format(**params)
#
#             # Make the request and ensure it succeeds.
#             try:
#                 page_data = requests.get(url).json()
#             except:
#                 print(requests.get(url))
#                 raise
#             if page_data['stat'] != 'ok':
#                 raise Exception("Something is wrong: API returned {}".format(page_data['stat']))
#
#             for photo_item in page_data['photos']['photo']:
#                 image_urls.append(_get_image_url(photo_item))
#                 # Stop if we have collected enough images
#                 if len(image_urls) >= num_images:
#                     return image_urls[:num_images]
#
#     # If we get here, we didn't find enough images
#     raise Exception('Not enough images, only found {}'.format(len(image_urls)))

import requests

def _get_image_url(photo_item):
    # 这里假设 _get_image_url 是一个辅助函数，用于从photo_item中提取图片URL
    # 请确保这个函数的实现逻辑符合您的需求
    return "http://farm{farm}.staticflickr.com/{server}/{id}_{secret}{size}.jpg".format(**photo_item)

def get_image_url_from_group(groups, num_images):
    params = {
        'api_key': "d31c7cb60c57aa7483c5c80919df5371",
        'per_page': 500,  # 500 is the maximum allowed
        'content_type': 1,  # only photos
    }

    image_urls = []

    # Calculate the number of pages needed for each group
    pages_needed = num_images // 500 + (1 if num_images % 500 else 0)

    for group in groups:
        for page in range(pages_needed):
            params['page'] = page + 1  # Flickr pages start at 1
            params['group_id'] = group

            url = ('https://api.flickr.com/services/rest/?'
                   'method=flickr.photos.search&format=json&nojsoncallback=1'
                   '&api_key={api_key}&content_type={content_type}'
                   '&group_id={group_id}&page={page}&per_page={per_page}')
            url = url.format(**params)

            # Make the request and ensure it succeeds.
            try:
                page_data = requests.get(url).json()
            except Exception as e:
                print(f"Request failed: {e}")
                continue
            if page_data['stat'] != 'ok':
                print(f"API returned an error: {page_data['stat']}")
                continue

            for photo_item in page_data['photos']['photo']:
                image_urls.append(_get_image_url(photo_item))
                # Stop if we have collected enough images
                if len(image_urls) >= num_images:
                    return image_urls

    # If we get here, we didn't find enough images
    with open('flickr_style_url.txt', 'a') as file:
        for url in image_urls:
            file.write(url + "\n")

    print(f'Not enough images, only found {len(image_urls)}. URLs have been written to flickr_style_url.txt')
    return image_urls

# Example of how to call the function
# Replace 'your_groups_list' with the actual list of group IDs you want to search through
# image_urls = get_image_url_from_group(your_groups_list, 4000)

def _get_image_url(photo_item, size_flag=''):
    """
    size_flag: string ['']
        See http://www.flickr.com/services/api/misc.urls.html for options.
            '': 500 px on longest side
            '_m': 240px on longest side
    """
    url = "http://farm{farm}.staticflickr.com/{server}/{id}_{secret}{size}.jpg"
    return url.format(size=size_flag, **photo_item)


def fetch_images(url_file, img_info_file, image_folder):
    if os.path.exists(img_info_file):
        print('[Skip] Image info file exists: {}'.format(img_info_file))
        return

    os.makedirs(image_folder, exist_ok=True)

    with open(url_file, 'r') as f:
        lines = [line.strip() for line in f]

    image_info = []
    for line in lines:
        url, class_id = line.strip().split()
        image_name = _get_image_name(url, class_id)
        image_file = os.path.join(image_folder, image_name)

        # Download and verify
        if not os.path.exists(image_file):
            res = download_image(url, image_file)
        res = verify_image(image_file)

        if not res:
            print('[FAILURE] {}'.format(url))
        else:
            image_info.append((image_file, class_id))
            print('[SUCCESS] {}'.format(url))

    with open(img_info_file, 'w') as f:
        for image_file, class_id in image_info:
            print('{} {}'.format(image_file, class_id), file=f)

    print('Success: {}, Failure: {}'.format(len(image_info), len(lines) - len(image_info)))
    print('[Done] Image info file saves to: {}'.format(img_info_file))


def _get_image_name(url, class_id):
    return '{}_{}.jpg'.format(hashlib.sha1(url.encode()).hexdigest(), class_id)


def download_image(url, file):
    try:
        if os.path.exists(file):
            return True

        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(file, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
                return True
        else:
            return False
    except KeyboardInterrupt:
        raise Exception()  # multiprocessing doesn't catch keyboard exceptions
    except:
        return False


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True


def generate_train_test_dataset(img_info_file, train_file, test_file, train_ratio=0.8):
    class_to_images = defaultdict(list)
    with open(img_info_file, 'r') as f:
        lines = [line.strip() for line in f]

    random.seed(1211)
    random.shuffle(lines)
    train_size = int(len(lines) * train_ratio)

    with open(train_file, 'w') as f:
        for line in lines[:train_size]:
            print(line, file=f)

    with open(test_file, 'w') as f:
        for line in lines[train_size:]:
            print(line, file=f)

    print('[Done] Test file (size={}) saves to: {}'.format(train_size, train_file))
    print('[Done] Train file (size={}) saves to: {}'.format(len(lines) - train_size, test_file))


if __name__ == '__main__':
    main()