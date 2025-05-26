import os
import requests
import shutil
import json
from PIL import Image
from io import BytesIO


BASE_IMAGE_FOLDER = '/home/norlab/Documents/picture_editing/plants/base'
def labels_to_dict(labels):
    return_dict = {}
    for label in labels:
        return_dict[label['id']] = {'name': label['name'], 'count': 0}
    return return_dict

def samples_to_dict(samples):
    masks = []
    for sample in samples:
        masks.append({
            'base_img': sample['name'],
            'base_img_url': sample['attributes']['image']['url'],
            'url': sample['labels']['ground-truth']['attributes']['segmentation_bitmap']['url'],
            'category_id': sample['labels']['ground-truth']['attributes']['annotations'][0]['category_id'],
            'nb_of_masks': len(sample['labels']['ground-truth']['attributes']['annotations'])
        })
    return masks

def pull_base_imgs(samples, category_ids):
    for sample in samples:
        category_name = category_ids[sample['category_id']]['name']
        base_image_path = os.path.join(BASE_IMAGE_FOLDER, f'{category_name}/{sample["base_img"]}')
        new_path = os.path.join(os.path.dirname(__file__), f'data_set/base/{category_name}')
        if not os.path.exists(f'{new_path}/{sample["base_img"]}'):
            sample['base_img_name'] = f'{category_name}_paste_img_{category_ids[sample["category_id"]]["count"]}.png'
            sample['mask_img_name'] = f'{category_name}_paste_img_{category_ids[sample["category_id"]]["count"]}_label_ground_truth.png'
            category_ids[sample['category_id']]['count'] += 1
            if os.path.exists(base_image_path):
                os.makedirs(new_path, exist_ok=True)
                base_image = Image.open(base_image_path).convert("RGBA")
                base_image.save(f'{new_path}/{sample["base_img_name"]}')
            else:
                response = requests.get(sample['base_img_url'])
                if response.status_code == 200:
                    os.makedirs(new_path, exist_ok=True)
                    base_image = Image.open(BytesIO(response.content)).convert("RGBA")
                    base_image.save(f'{new_path}/{sample["base_img_name"]}')
                else:
                    print(f"L'image {sample['base_img']} n'a pas été trouvé et n'a pas pu être downloadé")

def pull_masks(samples, category_ids):
    fails = []
    for sample in samples:
        category_name = category_ids[sample['category_id']]['name']
        new_path = os.path.join(os.path.dirname(__file__), f'data_set/mask/{category_name}')
        response = requests.get(sample['url'])
        if response.status_code == 200:
            for i in range(sample['nb_of_masks']):
                os.makedirs(new_path, exist_ok=True)
                mask = Image.open(BytesIO(response.content))
                mask.save(f'{new_path}/{sample["mask_img_name"]}')
        else:
            fails.append(sample['mask_img_name'])
    return fails

def create_splits(samples):
    with open(os.path.join(os.path.dirname(__file__), f'data_set/splits.txt'), "w") as f:
        for sample in samples:
            f.write(f'{sample["base_img_name"]} {sample["mask_img_name"]}' + "\n")


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), 'Releases/masques_de_plantes-arggg.json')
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    category_ids = labels_to_dict(data['dataset']['task_attributes']['categories'])
    samples = samples_to_dict(data['dataset']['samples'])
    pull_base_imgs(samples, category_ids)
    pull_masks(samples, category_ids)
    create_splits(samples)





