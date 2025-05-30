import os
import json
import requests
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np


class SegmentsAI:
    def __init__(self, segments_data_json_path, labels_json_path, base_images_folder=None):
        with open(segments_data_json_path, 'r') as json_segment_data:
            self.mask_data = json.load(json_segment_data)
        self.base_images_folder = base_images_folder
        with open(labels_json_path, "r") as json_labels:
            labels = json.load(json_labels)
        self.labels = labels
        self.categories_ids = {}
        for category in self.mask_data['dataset']['task_attributes']['categories']:
            self.categories_ids[category['id']] = {'name': category['name'], 'count': 0}
        samples = []
        for sample in self.mask_data['dataset']['samples']:
            samples.append({
                'base_img': sample['name'],
                'base_img_url': sample['attributes']['image']['url'],
                'url': sample['labels']['ground-truth']['attributes']['segmentation_bitmap']['url'],
                'category_id': sample['labels']['ground-truth']['attributes']['annotations'][0]['category_id'],
                'nb_of_masks': len(sample['labels']['ground-truth']['attributes']['annotations'])
            })
        self.samples = samples

    def pull_cut_paste_learn_data(self):
        self.pull_base_imgs()
        self.pull_masks()
        self.create_json_splits()

    def get_img_from_api(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return ImageOps.exif_transpose(Image.open(BytesIO(response.content)).convert("RGBA"))
        else:
            return None

    def create_json_splits(self):
        with open(Path(__file__).parent / "data_set" / "splits.json", "w") as json_file:
            split_dict = []
            for i, sample in enumerate(self.samples):
                category_name = self.categories_ids[sample['category_id']]['name']
                split_dict.append({
                    'base_img': f'{category_name}/{sample["base_img_name"]}',
                    'mask_img': [],
                    'initial_img_size': sample['base_img_size'],
                    'nb_masks': sample['nb_of_masks']
                    })
                for j in range(sample['nb_of_masks']):
                    split_dict[i]['mask_img'].append(f'{category_name}/{sample["mask_img_name"]}_{j}.png')
            json.dump(split_dict, json_file, indent=4)

    def pull_base_imgs(self):
        for sample in self.samples:
            category_name = self.categories_ids[sample['category_id']]['name']
            new_path = Path(__file__).parent / 'data_set' / 'base' / category_name
            sample['base_img_name'] = f'{category_name}_paste_img_{self.categories_ids[sample["category_id"]]["count"]}.png'
            sample['mask_img_name'] = f'{category_name}_paste_img_{self.categories_ids[sample["category_id"]]["count"]}_label_ground_truth'
            if self.base_images_folder is not None:
                base_image_path = Path(self.base_images_folder) / category_name / sample["base_img"]
                self.categories_ids[sample['category_id']]['count'] += 1
                if Path(base_image_path).exists():
                    base_image = ImageOps.exif_transpose(Image.open(base_image_path).convert("RGBA"))
                else:
                    base_image = self.get_img_from_api(sample['base_img_url'])
            else:
                base_image = self.get_img_from_api(sample['base_img_url'])
            os.makedirs(new_path, exist_ok=True)
            sample['base_img_size'] = base_image.size
            base_image.save(f'{new_path}/{sample["base_img_name"]}')

    def pull_masks(self):
        fails = []
        for sample in self.samples:
            category_name = self.categories_ids[sample['category_id']]['name']
            new_path = Path(__file__).parent / "data_set" / "mask" / category_name
            response = requests.get(sample['url'])
            if response.status_code == 200:
                os.makedirs(new_path, exist_ok=True)
                mask = Image.open(BytesIO(response.content))
                for i in range(sample['nb_of_masks']):
                    mask_np = np.array(mask)
                    check_array = np.array([i + 1, 0, 0, 255])
                    affected_pixels_mask = (mask_np == check_array).all(axis=-1)
                    affected_pixel_indexes = np.argwhere(affected_pixels_mask)
                    mask_np[~affected_pixels_mask] = [0, 0, 0, 0]
                    if i != 0:
                        mask_np[(mask_np == check_array).all(axis=-1)] = [1, 0, 0, 255]
                    cut_mask = self.mask_to_LA(mask_np, category_name)
                    cut_mask.save(f'{new_path}/{sample["mask_img_name"]}_{i}.png')
            else:
                fails.append(sample['mask_img_name'])
        return fails

    def mask_to_LA(self, mask_np, category_name):
        la_mask_np = np.full((mask_np.shape[0], mask_np.shape[1], 2), [1, 0], dtype=np.uint8)
        mask_position = (mask_np == [1, 0, 0, 255]).all(axis=-1)
        label_id = self.labels[category_name]
        la_mask_np[mask_position] = [label_id, 255]
        la_img = Image.fromarray(la_mask_np, mode="LA")
        return la_img
