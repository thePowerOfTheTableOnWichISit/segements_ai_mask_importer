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
            annotation_ids = []
            if sample['labels']['ground-truth'] is not None and len(sample['labels']['ground-truth']['attributes']['annotations']) > 0:
                for annotation in sample['labels']['ground-truth']['attributes']['annotations']:
                    annotation_ids.append(annotation['id'])

                samples.append({
                    'base_img': sample['name'],
                    'base_img_url': sample['attributes']['image']['url'],
                    'url': sample['labels']['ground-truth']['attributes']['segmentation_bitmap']['url'],
                    'category_id': sample['labels']['ground-truth']['attributes']['annotations'][0]['category_id'],
                    'nb_of_masks': len(sample['labels']['ground-truth']['attributes']['annotations']),
                    'category_name': self.categories_ids[sample['labels']['ground-truth']['attributes']['annotations'][0]['category_id']]['name'],
                    'annotation_ids': annotation_ids,
                    'inflation': sample['labels']['ground-truth']['attributes']['image_attributes']['Inflation']
                })
        self.samples = samples
        self.split_dict = []

    def pull_cut_paste_learn_data(self):
        fails = []
        for sample in self.samples:
            try:
                self.pull_sample(sample)
            except Exception as e:
                fails.append(sample)
                raise e
        self.create_json_splits()
        print(fails)

    def get_img_from_api(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return ImageOps.exif_transpose(Image.open(BytesIO(response.content)).convert("RGBA"))
        else:
            return None

    def create_json_splits(self):
        with open(Path(__file__).parent / "data_set" / "splits.json", "w") as json_file:
            json.dump(self.split_dict, json_file, indent=4)

    def pull_base_img(self, sample):
        if self.base_images_folder is not None:
            base_image_path = Path(self.base_images_folder) / sample["category_name"] / sample["base_img"]
            if Path(base_image_path).exists():
                base_image = ImageOps.exif_transpose(Image.open(base_image_path).convert("RGBA"))
            else:
                base_image = self.get_img_from_api(sample['base_img_url'])
        else:
            base_image = self.get_img_from_api(sample['base_img_url'])
        return base_image

    def pull_mask(self, sample):
        response = requests.get(sample['url'])
        if response.status_code == 200:
            mask = Image.open(BytesIO(response.content))
            return mask
        else:
            raise Exception("Oupsies")

    def pull_sample(self, sample):
        fails = []
        sample["category_name"] = self.categories_ids[sample['category_id']]['name']
        new_paths = {
            'base': Path(__file__).parent / 'data_set' / 'base' / sample["category_name"],
            'mask': Path(__file__).parent / "data_set" / "mask" / sample["category_name"]
        }
        sample['base_img_name'] = f'{sample["category_name"]}_paste_img_{self.categories_ids[sample["category_id"]]["count"]}'
        sample['mask_img_name'] = f'{sample["category_name"]}_paste_img_{self.categories_ids[sample["category_id"]]["count"]}_label_ground_truth'
        self.categories_ids[sample['category_id']]['count'] += 1

        base_image = self.pull_base_img(sample)
        mask = self.pull_mask(sample)
        if base_image.size != mask.size:
            raise Exception("AhoUGA")
        sample['base_img_size'] = base_image.size
        finals = self.put_for_integration(sample, base_image, mask)

        os.makedirs(new_paths["base"], exist_ok=True)
        os.makedirs(new_paths["mask"], exist_ok=True)
        for i, final in enumerate(finals):
            final['cut_mask'].save(f'{new_paths["mask"] / sample["mask_img_name"]}_{i}.png')
            final['base_image'].save(f'{new_paths["base"] / sample["base_img_name"]}_{i}.png')
            self.split_dict.append({
                'base_img': f'{Path(sample["category_name"]) / sample["base_img_name"]}_{i}.png',
                'mask_img': f'{Path(sample["category_name"]) / sample["mask_img_name"]}_{i}.png',
                'initial_img_size': sample['base_img_size'],
                'cut_size': final['base_image'].size,
                'pixel_occupation': sample['pixel_occupation'],
                'inflation': sample['inflation']
            })

    def put_for_integration(self, sample, img, mask):
        return_list = []

        for i in range(sample['nb_of_masks']):
            mask_np_temp = np.array(mask)
            img_np_temp = np.array(img)
            check_array = np.array([sample['annotation_ids'][i], 0, 0, 255])

            affected_pixels_mask = (mask_np_temp == check_array).all(axis=-1)
            mask_np_temp[~affected_pixels_mask] = [0, 0, 0, 0]
            img_np_temp[~affected_pixels_mask] = [0, 0, 0, 0]
            if i != 0:
                mask_np_temp[(mask_np_temp == check_array).all(axis=-1)] = [1, 0, 0, 255]

            affected_pixel_indexes = np.argwhere(affected_pixels_mask)
            sample['pixel_occupation'] = len(affected_pixel_indexes)
            if len(affected_pixel_indexes) != 0:
                pixel_vip = {
                    'top_left': (affected_pixel_indexes[0][0], np.min(affected_pixel_indexes[:, 1])),
                    'bottom_right': (affected_pixel_indexes[-1][0], np.max(affected_pixel_indexes[:, 1]))
                }
                if pixel_vip['top_left'][0] < pixel_vip['bottom_right'][0] and pixel_vip['top_left'][1] < pixel_vip['bottom_right'][1]:
                    img_np_temp = img_np_temp[pixel_vip['top_left'][0]:pixel_vip['bottom_right'][0] + 1, pixel_vip['top_left'][1]:pixel_vip['bottom_right'][1] + 1]
                    mask_np_temp = mask_np_temp[pixel_vip['top_left'][0]:pixel_vip['bottom_right'][0] + 1, pixel_vip['top_left'][1]:pixel_vip['bottom_right'][1] + 1]
                else:
                    print("arg")

            cut_mask = self.mask_to_LA(mask_np_temp, sample["category_name"])
            return_list.append({'cut_mask': cut_mask, 'base_image': Image.fromarray(img_np_temp, mode="RGBA")})

        return return_list

    def mask_to_LA(self, mask_np, category_name):
        la_mask_np = np.full((mask_np.shape[0], mask_np.shape[1], 2), [1, 0], dtype=np.uint8)
        mask_position = (mask_np == [1, 0, 0, 255]).all(axis=-1)
        label_id = self.labels[category_name]
        la_mask_np[mask_position] = [label_id, 255]
        la_img = Image.fromarray(la_mask_np, mode="LA")
        return la_img
