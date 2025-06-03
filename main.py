import os
from pathlib import Path
from segments_ai import SegmentsAI


BASE_IMAGE_FOLDER = Path('/home/norlab/Documents/picture_editing/plants/base')

if __name__ == "__main__":
    file_path = Path(__file__).parent / 'Releases' / 'masques_de_plantes-v0.0.1.json'

    segments_ai_manager = SegmentsAI(file_path, Path(__file__).parent / 'label_to_id.json', BASE_IMAGE_FOLDER)
    segments_ai_manager.pull_cut_paste_learn_data()
