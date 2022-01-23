import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

import shutil

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    file_path = os.path.abspath(os.path.dirname(__file__))
    files = os.listdir(data_dir)
    random.shuffle(files)

    n = len(files)

    for file in files[:int(n*0.7)]:
        src = f"{file_path}/data/waymo/training_and_validation/{file}"
        dst = f"{file_path}/data/train/{file}"
        shutil.move(src, dst)

    for file in files[int(n*0.7):int(n*0.85)]:
        src = f"{file_path}/data/waymo/training_and_validation/{file}"
        dst = f"{file_path}/data/val/{file}"
        shutil.move(src, dst)

        
    for file in files[int(n*0.85):]:
        src = f"{file_path}/data/waymo/training_and_validation/{file}"
        dst = f"{file_path}/data/test/{file}"
        shutil.move(src, dst)

        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)