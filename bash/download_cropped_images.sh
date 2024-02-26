#!/bin/bash
set -e

echo "Downloading cropped images"
mkdir -p /ssd/dylu/data/arctic/downloads/data/cropped_images_zips
python scripts_data/download_data.py --url_file ./bash/assets/urls/cropped_images.txt --out_folder /ssd/dylu/data/arctic/downloads/data/cropped_images_zips
