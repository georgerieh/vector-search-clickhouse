#!/usr/bin/python3
import argparse
import time
import json
import subprocess
from PIL import Image
import clip
import torch
import os
import shutil
from pathlib import Path
import subprocess
import numpy as np
data = []
import pytesseract
def was_taken(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-Make', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.stdout.strip().split(':')[-1].strip() 
    except Exception:
        return False
    
def created(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-CreateDate', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.stdout.strip().split(' ')[0].strip() 
    except Exception:
        return False
def height(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-ExifImageHeight', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.stdout.strip().split(':')[-1].strip() 
    except Exception:
        return False
def width(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-ExifImageWidth', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.stdout.strip().split(':')[-1].strip() 
    except Exception:
        return False
    
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_location(file_path):
    try:
        x = subprocess.run(
            ['exiftool', '-GPSLongitude', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        x = x.stdout.strip().split(':')[-1].strip()
        x_degrees = dms_to_decimal(float(x.split()[0]), float(x.split()[2].replace("'", '')), float(x.split()[3].replace('"', '')), x.split()[4])
        y = subprocess.run(
            ['exiftool', '-GPSLatitude', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        y = y.stdout.strip().split(':')[-1].strip()
        y_degrees = dms_to_decimal(float(y.split()[0]), float(y.split()[2].replace("'", '')), float(y.split()[3].replace('"', '')), y.split()[4])


        return json.dumps({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [x_degrees, y_degrees]
            }
        })
    except Exception:
        x = subprocess.run(
            ['exiftool', '-GPSLongitude', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        x = x.stdout.strip().split(':')[-1].strip()
        print(file_path, x)
        return ""
def get_text_from_image(file_path):
    try:
        text = pytesseract.image_to_string(Image.open(file_path), lang='eng+rus')
        return text
    except Exception as e:
        print(f'{e}, {file_path}')
        return ""
batch_size = 64
embeddings_buffer = []
i = 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate',
        description='Generate metadata')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', required=False)
    group.add_argument('--image', required=False)
    group.add_argument('--file', required=False)
    group.add_argument('--directory', required=False)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    device = torch.device(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    images = []
    if args.text:
        inputs = clip.tokenize(args.text).to(device)
        with torch.no_grad():
            print(model.encode_text(inputs)[0].tolist())
    elif args.image:
        image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
        with torch.no_grad():
            print(model.encode_image(image)[0].tolist())
    elif args.file:
        with open(args.file, 'r') as input_file, torch.no_grad(), open('output.txt', 'w') as output_file:
            st = time.time()
            c = 0
            for line in input_file:
                c += 1
                inputs = clip.tokenize(line).to(device)
                embedding = model.encode_text(inputs)[0].tolist()
                output_file.write(f'{embedding}\n')
            et = time.time()
            print(f'{c} embeddings generated in {round(et-st, 3)}s')
    elif args.directory:
        base_folder = Path(args.directory)
        destination = Path(str(args.directory) + '-out')
        destination.mkdir(exist_ok=True)
        output_file_path = str(destination / 'metadata.jsonl')
        for subfolder in base_folder.iterdir():
            if subfolder.is_dir():
                for file in subfolder.iterdir():
                    if file.is_file():
                        if file.name.startswith('.') or not ('jpeg' in file.name or 'jpg' in file.name) or not was_taken(file):
                            continue
                        embeddings_buffer.append([f'{str(subfolder.name).replace(" ", "_")}/{str(file.name)}', file.name, subfolder.name, created(file), height(file), width(file), get_location(file), ' '.join(get_text_from_image(file).split())])
                        # embeddings_buffer.append([f'{str(subfolder.name).replace(" ", "_")}/{str(file.name)}', file.name, subfolder.name, created(file)])
                        
                        if len(embeddings_buffer) >= batch_size:
                            with open(output_file_path, 'a') as outfile:
                                for entry in embeddings_buffer:
                                    json.dump(entry, outfile)
                                    if i % 50 == 0:
                                        print(f'Processed {i} images')
                                        print(f'Last entry: {entry}')
                                    i += 1
                                    outfile.write('\n')
                            embeddings_buffer = []