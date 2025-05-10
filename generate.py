#!/usr/bin/python3
import argparse
import time

from PIL import Image
import clip
import torch
import os
import shutil
from pathlib import Path
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate',
        description='Generate CLIP embeddings for images or text')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', required=False)
    group.add_argument('--image', required=False)
    group.add_argument('--file', required=False)
    group.add_argument('--directory', required=False)
    group.add_argument('--directory_out', required=False)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    device = torch.device(device)
    model, preprocess = clip.load("ViT-L/14", device=device)
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
    elif args.directory():
        base_folder = Path(args.directory)
        destination = Path(args.directory_out)
        destination.mkdir(exist_ok=True)
        with open('output.txt', 'w') as output_file:
            for subfolder in base_folder.iterdir():
                if subfolder.is_dir():
                    for file in subfolder.iterdir():
                        if file.is_file():
                            try:
                                if file.name.startswith('.') or not (file.name.contains('JPEG') and file.name.contains('jpg')):
                                    continue
                                image = preprocess(Image.open(file)).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    output_file.write(f'{file.name}: {model.encode_image(image)[0].tolist()}\n')
                            except Exception as e:
                                print(e)
        