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
import pickle
def was_taken(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-Make', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.stdout.strip().split(':')[-1].strip() != ''
    except Exception:
        return False
    

batch_size = 64
embeddings_buffer = []
i = 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate',
        description='Generate CLIP embeddings for images or text')
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
        output_file_path = str(destination / 'output.jsonl')
        for subfolder in base_folder.iterdir():
            if subfolder.is_dir():
                for file in subfolder.iterdir():
                    if file.is_file():
                        if file.name.startswith('.') or not ('jpeg' in file.name or 'jpg' in file.name) or not was_taken(file):
                            continue
                        image = preprocess(Image.open(file)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = model.encode_image(image)[0].tolist()
                        entry = {'filename': f'{str(subfolder.name).replace(" ", "_")}/{str(file.name)}', 'embedding': embedding}
                        embeddings_buffer.append(entry)
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