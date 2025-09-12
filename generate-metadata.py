#!/usr/bin/python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import clip
import torch
from PIL import Image

data = []
import pytesseract
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def gps_to_decimal(gps_str):
    try:
        parts = gps_str.split()
        deg = float(parts[0])
        minutes = float(parts[2].replace("'", ""))
        seconds = float(parts[3].replace('"', ""))
        direction = parts[4]
        return dms_to_decimal(deg, minutes, seconds, direction)
    except Exception:
        return ''

def get_location(exif_data):
    if "GPSLongitude" in exif_data and "GPSLatitude" in exif_data:
        lon = gps_to_decimal(exif_data["GPSLongitude"])
        lat = gps_to_decimal(exif_data["GPSLatitude"])
        if lon is not None and lat is not None:
            return [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}}, lat, lon]
    return ['', 0.0, 0.0]

def get_text_from_image(file_path):
    try:
        return ' '.join(pytesseract.image_to_string(Image.open(file_path), lang='eng+rus').split())
    except Exception:
        return ""

def parse_exiftool_json(json_data):
    for item in json_data:
        filename = os.path.basename(item.get("SourceFile", ""))
        if filename.startswith("."):
            continue  # skip dot-files

        path = Path(item.get("SourceFile"))
        subfolder = path.parent.name
        file_name = path.name

        location, lat, lon = get_location(item)
        created_date = item.get("CreateDate", "").split(" ")[0] if "CreateDate" in item else ''
        height = item.get("ExifImageHeight")
        width = item.get("ExifImageWidth")

        row = [
            str(path.relative_to(path.parents[1])).replace(" ", "_"),  # relative path with subfolder
            file_name,
            subfolder,
            created_date,
            height,
            width,
            json.dumps(location) if location else '',
            # get_text_from_image(path),
            '',
            lat,
            lon
        ]
        yield row

def run_exiftool(directory):
    output_file = os.path.join(directory, 'output.json')
    with open(output_file, "w") as f:
        subprocess.run([
            "exiftool",
            "-r",  # recursive
            "-Make",
            "-CreateDate",
            "-ExifImageHeight",
            "-ExifImageWidth",
            "-GPSLongitude",
            "-GPSLatitude",
            "-j",  # JSON output
            directory
        ], stdout=f)

def reorganize_to_jsonl(json_input, jsonl_output):
    with open(json_input) as f, open(jsonl_output, "w") as out_f:
        data = json.load(f)
        for row in parse_exiftool_json(data):
            out_f.write(json.dumps(row) + "\n")
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
    # model, preprocess = clip.load("ViT-B/32", device=device)
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
        run_exiftool(directory=args.directory)
        reorganize_to_jsonl(args.directory + '/output.json', args.directory + '-out/metadata.jsonl')