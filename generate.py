#!/usr/bin/python3
import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from tqdm import tqdm

# -----------------------
# CPU-only device
# -----------------------
device = torch.device("cpu")

# -----------------------
# Models
# -----------------------
# DINO: pretrained ViT-Small
dino_model = timm.create_model(
    "vit_base_patch16_224.dino", pretrained=True, num_classes=0
)
dino_model.eval().to(device)
dino_model.to(device)

# FaceNet
mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Preprocessing
dino_preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


# -----------------------
# Helper function
# -----------------------
def normalize_vector(v):
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()


def process_image(file_path):
    """Compute DINO and FaceNet embeddings for a single image"""
    entry = {"filename": str(file_path), "faces": []}

    # --- DINO embedding ---
    img = Image.open(file_path).convert("RGB")
    tensor = dino_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino_model.forward_features(tensor)
        if feats.ndim == 3:
            dino_feat = feats[:, 0, :]  # CLS token only, shape: [1, 768]
        elif feats.ndim == 2:
            dino_feat = feats  # already [1, 768]
        dino_feat = dino_feat.squeeze(0)  # final shape: [768]

    entry["dino_embedding"] = normalize_vector(dino_feat.cpu().numpy())

    # --- Face detection + FaceNet ---
    boxes, probs = mtcnn.detect(img)
    faces = mtcnn(img)  # aligned face tensors
    if faces is not None and boxes is not None:
        for face_tensor, prob in zip(faces, probs):
            if prob is None or prob < 0.90:  # skip low-confidence
                continue
            with torch.no_grad():
                face_feat = facenet_model(face_tensor.unsqueeze(0).to(device))
            entry["faces"].append(
                {
                    "confidence": float(prob),
                    "embedding": normalize_vector(face_feat[0].cpu().numpy()),
                }
            )
    return entry


# -----------------------
# Main script
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        help="Path to image folder",
        default="/Volumes/T7/photos_from_icloud",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output", default="/Volumes/T7/photos_from_icloud-out/embeddings_new.jsonl"
    )
    args = parser.parse_args()

    base_folder = Path(args.directory)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    buffer = []
    i = 0
    processed_files = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_files.add(obj["filename"])
                except:
                    continue
    files_to_process = []
    for subfolder in base_folder.iterdir():
        if subfolder.is_dir():
            for file in subfolder.iterdir():
                if (
                        not file.is_file()
                        or not file.suffix.lower() in [".jpg", ".jpeg"]
                        or file.name.startswith(".")
                ):
                    continue
                elif str(file) in processed_files:
                    continue
                files_to_process.append(file)
    for file in tqdm(files_to_process, desc="Processing images"):
        try:
            entry = process_image(file)
            buffer.append(entry)
            i += 1
        except Exception as e:
            print(f"Skipping {file}: {e}")

        if len(buffer) >= args.batch_size:
            with open(output_path, "a") as f:
                for e in buffer:
                    f.write(json.dumps(e) + "\n")
            buffer = []

    # Write remaining entries
    if buffer:
        with open(output_path, "a") as f:
            for e in buffer:
                f.write(json.dumps(e) + "\n")
        print(f"Processed total {i} images")
