#!/usr/bin/python3
import os
import time
from urllib.parse import unquote

import clickhouse_connect
import numpy as np
import timm
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms

# -------------------
# CPU-only setup
# -------------------
device = torch.device("cpu")

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


# -------------------
# Utilities
# -------------------
def normalize_vector(v):
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()


def extract_dino(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = dino_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino_model.forward_features(tensor)
        if feats.ndim == 3:
            cls_embedding = feats[:, 0, :]  # CLS token
        else:
            cls_embedding = feats
        cls_embedding = cls_embedding.squeeze(0)
    return normalize_vector(cls_embedding.cpu().numpy())


def extract_faces(image_path):
    img = Image.open(image_path).convert("RGB")
    boxes, probs = mtcnn.detect(img)
    embeddings = []
    if boxes is not None:
        faces = mtcnn.extract(img, boxes, save_path=None)
        for face_tensor, prob in zip(faces, probs):
            if prob < 0.9:
                continue
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = facenet_model(face_tensor)
            embeddings.append(normalize_vector(feat[0].cpu().numpy()))


def to_clickhouse_array(vec):
    return "[" + ",".join(f"{x:.16f}" for x in vec) + "]"


def list_of_lists_to_clickhouse_array(lst_of_lsts):
    inner_arrays = [
        "[" + ",".join(f"{x:.16f}" for x in inner) + "]" for inner in lst_of_lsts
    ]
    return "[" + ",".join(inner_arrays) + "]"


def _search(
        client,
        dino_query,
        facenet_query,
        limit=10,
        filter_expr="",
        start_date="",
        end_date="",
):
    print("_searching")
    seen = set()
    rows = []
    start = time.time()
    if not start_date:
        ignore = False
        st = time.time()
        dino_query = dino_query or [0.0] * 768
        if facenet_query is None or len(facenet_query) == 0:
            facenet_query = [[0.0] * 512]  # placeholder
        elif all(isinstance(x, float) for x in facenet_query):
            # single face vector → wrap it in a list
            facenet_query = [facenet_query]
        else:
            # assume list of vectors
            facenet_query = [
                vec if len(vec) == 512 else [0.0] * 512 for vec in facenet_query
            ]
        if facenet_query == [[0.0] * 512]:
            ignore = True
        dino_query_str = to_clickhouse_array(dino_query)
        facenet_query_str = list_of_lists_to_clickhouse_array(facenet_query)
        # AND length(dino_embedding) = 768 AND length(facenet_embedding) = 512
        filter_created = (
            f"WHERE {filter_expr} AND length(facenet_embedding) == 512 AND length(dino_embedding) == 768"
            if filter_expr
            else f"WHERE length(facenet_embedding) == 512 AND length(dino_embedding) == 768 AND score > 1"
        )
        print(len(dino_query))
        print(len(facenet_query[0]))
        query = f"""
        WITH 
            {dino_query_str} AS dino_query_array,
            {facenet_query_str} AS facenet_query_array
        SELECT path, location, subfolder, filename, MIN(total_score) AS total_score, lat, lon
            FROM (
                SELECT 
                    path, location, subfolder, filename,
                    L2Distance(dino_embedding, dino_query_array) +
                    arrayMin(arrayMap(fq -> L2Distance(facenet_embedding, fq), facenet_query_array)) AS total_score, lat, lon
                FROM photos_db
                WHERE length(dino_embedding)=768 AND length(facenet_embedding)=512
            )
        GROUP BY path, location, subfolder, filename, lat, lon
        ORDER BY total_score ASC
        LIMIT 50
        """
        print(query)
        result = client.query(query)
        for row in result.result_rows:
            if row[0] not in seen:
                seen.add(row[0])
                rows.append(
                    {
                        # 'url': unquote(row[2] + "/" + row[3])
                        "location": row[1],
                        "url": unquote(row[0]).replace(
                            "/Volumes/T7/photos_from_icloud/", ""
                        ),
                        "score": round(row[4], 3),
                        "lat": row[5],
                        "lon": row[6],
                        # 'embedding': row[7],
                        # 'location': [row[3], row[4]],
                        "timestamp": int(os.path.getmtime(row[0])),
                    }
                )

        et = time.time()
        return rows, {"query_time": round(et - st, 3)}
    elif start_date and not end_date:
        limit = limit if limit else 1
        st = time.time()
        query = f"""
                SELECT path, location, subfolder, filename, lat, lon, date
                    FROM (
                        SELECT 
                            path, location, subfolder, filename, lat, lon, date
                        FROM photos_db
                    )
                WHERE toDate(date) BETWEEN toDate('{start_date.replace("-", ":")}') 
                AND toDate('{start_date.replace("-", ":")}') + INTERVAL {limit if isinstance(limit, int) else 1} DAY
                ORDER BY path ASC
                """
        print(query)
        result = client.query(query)
        et = time.time()
        for row in result.result_rows:
            if row[0] not in seen:
                seen.add(row[0])
                rows.append(
                    {
                        # 'url': unquote(row[2] + "/" + row[3])
                        "location": row[1],
                        "url": unquote(row[0]).replace(
                            "/Volumes/T7/photos_from_icloud/", ""
                        ),
                        "score": round(row[4], 3),
                        "lat": row[4],
                        "lon": row[5],
                        # 'embedding': row[7],
                        # 'location': [row[3], row[4]],
                        "timestamp": int(os.path.getmtime(row[0])),
                    }
                )
        return rows, {"query_time": round(et - st, 3)}
    elif start_date and end_date:
        st = time.time()
        query = f"""
                        SELECT path, location, subfolder, filename, lat, lon, date
                            FROM (
                                SELECT 
                                    path, location, subfolder, filename, lat, lon, date
                                FROM photos_db
                            )
                        WHERE toDate(date) BETWEEN toDate('{start_date.replace("-", ":")}') 
                        AND toDate('{end_date.replace("-", ":")}')
                        ORDER BY path ASC
                        """
        print(query)
        result = client.query(query)
        et = time.time()
        for row in result.result_rows:
            if row[0] not in seen:
                seen.add(row[0])
                rows.append(
                    {
                        # 'url': unquote(row[2] + "/" + row[3])
                        "location": row[1],
                        "url": unquote(row[0]).replace(
                            "/Volumes/T7/photos_from_icloud/", ""
                        ),
                        "score": round(row[4], 3),
                        "lat": row[4],
                        "lon": row[5],
                        # 'embedding': row[7],
                        # 'location': [row[3], row[4]],
                        "timestamp": int(os.path.getmtime(row[0])),
                    }
                )
        return rows, {"query_time": round(et - st, 3)}
    else:
        return [
            {
                # 'url': unquote(row[2] + "/" + row[3])
                "location": "",
                "url": "",
                "score": "",
                "lat": "",
                "lon": "",
                # 'embedding': row[7],
                # 'location': [row[3], row[4]],
                "timestamp": "",
            }
        ]


def search_with_images(
        client, image, limit, filter_expr="", start_date="", end_date="", use_dino_extract=1
):
    print("searching with image")
    if use_dino_extract:
        st = time.time()
        with torch.no_grad():
            dino_features = (
                extract_dino(image)
                if isinstance(extract_dino(image), (list, np.ndarray))
                else [0.0] * 768
            )
            facenet_features = (
                extract_faces(image)
                if isinstance(extract_faces(image), (list, np.ndarray))
                else [0.0] * 512
            )
            et = time.time()
            print("limit", limit)
            rows, stats = _search(
                client,
                dino_features,
                facenet_features,
                limit=limit,
                filter_expr=filter_expr,
                start_date=start_date,
                end_date=end_date,
            )
            print("search successful")
            stats["generation_time"] = round(et - st, 3)
            return rows, stats
    else:
        st = time.time()
        with torch.no_grad():
            dino_features = [0.0] * 768
            facenet_features = [0.0] * 512
            et = time.time()
            print("limit", limit)
            rows, stats = _search(
                client,
                dino_features,
                facenet_features,
                limit=limit,
                filter_expr=filter_expr,
                start_date=start_date,
                end_date=end_date,
            )
            print("search successful")
            stats["generation_time"] = round(et - st, 3)
            return rows, stats


def return_file(
        search_parser, text, image, table, limit, filter_expr="", start_date="", end_date=""
):
    client = clickhouse_connect.get_client(
        host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
        username=os.environ.get("CLICKHOUSE_USERNAME", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        port=os.environ.get("CLICKHOUSE_PORT", 8123),
    )

    print("Execution started")
    command = search_parser
    limit = limit if limit is not None else 50
    images = []
    stats = {}
    if command == "search":
        if image:
            images, stats = search_with_images(
                client,
                image,
                limit=limit,
                filter_expr="",
                start_date="",
                end_date="",
                use_dino_extract=1,
            )
        else:
            images, stats = search_with_images(
                client,
                image,
                limit=limit,
                filter_expr="",
                start_date=start_date,
                end_date=end_date,
                use_dino_extract=0,
            )
    print("image", image)
    output = {
        "images": images if isinstance(images, list) else [],
        "table": table,
        "search_text": text,
        "source_image": unquote(image).replace("/Volumes/T7/photos_from_icloud/", ""),
        "gen_time": stats.get("generation_time", 0),
        "query_time": stats.get("query_time", 0),
        "start_date": start_date if start_date is not None else "",
    }
    return output
