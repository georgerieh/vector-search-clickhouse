'''Module for searching images'''
#!/usr/bin/python3
import os
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote
import time
from PIL import Image
import clickhouse_connect
import clip
import torch
from pyparsing import pyparsing_common


def _search(client, table, column, features, limit=10, filter=''):
    st = time.time()
    order = "ASC"
    columns = ['path', 'location', f'L2Distance({column},{features}) AS score', \
                column, 'lat', 'lon']
    filter_created = f"WHERE {filter}" if filter != '' else ""
    query = f'SELECT {",".join(columns)} FROM {table} \
        {filter_created} ORDER BY score {order} LIMIT {limit}'
    result = client.query(query)
    et = time.time()
    rows = [{
        'url': unquote(row[0]),
        'caption': row[1],
        'score': round(row[2], 3),
        'embedding': row[3],
        'location': [row[4], row[5]] 
    }
        for row in result.result_rows]
    return rows, {'read_rows': result.summary['read_rows'], 'query_time': round(et - st, 3)}

'''search with text'''
def search_with_text(client, model, table, text, limit):
    limit = limit if limit is not None else 50
    st = time.time()
    inputs = clip.tokenize(text)
    with torch.no_grad():
        text_features = model.encode_text(inputs)[0].tolist()
        et = time.time()
        rows, stats, = _search(client, table, 'embedding', text_features, limit=limit)
        stats['generation_time'] = round(et - st, 3)
        return rows, stats


def search_with_images(preprocess, device, client, model, table, image, limit):
    st = time.time()
    image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    print("preprocess successful")
    with torch.no_grad():
        image_features = model.encode_image(image)[0].tolist()
        print("features successful")
        et = time.time()
        print("limit", limit)
        rows, stats = _search(client, table, 'embedding', image_features, limit=limit)
        print("search successful")
        stats['generation_time'] = round(et - st, 3)
        return rows, stats


@dataclass
class Expression:
    text: str


@dataclass
class Feature:
    vector: List[float]


def return_file(search_parser, text, image, table, limit):
    client = clickhouse_connect.get_client(
        host=os.environ.get('CLICKHOUSE_HOST', 'localhost'),
        username=os.environ.get('CLICKHOUSE_USERNAME', 'default'),
        password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
        port=os.environ.get('CLICKHOUSE_PORT', 8123))

    print('Execution started')
    command = search_parser
    limit = limit if limit is not None else 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model, preprocess = clip.load("ViT-B/32")
    model.to(device)
    images = []
    stats = {}
    if command == 'search':
        if text:
            images, stats = search_with_text(client, model, table, text, limit=limit )
        elif image:
            images, stats = search_with_images(preprocess, \
                        device, client, model, table, image, limit=limit)

    output = {
    "images": images,
    "table": table,
    "search_text": text,
    "source_image": image,
    "gen_time": stats.get('generation_time', 0),
    "query_time": stats.get('query_time', 0),
}
    return output
