import json
import multiprocessing as mp
import time
from functools import partial
import pandas as pd
import numpy as np
from pathlib import Path
import os
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
i=0
import pandas as pd
import json
data =[]
with open ('/Volumes/T7/photos_from_icloud-out/output.jsonl') as f:
    for line in f:
        data.append(json.loads(line).values())
df=pd.DataFrame(data, columns=['filename', 'embedding']).set_index('filename').to_json()
dff=json.loads(df)

def gps_lat(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-GPSLatitude', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.stdout.strip().split(':')[-1].strip()
    except Exception:
        return False
def gps_lon(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-GPSLongitude', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.stdout.strip().split(':')[-1].strip()
    except Exception:
        return False
    

def process_file(output_directory, paths):
    try:
        start = time.time()
        id = paths[0]
        print(f'Processing {id}')
        metadata_file = paths[1]
        text_npy = paths[3]
        batch_size = 6553
        print(f'Loading npy files for {paths[0]}')
        im_emb = dff['embedding']['id']
        parquet_schema = pa.schema([('key', pa.string()),
                                    ('url', pa.string()),
                                    ('caption', pa.string()),
                                    ('similarity', pa.float64()),
                                    ('width', pa.int64()),
                                    ('height', pa.int64()),
                                    ('status', pa.string()),
                                    ('NSFW', pa.string()),
                                    ('exif', pa.map_(pa.string(), pa.string())),
                                    ('text_embedding', pa.list_(pa.float64())),
                                    ('image_embedding', pa.list_(pa.float64()))])

        with pq.ParquetWriter(os.path.join(output_directory, id + '.parquet'), parquet_schema,
                              compression='zstd') as writer:
            i = 0
            c = 0
            parquet_file = pq.ParquetFile(metadata_file)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                print(f'Handling batch {c} for {paths[0]}')
                meta_chunk = batch.to_pandas()
                if im_emb is None:
                    data = pd.concat([meta_chunk, pd.DataFrame({'image_embedding': [[0.0] * 768] * len(meta_chunk)})],
                                     axis=1, copy=False)
                else:
                    data = pd.concat([meta_chunk, pd.DataFrame({'image_embedding': [*im_emb[i:i + len(meta_chunk)]]})],
                                     axis=1, copy=False)
                    data['image_embedding'] = [row.tolist() for row in data['image_embedding']]
                if text_emb is None:
                    data = pd.concat([data, pd.DataFrame({'text_embedding': [[0.0] * 768] * len(meta_chunk)})],
                                     axis=1, copy=False)
                else:
                    data = pd.concat([data, pd.DataFrame({'text_embedding': [*text_emb[i:i + len(meta_chunk)]]})],
                                     axis=1, copy=False)
                    data['text_embedding'] = [row.tolist() for row in data['text_embedding']]
                i = i + len(meta_chunk)
                # you can save more columns
                data = data[list(parquet_schema.names)]
                data['caption'] = [row.replace(''', ' ').replace(''', ' ') for row in data['caption']]
                data['exif'] = [json.loads(row) if row is not None else {} for row in data['exif']]
                table = pa.Table.from_pandas(data, schema=parquet_schema)
                writer.write_table(table)
                print(f'{i} rows exported for {id}')
                end = time.time()
                c = c + 1
            print(f'{id} took {end - start}s')
            return {
                'id': id,
                'success': True
            }
    except Exception as e:
        return {
            'id': id,
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='process',
        description='Convert laion metadata and embeddings into Parquet')
    parser.add_argument('--input_folder', required=False, default='')
    parser.add_argument('--output_folder', required=False, default='./laion_combined')
    parser.add_argument('--processes', required=False, default=mp.cpu_count(), type=int)
    args = parser.parse_args()
    mp.freeze_support()
    input_directory = args.input_folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    processes = mp.cpu_count()
    processes = args.processes
    meta_paths = Path(os.path.join(input_directory, 'metadata')).glob('metadata_*.parquet')
    ids = [os.path.splitext(os.path.basename(metadata_file))[0].split('_')[1] for metadata_file in meta_paths]
    ids = sorted(ids, key=lambda x: int(x))
    jobs = [[id, os.path.join(input_directory, 'metadata', 'metadata_' + id + '.parquet'),
             os.path.join(input_directory, 'img_emb', 'img_emb_' + id + '.npy'),
             os.path.join(input_directory, 'text_emb', 'text_emb_' + id + '.npy')] for id in ids]
    with mp.Pool(processes=processes) as pool:
        func = partial(process_file, args.output_folder)
        results = pool.map(func, jobs, chunksize=1)
        for result in results:
            if not result['success']:
                print(f'job {result["id"]} failed with {result["error"]}')
        print('Complete')
