import pandas as pd
import clickhouse_connect
import numpy as np

import json
data =[]
with open ('/Volumes/T7/photos_from_icloud-out/metadata.jsonl') as f:
    for line in f:
        data.append(json.loads(line))
df1 = pd.DataFrame(data, columns=[
    'path', 'filename', 'subfolder', 'date', 'height', 'width', 'location', 'text', 'lat', 'lon'
])
data =[]
with open ('/Volumes/T7/photos_from_icloud-out/output.jsonl') as f:
    for line in f:
             data.append(json.loads(line).values())
df2=pd.DataFrame(data, columns=['filename', 'embedding'])

df_combined = pd.concat([df1, df2], axis=1)
df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
df_combined['paths'] = '/Volumes/T7/photos_from_icloud/' + df_combined['subfolder']
df_combined['path'] = df_combined['paths'] + '/' + df_combined['filename']
df = df_combined
df = df.loc[:,~df.columns.duplicated()].copy()
df = df[[i for i in df.columns if i != 'paths']]
df['embedding'] = df['embedding'].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
df['location'] = df['location'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else '')
df['text'] = df['text'].fillna('').astype(str)
df['height'] = df['height'].fillna(0).astype('UInt32')
df['width'] = df['width'].fillna(0).astype('UInt32')
df['lat'] = df.get('lat', 0.0).astype('float32')
df['lon'] = df.get('lon', 0.0).astype('float32')
df['path'] = df['path'].fillna('').astype(str)
df['filename'] = df['filename'].fillna('').astype(str)
df['file'] = df['file'].fillna('').astype(str)
df['subfolder'] = df['subfolder'].fillna('').astype(str)
df['date'] = df['date'].fillna('').astype(str)
client = clickhouse_connect.get_client(host='localhost', port=8123, username='default', password='')
client.command('''DROP TABLE IF EXISTS photos_db;''')
client.command('''
CREATE TABLE photos_db (
    filename String,
    file Nullable(String),
    subfolder Nullable(String),
    date Nullable(String),
    height Nullable(UInt32),
    width Nullable(UInt32),
    location Nullable(String), 
    text Nullable(String),
    embedding Array(Float32),
    path Nullable(String),
    lat Nullable(Float32),
    lon Nullable(Float32)
) ENGINE = MergeTree()
ORDER BY filename;
''')


client.insert_df('photos_db', df)

print("Data uploaded successfully!")