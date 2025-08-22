import pandas as pd
import clickhouse_connect


import json
data =[]
with open ('/Volumes/T7/photos_from_icloud-out/metadata.jsonl') as f:
    for line in f:
        data.append(json.loads(line))
df1=pd.DataFrame(data, columns=['filename', 'file', 'subfolder','date', 'height', 'width','location','text'])
data =[]
with open ('/Volumes/T7/photos_from_icloud-out/output.jsonl') as f:
    for line in f:
             data.append(json.loads(line).values())
df2=pd.DataFrame(data, columns=['filename', 'embedding'])

df_combined = pd.concat([df1, df2], axis=1)
df_combined['path'] = '/Volumes/T7/photos_from_icloud/' + df_combined['subfolder']
df_combined['path'] = df_combined['path'] + '/' + df_combined['file']
df = df_combined
df = df.loc[:,~df.columns.duplicated()].copy()

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