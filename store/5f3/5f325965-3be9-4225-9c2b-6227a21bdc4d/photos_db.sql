ATTACH TABLE _ UUID '002920bf-8ac8-4f80-a9be-33f09a9d9d4d'
(
    `filename` String,
    `file` Nullable(String),
    `subfolder` Nullable(String),
    `some_column` Nullable(String),
    `height` Nullable(UInt32),
    `width` Nullable(UInt32),
    `location` Nullable(String),
    `text` Nullable(String),
    `embedding` Array(Float32),
    `path` Nullable(String)
)
ENGINE = MergeTree
ORDER BY filename
SETTINGS index_granularity = 8192
