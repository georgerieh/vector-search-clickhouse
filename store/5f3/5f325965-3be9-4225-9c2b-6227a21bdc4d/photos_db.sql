ATTACH TABLE _ UUID 'e01411d2-8bf0-4e24-bab6-09509c4ac972'
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
