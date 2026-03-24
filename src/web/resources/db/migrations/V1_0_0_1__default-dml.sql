INSERT OR IGNORE INTO users (username, password, role)
VALUES ('user', 'user', 'user');

INSERT OR IGNORE INTO users (username, password, role)
VALUES ('admin', 'admin', 'admin');


INSERT INTO datasets (
    user_id,
    name,
    type,
    source,
    language,
    size,
    status,
    note,
    file_name,
    file_path,
    sample_data,
    cover_path
)
VALUES
(
    (SELECT id FROM users WHERE username = 'user' LIMIT 1),
    'customers_2025',
    'instruction',
    'crm',
    'multi',
    10240,
    'uploaded',
    '客户主数据样例',
    'customers_2025.csv',
    '/abs/path/uploads/1/1',
    '[{"id":"C001","name":"Alice"}]',
    NULL
),
(
    (SELECT id FROM users WHERE username = 'user' LIMIT 1),
    'transactions_q4',
    'instruction',
    'payment',
    'multi',
    20480,
    'uploaded',
    '交易流水样例',
    'transactions_q4.parquet',
    '/abs/path/uploads/1/2',
    '[{"tx_id":"T001","amount":100}]',
    NULL
),
(
    (SELECT id FROM users WHERE username = 'admin' LIMIT 1),
    'sales_daily_2026_03',
    'instruction',
    'sales',
    'zh',
    5120,
    'uploaded',
    '销售日报样例',
    'sales_daily_2026_03.csv',
    '/abs/path/uploads/2/3',
    '[{"date":"2026-03-01","sales":12345}]',
    NULL
),
(
    (SELECT id FROM users WHERE username = 'admin' LIMIT 1),
    'exp_logs_202603',
    'evaluation',
    'lab',
    'en',
    8192,
    'uploaded',
    '实验日志样例',
    'exp_logs_202603.jsonl',
    '/abs/path/uploads/2/4',
    '[{"exp":"e1","status":"failed"}]',
    NULL
);


