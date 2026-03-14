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


INSERT INTO agentic_synthesis_tasks (
    user_id,
    dataset_id,
    prompt_text,
    action_tags_json,
    llm_api_key,
    llm_base_url,
    llm_model_name,
    output_file_path,
    total_workspaces,
    processed_workspaces,
    started_time,
    finished_time,
    error_message
)
VALUES
(
    (SELECT id FROM users WHERE username = 'user' LIMIT 1),
    (SELECT id FROM datasets WHERE name = 'customers_2025' LIMIT 1),
    '为业务分析生成客户画像摘要与风险标签',
    '["summary", "classification", "risk"]',
    'demo-key-001',
    'https://api.openai.com/v1',
    'gpt-4.1-mini',
    'output/1/1/result.jsonl',
    2,
    2,
    '2026-03-07 10:00:00',
    '2026-03-07 10:12:34',
    NULL
),
(
    (SELECT id FROM users WHERE username = 'admin' LIMIT 1),
    (SELECT id FROM datasets WHERE name = 'sales_daily_2026_03' LIMIT 1),
    '对销售日报做异常检测并输出告警建议',
    '["anomaly-detection", "alerting"]',
    'demo-key-002',
    'https://api.openai.com/v1',
    'gpt-4.1',
    'output/2/2/result.jsonl',
    3,
    1,
    '2026-03-07 13:50:00',
    NULL,
    NULL
),
(
    (SELECT id FROM users WHERE username = 'admin' LIMIT 1),
    (SELECT id FROM datasets WHERE name = 'exp_logs_202603' LIMIT 1),
    '从实验日志中提取关键失败原因并归类',
    '["extraction", "root-cause-analysis"]',
    'demo-key-003',
    'https://api.openai.com/v1',
    'gpt-4o-mini',
    'output/2/3/result.jsonl',
    2,
    2,
    '2026-03-07 11:20:00',
    '2026-03-07 11:27:42',
    '部分日志文件编码异常，解析中断'
),
(
    (SELECT id FROM users WHERE username = 'user' LIMIT 1),
    (SELECT id FROM datasets WHERE name = 'transactions_q4' LIMIT 1),
    '整理多源数据并生成统一字段映射说明',
    '["schema-mapping", "documentation"]',
    'demo-key-004',
    'https://api.openai.com/v1',
    'gpt-4.1-mini',
    'output/1/4/result.jsonl',
    1,
    0,
    NULL,
    NULL,
    NULL
);


INSERT INTO agentic_synthesis_results (
    task_id,
    user_id,
    dataset_id,
    workspace_name,
    question,
    trajectory,
    evaluation_json,
    status,
    error_message
)
VALUES
(
    1,
    (SELECT id FROM users WHERE username = 'user' LIMIT 1),
    (SELECT id FROM datasets WHERE name = 'customers_2025' LIMIT 1),
    'workspace_a',
    'How many heads of the departments are older than 56 ?',
    '<Analyze>Fixed trajectory from result_example.</Analyze>',
    '{"difficulty":4,"quality":5,"ability":"Data Analysis"}',
    'completed',
    NULL
),
(
    1,
    (SELECT id FROM users WHERE username = 'user' LIMIT 1),
    (SELECT id FROM datasets WHERE name = 'customers_2025' LIMIT 1),
    'workspace_a',
    'List all department heads and summarize their age distribution.',
    '<Analyze>Fixed trajectory from result_example.</Analyze>',
    '{"difficulty":4,"quality":5,"ability":"Data Analysis"}',
    'completed',
    NULL
);