ALTER TABLE datasets ADD COLUMN origin_stage TEXT NULL;
ALTER TABLE datasets ADD COLUMN origin_dataset_id INTEGER NULL;
ALTER TABLE datasets ADD COLUMN origin_task_type TEXT NULL;
ALTER TABLE datasets ADD COLUMN origin_task_id INTEGER NULL;
ALTER TABLE datasets ADD COLUMN generation_meta_json TEXT NULL;

ALTER TABLE agentic_synthesis_tasks ADD COLUMN generated_dataset_id INTEGER NULL;

CREATE TABLE IF NOT EXISTS reasoning_distillation_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('dataset', 'trajectory_task')),
    source_dataset_id INTEGER NULL,
    source_task_id INTEGER NULL,
    strategy TEXT NOT NULL,
    target_max_tokens INTEGER NOT NULL DEFAULT 1024,
    compression_ratio REAL NOT NULL DEFAULT 0.5,
    keep_tool_trace INTEGER NOT NULL DEFAULT 0 CHECK (keep_tool_trace IN (0, 1)),
    note TEXT NULL,
    llm_api_key TEXT NOT NULL,
    llm_base_url TEXT NOT NULL,
    llm_model_name TEXT NOT NULL,
    output_file_path TEXT NOT NULL,
    generated_dataset_id INTEGER NULL,
    total_items INTEGER NOT NULL DEFAULT 0,
    processed_items INTEGER NOT NULL DEFAULT 0,
    distilled_samples INTEGER NOT NULL DEFAULT 0,
    avg_tokens INTEGER NOT NULL DEFAULT 0,
    started_time DATETIME NULL,
    finished_time DATETIME NULL,
    error_message TEXT NULL,
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (source_dataset_id) REFERENCES datasets (id),
    FOREIGN KEY (source_task_id) REFERENCES agentic_synthesis_tasks (id),
    FOREIGN KEY (generated_dataset_id) REFERENCES datasets (id)
);

CREATE INDEX IF NOT EXISTS idx_reasoning_distillation_tasks_user_id ON reasoning_distillation_tasks (user_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_distillation_tasks_source_dataset_id ON reasoning_distillation_tasks (source_dataset_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_distillation_tasks_source_task_id ON reasoning_distillation_tasks (source_task_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_distillation_tasks_insert_time ON reasoning_distillation_tasks (insert_time DESC);

CREATE TRIGGER IF NOT EXISTS trg_reasoning_distillation_tasks_update_time
AFTER UPDATE ON reasoning_distillation_tasks
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE reasoning_distillation_tasks
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS reasoning_distillation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('dataset', 'trajectory_task')),
    source_ref_id INTEGER NOT NULL,
    item_key TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    reasoning_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    record_json TEXT NOT NULL DEFAULT '{}',
    token_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed')),
    error_message TEXT NULL,
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES reasoning_distillation_tasks (id),
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE INDEX IF NOT EXISTS idx_reasoning_distillation_results_task_id ON reasoning_distillation_results (task_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_distillation_results_user_id ON reasoning_distillation_results (user_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_distillation_results_status ON reasoning_distillation_results (status);

CREATE TRIGGER IF NOT EXISTS trg_reasoning_distillation_results_update_time
AFTER UPDATE ON reasoning_distillation_results
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE reasoning_distillation_results
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;
