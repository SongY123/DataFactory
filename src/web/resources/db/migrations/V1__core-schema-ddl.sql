PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    last_login DATETIME NULL,
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_users_status ON users (status);

CREATE TRIGGER IF NOT EXISTS trg_users_update_time
AFTER UPDATE ON users
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE users
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS agentic_synthesis_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    action_tags_json TEXT NOT NULL DEFAULT '[]',
    llm_api_key TEXT NOT NULL,
    llm_base_url TEXT NOT NULL,
    llm_model_name TEXT NOT NULL,
    output_file_path TEXT NOT NULL,
    total_workspaces INTEGER NOT NULL DEFAULT 0,
    processed_workspaces INTEGER NOT NULL DEFAULT 0,
    started_time DATETIME NULL,
    finished_time DATETIME NULL,
    error_message TEXT NULL,
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
);

CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_tasks_insert_time ON agentic_synthesis_tasks (insert_time DESC);
CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_tasks_user_id ON agentic_synthesis_tasks (user_id);
CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_tasks_dataset_id ON agentic_synthesis_tasks (dataset_id);

CREATE TRIGGER IF NOT EXISTS trg_agentic_synthesis_tasks_update_time
AFTER UPDATE ON agentic_synthesis_tasks
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE agentic_synthesis_tasks
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;


CREATE TABLE IF NOT EXISTS agentic_synthesis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL,
    workspace_name TEXT NOT NULL,
    question TEXT NOT NULL,
    trajectory TEXT NOT NULL,
    evaluation_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed')),
    error_message TEXT NULL,
    model_output_audit TEXT NULL,
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES agentic_synthesis_tasks (id),
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
);

CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_results_task_id ON agentic_synthesis_results (task_id);
CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_results_user_id ON agentic_synthesis_results (user_id);
CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_results_status ON agentic_synthesis_results (status);

CREATE TRIGGER IF NOT EXISTS trg_agentic_synthesis_results_update_time
AFTER UPDATE ON agentic_synthesis_results
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE agentic_synthesis_results
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;


CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'instruction',
    source TEXT NULL,
    language TEXT NOT NULL DEFAULT 'multi',
    size INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'uploaded',
    note TEXT NULL,
    file_name TEXT NULL,
    file_path TEXT NULL,
    sample_data TEXT NULL,
    cover_path TEXT NULL,
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets (name);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets (status);
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets (user_id);

CREATE TRIGGER IF NOT EXISTS trg_datasets_update_time
AFTER UPDATE ON datasets
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE datasets
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;

