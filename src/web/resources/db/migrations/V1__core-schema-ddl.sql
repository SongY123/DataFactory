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
    prompt_text TEXT NOT NULL,
    action_tags_json TEXT NOT NULL DEFAULT '[]',
    llm_api_key TEXT NOT NULL,
    llm_base_url TEXT NOT NULL,
    llm_model_name TEXT NOT NULL,
    dataset_paths_json TEXT NOT NULL DEFAULT '[]',
    output_file_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    total_files INTEGER NOT NULL DEFAULT 0,
    processed_files INTEGER NOT NULL DEFAULT 0,
    success_files INTEGER NOT NULL DEFAULT 0,
    failed_files INTEGER NOT NULL DEFAULT 0,
    started_time DATETIME NULL,
    finished_time DATETIME NULL,
    error_message TEXT NULL,
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_tasks_status ON agentic_synthesis_tasks (status);
CREATE INDEX IF NOT EXISTS idx_agentic_synthesis_tasks_insert_time ON agentic_synthesis_tasks (insert_time DESC);

CREATE TRIGGER IF NOT EXISTS trg_agentic_synthesis_tasks_update_time
AFTER UPDATE ON agentic_synthesis_tasks
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE agentic_synthesis_tasks
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;
