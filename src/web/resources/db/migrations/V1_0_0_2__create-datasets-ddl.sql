CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    insert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets (name);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets (status);

CREATE TRIGGER IF NOT EXISTS trg_datasets_update_time
AFTER UPDATE ON datasets
FOR EACH ROW
WHEN NEW.update_time = OLD.update_time
BEGIN
    UPDATE datasets
    SET update_time = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;
