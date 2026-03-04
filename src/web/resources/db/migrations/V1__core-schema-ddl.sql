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
