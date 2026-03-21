ALTER TABLE datasets ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'upload';
ALTER TABLE datasets ADD COLUMN hf_repo_id TEXT NULL;
ALTER TABLE datasets ADD COLUMN hf_revision TEXT NULL;
ALTER TABLE datasets ADD COLUMN readme_text TEXT NULL;
ALTER TABLE datasets ADD COLUMN modality_tags_json TEXT NULL DEFAULT '[]';
ALTER TABLE datasets ADD COLUMN format_tags_json TEXT NULL DEFAULT '[]';
ALTER TABLE datasets ADD COLUMN language_tags_json TEXT NULL DEFAULT '[]';
ALTER TABLE datasets ADD COLUMN license_tag TEXT NULL;
ALTER TABLE datasets ADD COLUMN import_progress INTEGER NOT NULL DEFAULT 100;
ALTER TABLE datasets ADD COLUMN import_total_files INTEGER NOT NULL DEFAULT 0;
ALTER TABLE datasets ADD COLUMN import_downloaded_files INTEGER NOT NULL DEFAULT 0;
ALTER TABLE datasets ADD COLUMN import_error_message TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_datasets_source_kind ON datasets (source_kind);
CREATE INDEX IF NOT EXISTS idx_datasets_hf_repo_id ON datasets (hf_repo_id);
