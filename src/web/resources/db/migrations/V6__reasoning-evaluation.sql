ALTER TABLE reasoning_distillation_tasks ADD COLUMN evaluation_enabled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE reasoning_distillation_tasks ADD COLUMN evaluation_prompt_text TEXT NULL;

ALTER TABLE reasoning_distillation_results ADD COLUMN evaluation_json TEXT NULL;
ALTER TABLE reasoning_distillation_results ADD COLUMN evaluation_raw_text TEXT NULL;
ALTER TABLE reasoning_distillation_results ADD COLUMN evaluation_error_message TEXT NULL;
