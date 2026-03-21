ALTER TABLE agentic_synthesis_tasks ADD COLUMN parallelism INTEGER NOT NULL DEFAULT 1;
ALTER TABLE agentic_synthesis_tasks ADD COLUMN llm_params_json TEXT NULL;

ALTER TABLE reasoning_distillation_tasks ADD COLUMN prompt_text TEXT NULL;
ALTER TABLE reasoning_distillation_tasks ADD COLUMN parallelism INTEGER NOT NULL DEFAULT 1;
ALTER TABLE reasoning_distillation_tasks ADD COLUMN llm_params_json TEXT NULL;
