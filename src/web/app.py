from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.config_loader import ConfigLoader, get_config
from web.api import agentic_synthesis_router, auth_router, user_router
from web.db_migration_runner import SqlMigrationRunner
from web.entity.model import get_engine, init_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_CONFIG_PATH = PROJECT_ROOT / "src" / "web" / "resources" / "config.yaml"
CURRENT_WEB_CONFIG_PATH = DEFAULT_WEB_CONFIG_PATH


def _load_runtime_config(config_path: str | Path) -> None:
    global CURRENT_WEB_CONFIG_PATH
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    CURRENT_WEB_CONFIG_PATH = path
    ConfigLoader.load_config(str(path))


_load_runtime_config(DEFAULT_WEB_CONFIG_PATH)

from utils.logger import logger


def _resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _run_sql_migrations() -> None:
    sql_dir = _resolve_path(get_config("database.sql_dir"))
    if not sql_dir:
        logger.warning("database.sql_dir is empty, skip SQL migrations.")
        return
    engine = get_engine()
    SqlMigrationRunner(engine=engine, sql_dir=sql_dir).run()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_engine(config_path=str(CURRENT_WEB_CONFIG_PATH))
    _run_sql_migrations()
    logger.info("Web app initialized.")
    yield
    logger.info("Web app shutdown.")


app = FastAPI(
    title="DataFactory Web API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(agentic_synthesis_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataFactory FastAPI server")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_WEB_CONFIG_PATH),
        help="Path to web config yaml.",
    )
    args = parser.parse_args()

    _load_runtime_config(args.config)

    host = str(get_config("server.host", "127.0.0.1"))
    port = int(get_config("server.port", 8888))

    logger.info("=" * 60)
    logger.info("Launch DataFactory Web Service")
    logger.info("Service Address: http://%s:%s", host, port)
    logger.info("Swagger Doc: http://%s:%s/docs", host, port)
    logger.info("ReDoc: http://%s:%s/redoc", host, port)

    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
