import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.runs import router as runs_router
from agentanalytics_server.routes.plugins import router as plugins_router


def create_app(runs_dir: str) -> FastAPI:
    app = FastAPI(title="AgentAnalytics Server", version="0.1.0")

    # Basic CORS for local UI dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    os.makedirs(runs_dir, exist_ok=True)
    app.state.runs_dir = os.path.abspath(runs_dir)

    app.include_router(runs_router, prefix="/api")
    app.include_router(plugins_router, prefix="/api")

    return app
