import yaml
from fastapi import APIRouter, HTTPException

from agentanalytics.plugins.registry import list_plugin_meta, get_plugin_meta

router = APIRouter()


@router.get("/plugins")
async def plugins():
    metas = list_plugin_meta()
    return [
        {
            "name": m.name,
            "version": m.version,
            "title": m.title,
            "description": m.description,
            "requires": m.requires,
            "defaults": m.defaults_dict(),
            "params": [
                {
                    "key": p.key,
                    "type": p.type,
                    "title": p.title,
                    "description": p.description,
                    "default": p.default,
                    "required": p.required,
                    "enum": p.enum,
                    "min": p.min,
                    "max": p.max,
                }
                for p in m.params
            ],
        }
        for m in metas
    ]


@router.get("/plugins/{name}")
async def plugin_by_name(name: str):
    try:
        m = get_plugin_meta(name)
    except Exception:
        raise HTTPException(status_code=404, detail="Unknown plugin")

    return {
        "name": m.name,
        "version": m.version,
        "title": m.title,
        "description": m.description,
        "requires": m.requires,
        "defaults": m.defaults_dict(),
        "params": [
            {
                "key": p.key,
                "type": p.type,
                "title": p.title,
                "description": p.description,
                "default": p.default,
                "required": p.required,
                "enum": p.enum,
                "min": p.min,
                "max": p.max,
            }
            for p in m.params
        ],
    }


@router.get("/config/template")
async def config_template():
    metas = list_plugin_meta()
    # Provide a minimal skeleton plus plugin defaults
    template = {
        "mlflow": {"tracking_uri": "http://localhost:5000"},
        "timeframe": {"last_n_days": 7},
        "output": {"output_dir": "./artifacts", "overwrite": True},
        "plugins": [
            {"name": m.name, "enabled": False, "config": m.defaults_dict()}
            for m in metas
        ],
        "schema_bindings": {},
    }
    return {"yaml": yaml.safe_dump(template, sort_keys=False), "json": template}
