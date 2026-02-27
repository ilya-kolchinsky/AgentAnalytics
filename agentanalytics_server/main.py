import argparse

import uvicorn
from .app import create_app


def main():
    parser = argparse.ArgumentParser(prog="agentanalytics-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--runs-dir", default="./artifacts/runs", help="Directory where run folders are stored")
    args = parser.parse_args()

    app = create_app(runs_dir=args.runs_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
