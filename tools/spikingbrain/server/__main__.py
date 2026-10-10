import argparse
import uvicorn
from spikingbrain.server.app import create_app


def main():
    parser = argparse.ArgumentParser(description="SpikingBrain-7B OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default 8000)")
    args = parser.parse_args()

    app = create_app()
    print(f"[*] Starting SpikingBrain server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
