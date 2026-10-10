import argparse
from spikingbrain.webui.dashboard import build_webui


def main():
    parser = argparse.ArgumentParser(description="SpikingBrain-7B Gradio WebUI Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    app = build_webui()
    print(f"[*] Launching SpikingBrain WebUI on http://{args.host}:{args.port}")
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
