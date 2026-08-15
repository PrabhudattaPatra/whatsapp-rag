"""
Entry point for running the chat frontend as its own standalone service.

Why this script exists:
    The frontend (frontend/index.html) is a plain static page with no
    build step, so it doesn't need a real dev server — Python's built-in
    `http.server` is enough. Running it as a script (rather than
    `python -m http.server` from inside frontend/) means it always serves
    the right directory regardless of your current working directory, and
    always binds the same port the API's CORS allow-list
    (`FRONTEND_ORIGINS` in api/main.py) expects.

Usage:
    uv run python -m scripts.run_frontend
    # then open http://localhost:5500/ in a browser
    # (the API must be running separately — see scripts/run_api.py)
"""

from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
PORT = 5500


def main() -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(FRONTEND_DIR))
    server = HTTPServer(("0.0.0.0", PORT), handler)
    print(f"Serving {FRONTEND_DIR} at http://localhost:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
