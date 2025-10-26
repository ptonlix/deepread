"""
Entry point enabling both CLI document processing and FastAPI service startup.

Running behaviours:
    python -m deepread submit document.pdf
    python -m deepread cli submit document.pdf
    python -m deepread serve --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from deepread.cli import commands

_CLI_COMMANDS = {"submit", "status", "fetch", "manifest"}


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        _build_parser().print_help()
        return

    if args[0] in _CLI_COMMANDS:
        commands.main(args)
        return

    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.entrypoint == "cli":
        if not parsed.cli_args:
            parser.error("No CLI command provided. Try 'deepread cli submit <path>'.")
        commands.main(list(parsed.cli_args))
        return

    if parsed.entrypoint == "serve":
        _run_api(
            host=parsed.host,
            port=parsed.port,
            reload=parsed.reload,
            workspace_root=parsed.workspace_root,
        )
        return

    parser.print_help()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepread",
        description="Deepread document insight toolkit entry point.",
    )
    subparsers = parser.add_subparsers(dest="entrypoint")

    serve = subparsers.add_parser(
        "serve", help="Start the FastAPI service via uvicorn."
    )
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    serve.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    serve.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only).",
    )
    serve.add_argument(
        "--workspace-root",
        type=Path,
        help="Directory for pipeline workspace artifacts.",
    )

    cli = subparsers.add_parser(
        "cli",
        help="Forward to the document processing CLI commands.",
    )
    cli.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the CLI (e.g. submit, status).",
    )

    return parser


def _run_api(
    *,
    host: str,
    port: int,
    reload: bool,
    workspace_root: Path | None,
) -> None:
    try:
        from deepread.api.router import create_app
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "FastAPI components are unavailable. Install project dependencies via "
            "`uv sync --all-extras` before starting the API service."
        ) from exc

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "uvicorn is required to run the API server. Install it via "
            "`uv add uvicorn` or include it in your environment."
        ) from exc

    app = create_app(workspace_root=workspace_root)
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover - module execution guard
    main()
