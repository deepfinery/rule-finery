#!/usr/bin/env python3
"""Expose the Drools runner jar via a lightweight FastAPI service."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class DroolsRequest(BaseModel):
    facts: Dict[str, Any]


class DroolsRunner:
    def __init__(self, jar_path: Path, rules_path: Path):
        self.jar_path = jar_path
        self.rules_path = rules_path
        if not jar_path.exists():
            raise FileNotFoundError(f"Drools runner jar not found: {jar_path}")
        if not rules_path.exists():
            raise FileNotFoundError(f"Drools rules not found: {rules_path}")

    def run(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        cmd = [
            "java",
            "-jar",
            str(self.jar_path),
            str(self.rules_path),
            "-",
        ]
        payload = json.dumps(facts).encode("utf-8")
        try:
            result = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                check=True,
                timeout=15,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.decode() or exc.stdout.decode()) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Drools runner timed out") from exc
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid Drools output: {result.stdout}") from exc


def create_app(jar_path: Path, rules_path: Path) -> FastAPI:
    runner = DroolsRunner(jar_path, rules_path)
    app = FastAPI(title="Drools Scoring Service", version="1.0.0")

    @app.post("/v1/drools/score")
    def score(req: DroolsRequest) -> Dict[str, Any]:
        try:
            return runner.run(req.facts)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jar",
        default="rule-engine/drool-runner/target/drools-runner-1.0.0-shaded.jar",
        help="Path to the built drools runner jar.",
    )
    parser.add_argument(
        "--rules",
        default="rule-engine/rules/tx_aml.drl",
        help="Path to the Drools rules file.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(Path(args.jar).resolve(), Path(args.rules).resolve())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
