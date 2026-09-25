"""
Unit and integration tests for InvariantOne CLI.
"""

from __future__ import annotations

import json
import pytest

from cli.invariantone_cli import build_parser, main


def test_cli_parser_help(capsys):
    """Verifies that --help prints usage without error."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


def test_cli_info_json(capsys):
    """Verifies that 'invariantone info --json' returns valid metadata JSON."""
    exit_code = main(["info", "--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["model_name"] == "InvariantOne"
    assert data["model_version"] == "1.0.0"
    assert data["architecture"] == "D5-L4"
    assert data["training_operator_breadth"] == 16
    assert data["calibration_temperature"] == 1.0091


def test_cli_verify_checkpoints():
    """Verifies that 'invariantone verify' validates local v1 checkpoints."""
    exit_code = main(["verify", "--checkpoint-dir", "checkpoints/invariantone-v1"])
    assert exit_code == 0


def test_cli_decide_insufficient_options(capsys):
    """Verifies that deciding with fewer than 2 options fails with exit code 1."""
    exit_code = main([
        "decide",
        "--state", "Pressure rising",
        "--question", "What to do?",
        "--option", "Close valve",
        "--json",
    ])
    assert exit_code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "error" in data
