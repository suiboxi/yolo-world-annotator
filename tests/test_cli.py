from __future__ import annotations

import pytest


def test_cli_reports_version_without_starting_qt(capsys) -> None:
    from yolo_world_annotator.cli import main

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "yolo-world-annotator 0.1.0"


def test_cli_accepts_explicit_cpu_device() -> None:
    from yolo_world_annotator.cli import parse_args

    args = parse_args(["--device", "cpu"])

    assert args.device == "cpu"


@pytest.mark.parametrize("value", ["gpu", "mps", "cuda:-1", "cuda:abc"])
def test_cli_rejects_invalid_device_syntax(value: str) -> None:
    from yolo_world_annotator.cli import parse_args

    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--device", value])

    assert exc_info.value.code == 2
