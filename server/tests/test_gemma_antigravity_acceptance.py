"""The local real-model acceptance runner must stop bounded tasks."""
import asyncio
import json
from types import SimpleNamespace

from server import gemma_antigravity_acceptance as acceptance


def test_case_deadline_records_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "_identity", lambda *args: {"same": True})

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(acceptance, "_one_file", slow)
    args = SimpleNamespace(
        endpoint="http://127.0.0.1:1/v1", model="gemma-test",
        expected_serving_identity="bundle", expected_session_library_sha256="session",
        workspace=tmp_path / "workspace", output=tmp_path / "report.json",
        case_timeout_seconds=0.001,
    )
    assert asyncio.run(acceptance._run(args)) == 1
    report = json.loads(args.output.read_text())
    assert report["status"] == "fail"
    assert "one_file exceeded" in report["error"]
    assert report["case_timeout_seconds"] == args.case_timeout_seconds
