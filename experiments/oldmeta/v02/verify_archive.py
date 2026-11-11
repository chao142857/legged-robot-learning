"""Verify lightweight archive contents; no simulation, network or ML imports."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    package = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    for name, expected in package["files"].items():
        path = root / name
        raw = path.read_bytes()
        assert len(raw) == expected["bytes"], name
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"], name
        if path.suffix == ".py":
            ast.parse(raw, filename=name)
        if path.suffix == ".json":
            json.loads(raw)
    executed = json.loads((root / "evidence/source_manifest.json").read_text())
    for name, expected in executed["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    snapshot = json.loads((root / "evidence/progress_snapshot.json").read_text())
    meta = json.loads((root / "evidence/meta_summary_compact.json").read_text())
    assert meta["complete"] and meta["meta"]["completed_teacher_updates"] == 8
    assert all(p["engineering"]["passed"] for p in meta["meta"]["points"])
    assert snapshot["transfer"]["uniform"]["first_committed_cycle"]["passed"]
    log = (root / "evidence/jobs/cpu-tests-r03/console.txt").read_text()
    assert "Ran 36 tests" in log and log.rstrip().endswith("OK")
    print(f"PASS: {len(package['files'])} archived files; "
          f"{len(executed['files'])} executed source hashes; 8 completed meta starts.")


if __name__ == "__main__":
    main()
