"""Exercise source-specific helper subprocesses with tiny local package fixtures."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

from src.deployment.rollout_runtime import SourceRunner
from src.deployment.rollout_state import durable_json, read_json
from tests.test_deployment_release import (  # pylint: disable=unused-import
    bound_release,  # noqa: F401
)
from tests.test_release_manifest import (  # pylint: disable=unused-import
    receipts as release_receipts,  # noqa: F401
)
from tests.test_rollout import scenario  # noqa: F401 # pylint: disable=unused-import


def package(checkout, marker):
    """Fixture source has its own model validator and fake owned runtime; no Docker executes."""
    modules = {
        "src/__init__.py": "",
        "src/deployment/__init__.py": "",
        "src/deployment/config.py": """
import json, hashlib
class DeploymentConfig:
    @classmethod
    def load(cls, path):
        raw=json.loads(path.read_text())
        obj=cls()
        obj.__dict__.update(raw)
        obj.config_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return obj
""",
        "src/deployment/release.py": """
def verify_release(config):
    if config.source_commit != SOURCE or config.checkpoint_sha256 != MODEL:
        raise ValueError('Source-specific fixture selection rejected')
""".replace("SOURCE", repr(marker * 40)).replace("MODEL", repr(marker * 64)),
        "src/deployment/runtime.py": """
import json
from pathlib import Path
class Deployment:
    def __init__(self,config): self.config=config
    def target(self): return {'fixture':True}
    def preflight(self): return {'fixture_source':SOURCE,'model':self.config.checkpoint_sha256}
    def receipt(self,verified=None): return self.preflight()
    def start(self):
        raise RuntimeError('fixture candidate readiness failed')
    def owned(self,name=None):
        return {'Id': 'fixture-container','Image':self.config.image_id,
                'State': {'Running':False,'ExitCode':3,'OOMKilled':False}}
    def docker(self,*args): return self.config.deployment_id + '-preflight\\n' + self.config.deployment_id
    def lifecycle(self,action): return {'status':'passed','running':False,'clean_shutdown':True}
""".replace("SOURCE", repr(marker * 40)),
        "src/deployment/verify.py": "def verify(*args,**kwargs): return {'status':'failed','fixture_only':True}\n",
    }
    for name, code in modules.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
    interpreter = checkout / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)


@pytest.fixture
def sources(scenario, tmp_path):  # noqa: F811
    """Use different source/model identities that the controller's provenance does not select."""
    store = scenario["store"]
    result = []
    for marker in ("a", "b"):
        checkout = tmp_path / ("source-" + marker)
        package(checkout, marker)
        config = replace(
            scenario["configs"]["baseline"],
            source_commit=marker * 40,
            checkpoint_sha256=marker * 64,
        )
        config_file = tmp_path / (marker + "-config.json")
        durable_json(config_file, config.as_dict())
        descriptor = tmp_path / (marker + "-slot.json")
        durable_json(descriptor, {"checkout": str(checkout), "config": str(config_file)})
        result.append(store.snapshot(descriptor))
    return store, result


def test_helper_uses_each_sources_own_code_model_and_environment(sources, monkeypatch):
    store, slots = sources
    monkeypatch.setenv("PYTHONPATH", "/fixture-poisoned-search-path")
    with store.locked() as descriptor:
        for slot, marker in zip(slots, ("a", "b")):
            output = store.attempt() / "source-specific.json"
            response = SourceRunner(store).run(slot, "preflight", {}, output, descriptor)
            assert response["status"] == "passed"
            assert response["result"] == {"fixture_source": marker * 40, "model": marker * 64}


def test_helper_rejects_a_different_source_model_before_runtime_action(sources):
    store, slots = sources
    slot = {**slots[0], "checkout": slots[1]["checkout"]}
    with store.locked() as descriptor:
        response = SourceRunner(store).run(
            slot, "preflight", {}, store.attempt() / "rejected.json", descriptor
        )
    assert response["status"] == "failed"
    assert response["error_type"] == "ValueError"


def test_readiness_failure_retains_owned_exit_and_oom_state(sources):
    store, slots = sources
    with store.locked() as descriptor:
        response = SourceRunner(store).run(
            slots[0], "start", {}, store.attempt() / "readiness.json", descriptor
        )
    assert response["status"] == "failed"
    assert response["observed_container"]["state"] == {
        "Running": False,
        "ExitCode": 3,
        "OOMKilled": False,
    }


def test_broken_release_binding_does_not_prevent_owned_candidate_quiescence(sources):
    store, slots = sources
    # Source-specific release validation fails, but cleanup uses #75 ownership fences.
    slot = {**slots[0], "checkout": slots[1]["checkout"]}
    with store.locked() as descriptor:
        response = SourceRunner(store).run(
            slot, "quiesce", {}, store.attempt() / "quiesce.json", descriptor
        )
    assert response["status"] == "passed"
    assert response["result"]["running"] is False
    assert response["result"]["interrupted_parsers"][0]["state"]["ExitCode"] == 3


def test_missing_interpreter_and_helper_import_failure_are_not_success(sources):
    store, slots = sources
    with store.locked() as descriptor:
        slot = {**slots[0], "checkout": str(store.directory)}
        with pytest.raises(ValueError, match="locked .venv"):
            SourceRunner(store).run(
                slot, "preflight", {}, store.attempt() / "missing.json", descriptor
            )
        code = Path(slots[0]["checkout"]) / "src/deployment/config.py"
        code.write_text("raise ImportError('fixture unavailable dependency')\n", encoding="utf-8")
        response = SourceRunner(store).run(
            slots[0], "preflight", {}, store.attempt() / "import.json", descriptor
        )
    assert response["status"] == "failed"
    assert response["error_type"] == "HelperExit"
    assert response["exit_code"] != 0


def test_evidence_cannot_be_overwritten_by_helper(sources):
    store, slots = sources
    output = store.attempt() / "preserved.json"
    durable_json(output, {"retained": True})
    before = output.read_bytes()
    with store.locked() as descriptor, pytest.raises(ValueError, match="fresh"):
        SourceRunner(store).run(slots[0], "preflight", {}, output, descriptor)
    assert output.read_bytes() == before
    assert not output.with_suffix(".request.json").exists()
    assert read_json(output) == {"retained": True}
