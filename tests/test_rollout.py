"""Deterministic transaction fixtures; no registry, target host or model executes."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.deployment import release, rollout, rollout_state
from src.deployment.config import file_sha256, fingerprint
from src.deployment.rollout import CheckFailed, Transaction
from src.deployment.rollout_state import Store, durable_json, read_json
from tests.test_deployment_release import (  # pylint: disable=unused-import
    bound_release,  # noqa: F401
)
from tests.test_release_manifest import (  # pylint: disable=unused-import
    receipts as release_receipts,  # noqa: F401
)


class PowerLoss(BaseException):
    """A crash that cannot run Python recovery, like SIGKILL or host power loss."""


class FixtureRunner:
    """Model owned container transitions and real verifier results, without a model runtime."""

    def __init__(self, store, receipts):
        self.store, self.receipts = store, receipts
        self.running = {slot: False for slot in receipts}
        self.calls = []
        self.failures = {}
        self.callback = None

    def run(self, slot, action, request, output, lock_fd):
        """Return structured fixture observations and retain them as operation evidence."""
        assert os.fstat(lock_fd)
        config = self.store.config(slot)
        config.validate()
        key = (config.deployment_id, action)
        self.calls.append(key)
        if self.callback:
            self.callback(slot, action)
        failure = self.failures.get(key)
        if isinstance(failure, BaseException):
            raise failure
        if action != "quiesce":
            release.verify_release(config)  # Real #78 verifier against fabricated evidence.
        result = {"status": "passed"}
        if action == "quiesce":
            self.running[config.deployment_id] = False
            result.update(running=False, clean_shutdown=True, exit_code=0, oom_killed=False)
        elif action in ("start", "restore"):
            if failure != "readiness":
                self.running[config.deployment_id] = True
            result = deepcopy(self.receipts[config.deployment_id])
        elif action == "inspect":
            assert self.running[config.deployment_id]
            result = deepcopy(self.receipts[config.deployment_id])
        elif action == "forecast":
            refused = not self.running[config.deployment_id]
            result = {
                "status": "failed" if failure or refused else "passed",
                "expect_stopped": False,
                "inspection_receipt_sha256": fingerprint(request["receipt"]),
                "request_sha256": fingerprint(request["payload"]),
                "release_manifest_sha256": config.release["manifest_sha256"],
                "image_reference": config.release["image_reference"],
                "source_commit": config.source_commit,
                "checks": [
                    {
                        "status": None if refused else s,
                        "passed": not bool(failure or refused),
                        "connection_refused": refused,
                    }
                    for s in (200, 200, 200, 422)
                ],
            }
            if failure == "wrong_identity":
                result["image_reference"] = "ghcr.io/fixture/other@sha256:" + "0" * 64
                result["status"] = "passed"
        response = {
            "schema_version": 1,
            "evidence_kind": "release-operation",
            "action": action,
            "config_sha256": config.config_sha256,
            "status": "failed"
            if failure in ("readiness", "stop", "restore", "forecast")
            else "passed",
            "result": result,
        }
        if failure == "stop":
            result.update(status="failed", clean_shutdown=False, exit_code=137)
        if failure == "running":
            result["running"] = True
        durable_json(output, response)
        return response


@pytest.fixture
def scenario(bound_release, tmp_path):  # noqa: F811
    """Two distinct slots reuse the producer's explicitly synthetic complete evidence bundle."""
    bundle, initial, receipt = bound_release
    directory = tmp_path / "service-state"
    directory.mkdir(mode=0o700)
    identity = {
        "hostname": initial.target_hostname,
        "uid": os.geteuid(),
        "machine_id_sha256": "d" * 64,
    }
    store = Store(directory, directory / "transaction.lock", identity)
    configs, receipts, descriptors = {}, {}, {}
    for name in ("baseline", "candidate"):
        params = tmp_path / (name + "-params.yaml")
        params.write_bytes(Path(initial.params).read_bytes())
        config = replace(
            initial,
            deployment_id="endesa75-" + name,
            checkpoint=str(tmp_path / (name + "-checkpoint")),
            params=str(params),
        )
        configs[name] = config
        observed = deepcopy(receipt)
        observed.update(config=config.as_dict(), config_sha256=config.config_sha256)
        receipts[config.deployment_id] = observed
        path = tmp_path / (name + ".json")
        durable_json(path, config.as_dict())
        descriptor = tmp_path / (name + "-slot.json")
        durable_json(descriptor, {"checkout": str(bundle.root), "config": str(path)})
        descriptors[name] = descriptor
    runner = FixtureRunner(store, receipts)
    runner.running[configs["baseline"].deployment_id] = True
    return {
        "bundle": bundle,
        "store": store,
        "runner": runner,
        "configs": configs,
        "descriptors": descriptors,
        "transaction": Transaction(store, runner),
        "payload": {"state": "TAS", "history": [-1.5] * 512},
    }


def adopt(scenario):
    """Accept the fabricated published baseline through the transaction interface."""
    scenario["transaction"].adopt(scenario["descriptors"]["baseline"], scenario["payload"])
    return scenario["configs"]["baseline"].config_sha256


def deploy(scenario, expected=None):
    """Trigger the fixture candidate with an explicitly pinned baseline identity."""
    return scenario["transaction"].deploy(
        scenario["descriptors"]["candidate"],
        expected if expected is not None else scenario["configs"]["baseline"].config_sha256,
        scenario["payload"],
    )


def test_success_retains_complete_previous_identity_and_http_proof(scenario):
    expected = adopt(scenario)
    result = deploy(scenario, expected)
    assert result["status"] == "deployed"
    state = scenario["store"].load()
    assert (
        state["current"]["slot"]["config_sha256"] == scenario["configs"]["candidate"].config_sha256
    )
    assert state["previous"]["slot"]["config_sha256"] == expected
    assert state["pending"] is None
    assert state["generation"] == 2
    assert scenario["runner"].running == {"endesa75-baseline": False, "endesa75-candidate": True}
    for name in ("current", "previous"):
        slot = state[name]["slot"]
        assert (
            scenario["store"].config(slot).as_dict()
            == scenario["configs"]["candidate" if name == "current" else "baseline"].as_dict()
        )


@pytest.mark.parametrize(
    "action,failure",
    [
        ("start", "readiness"),
        ("forecast", "forecast"),
        ("forecast", "wrong_identity"),
        ("start", OSError()),
    ],
)
def test_failed_candidate_restores_and_forecasts_complete_baseline(scenario, action, failure):
    expected = adopt(scenario)
    scenario["runner"].failures[("endesa75-candidate", action)] = failure
    result = deploy(scenario)
    assert result["status"] == "failed_recovered"
    state = scenario["store"].load()
    assert state["current"]["slot"]["config_sha256"] == expected
    assert state["pending"] is None
    assert scenario["runner"].running == {"endesa75-baseline": True, "endesa75-candidate": False}
    assert scenario["runner"].calls[-1] == ("endesa75-baseline", "forecast")
    status = scenario["transaction"].status()
    assert any(report["status"] == "failed_recovered" for report in status["reports"])
    assert list(scenario["store"].directory.glob("attempts/*/*candidate*.json"))


@pytest.mark.parametrize("failure", ["restore", "forecast", "wrong_identity"])
def test_recovery_cannot_pass_without_restored_start_and_forecast(scenario, failure):
    expected = adopt(scenario)
    scenario["runner"].failures[("endesa75-candidate", "start")] = "readiness"

    # Baseline validation before stop must pass; inject failure only after candidate failure.
    def inject(_slot, action):
        if action == "start":
            scenario["runner"].failures[
                ("endesa75-baseline", "restore" if failure == "restore" else "forecast")
            ] = failure

    scenario["runner"].callback = inject
    with pytest.raises(CheckFailed, match="Recovery failed"):
        deploy(scenario)
    state = scenario["store"].load()
    pending = deepcopy(state["pending"])
    assert pending is not None
    calls = len(scenario["runner"].calls)
    with pytest.raises(ValueError, match="explicit recover"):
        deploy(scenario)
    assert len(scenario["runner"].calls) == calls
    scenario["runner"].failures.clear()
    scenario["runner"].callback = None
    assert (
        scenario["transaction"].recover(expected, scenario["payload"])["status"]
        == "failed_recovered"
    )
    assert scenario["store"].load()["pending"] is None
    statuses = {r["status"] for r in scenario["transaction"].status()["reports"]}
    assert {"recovery_failed", "failed_recovered"} <= statuses


def test_competing_deploy_cannot_enter_candidate_checks_or_recovery(scenario):
    expected = adopt(scenario)
    contender = Transaction(scenario["store"], scenario["runner"])
    observed = []

    def contend(_slot, action):
        if action in ("start", "restore", "forecast"):
            with pytest.raises(ValueError, match="lock is busy"):
                contender.deploy(
                    scenario["descriptors"]["candidate"], expected, scenario["payload"]
                )
            observed.append(action)

    scenario["runner"].callback = contend
    scenario["runner"].failures[("endesa75-candidate", "start")] = "readiness"
    assert deploy(scenario)["status"] == "failed_recovered"
    assert "restore" in observed and "forecast" in observed


@pytest.mark.parametrize("action", ["quiesce", "start", "forecast"])
def test_power_loss_leaves_durable_intent_and_requires_explicit_recovery(scenario, action):
    expected = adopt(scenario)
    key = ("endesa75-baseline" if action == "quiesce" else "endesa75-candidate", action)
    scenario["runner"].failures[key] = PowerLoss()
    with pytest.raises(PowerLoss):
        deploy(scenario)
    pending = deepcopy(scenario["store"].load()["pending"])
    assert pending is not None
    with pytest.raises(ValueError, match="explicit recover"):
        deploy(scenario)
    assert scenario["store"].load()["pending"] == pending
    assert list(scenario["store"].directory.glob("attempts/*/*.intent.json"))
    scenario["runner"].failures.clear()
    resumed = Transaction(scenario["store"], scenario["runner"])
    assert resumed.recover(expected, scenario["payload"])["status"] == "failed_recovered"


def test_term_is_deferred_until_helper_settles_then_recovery_holds_lock(scenario):
    adopt(scenario)
    cancelled = []
    scenario["transaction"].cancelled = lambda: bool(cancelled)
    scenario["runner"].callback = lambda _s, action: (
        cancelled.append(True) if action == "start" else None
    )
    assert deploy(scenario)["status"] == "failed_recovered"
    assert scenario["store"].load()["pending"] is None


@pytest.mark.parametrize(
    "change",
    ["manifest", "source", "bundle", "params", "hash", "missing-model", "missing-identity"],
)
def test_stale_malformed_or_missing_identities_fail_before_baseline_stop(scenario, change):
    adopt(scenario)
    config = scenario["configs"]["candidate"]
    if change == "manifest":
        Path(config.release["manifest"]).write_text("{malformed")
    elif change == "source":
        (scenario["bundle"].root / "src/__init__.py").write_text("# changed source\n")
    elif change == "bundle":
        (scenario["bundle"].directory / "report.json").write_text("{}")
    elif change == "params":
        config = replace(config, params_sha256="0" * 64)
    elif change == "hash":
        config = replace(config, release={**config.release, "manifest_sha256": "0" * 64})
    elif change == "missing-identity":
        path = Path(config.release["manifest"])
        raw = read_json(path)
        del raw["image"]["digest"]
        path.write_text(json.dumps(raw), encoding="utf-8")
        config = replace(config, release={**config.release, "manifest_sha256": file_sha256(path)})
    else:
        config = replace(config, checkpoint_sha256="0" * 64)
    if change in ("params", "hash", "missing-model", "missing-identity"):
        path = Path(read_json(scenario["descriptors"]["candidate"])["config"])
        path.write_text(json.dumps(config.as_dict()))
    before = list(scenario["runner"].calls)
    with pytest.raises(CheckFailed):
        deploy(scenario)
    assert all(action != "quiesce" for _, action in scenario["runner"].calls[len(before) :])
    assert scenario["runner"].running["endesa75-baseline"]
    assert scenario["store"].load()["pending"] is None


def test_schema1_cannot_be_adopted_as_a_published_baseline(scenario):
    descriptor = scenario["descriptors"]["baseline"]
    path = Path(read_json(descriptor)["config"])
    config = replace(scenario["configs"]["baseline"], schema_version=1, release=None)
    path.write_text(json.dumps(config.as_dict()))
    with pytest.raises(ValueError, match="schema-2"):
        scenario["transaction"].adopt(descriptor, scenario["payload"])
    assert not scenario["runner"].calls
    assert scenario["store"].load() is None


def test_stale_expected_current_and_reused_slot_refuse_without_runtime_work(scenario):
    adopt(scenario)
    before = list(scenario["runner"].calls)
    with pytest.raises(ValueError, match="Stale"):
        deploy(scenario, "0" * 64)
    with pytest.raises(ValueError, match="distinct"):
        scenario["transaction"].deploy(
            scenario["descriptors"]["baseline"],
            scenario["configs"]["baseline"].config_sha256,
            scenario["payload"],
        )
    assert scenario["runner"].calls == before


def test_failed_candidate_stop_does_not_claim_recovery(scenario):
    adopt(scenario)
    scenario["runner"].failures.update(
        {
            ("endesa75-candidate", "start"): "readiness",
            ("endesa75-candidate", "quiesce"): "running",
        }
    )
    with pytest.raises(CheckFailed, match="Recovery failed"):
        deploy(scenario)
    assert scenario["store"].load()["pending"] is not None
    assert ("endesa75-baseline", "restore") not in scenario["runner"].calls


def test_exit137_is_preserved_and_forces_failure_even_without_oom(scenario):
    adopt(scenario)
    scenario["runner"].failures[("endesa75-baseline", "quiesce")] = "stop"
    result = deploy(scenario)
    assert result["status"] == "failed_recovered"
    records = [
        read_json(p) for p in scenario["store"].directory.glob("attempts/*/*baseline-stop.json")
    ]
    assert records[0]["result"]["exit_code"] == 137
    assert records[0]["result"]["oom_killed"] is False
    assert records[0]["result"]["clean_shutdown"] is False


def test_state_write_failure_prevents_shutdown(scenario, monkeypatch):
    adopt(scenario)

    def broken_save(_value):
        raise OSError("fixture simulated fsync failure")

    monkeypatch.setattr(scenario["store"], "save", broken_save)
    before = len(scenario["runner"].calls)
    with pytest.raises(OSError):
        deploy(scenario)
    assert all(action != "quiesce" for _, action in scenario["runner"].calls[before:])
    assert scenario["store"].load()["pending"] is None


def test_controlled_stop_drill_observes_failed_forecast_and_recovers(scenario):
    expected = adopt(scenario)
    result = scenario["transaction"].deploy(
        scenario["descriptors"]["candidate"],
        expected,
        scenario["payload"],
        drill_stop_before_forecast=True,
    )
    assert result["status"] == "failed_recovered"
    assert ("endesa75-candidate", "quiesce") in scenario["runner"].calls
    paths = list(scenario["store"].directory.glob("attempts/*/*candidate-forecast.json"))
    report = read_json(paths[0])["result"]
    assert report["status"] == "failed"
    assert all(row["connection_refused"] for row in report["checks"])
    assert scenario["store"].load()["current"]["slot"]["config_sha256"] == expected


def test_candidate_slot_cannot_be_replayed_after_failed_attempt(scenario):
    adopt(scenario)
    scenario["runner"].failures[("endesa75-candidate", "start")] = "readiness"
    assert deploy(scenario)["status"] == "failed_recovered"
    calls = list(scenario["runner"].calls)
    with pytest.raises(ValueError, match="fresh candidate slot"):
        deploy(scenario)
    assert scenario["runner"].calls == calls


def test_crash_after_candidate_verified_cannot_create_false_commit(scenario, monkeypatch):
    expected = adopt(scenario)
    save = scenario["store"].save

    def power_loss(state):
        if state["current"]["slot"]["config_sha256"] != expected:
            raise PowerLoss()
        save(state)

    monkeypatch.setattr(scenario["store"], "save", power_loss)
    with pytest.raises(PowerLoss):
        deploy(scenario)
    assert scenario["store"].load()["pending"] is not None
    reports = scenario["transaction"].status()["reports"]
    assert any(report["status"] == "candidate_verified" for report in reports)
    assert not any(report["status"] == "deployed" for report in reports)
    monkeypatch.setattr(scenario["store"], "save", save)
    assert (
        scenario["transaction"].recover(expected, scenario["payload"])["status"]
        == "failed_recovered"
    )


@pytest.mark.parametrize(
    "field", ["target_hostname", "docker_socket", "port", "checkpoint", "params"]
)
def test_slot_must_preserve_host_service_and_independent_files(scenario, field):
    adopt(scenario)
    baseline = scenario["configs"]["baseline"]
    values = {
        "target_hostname": "another-host.invalid",
        "docker_socket": "/tmp/another.sock",
        "port": 19079,
        "checkpoint": baseline.checkpoint,
        "params": baseline.params,
    }
    candidate = replace(scenario["configs"]["candidate"], **{field: values[field]})
    path = Path(read_json(scenario["descriptors"]["candidate"])["config"])
    path.write_text(json.dumps(candidate.as_dict()), encoding="utf-8")
    calls = list(scenario["runner"].calls)
    with pytest.raises(ValueError):
        deploy(scenario)
    assert scenario["runner"].calls == calls


def test_public_status_is_read_only_while_transaction_lock_is_held(scenario):
    adopt(scenario)
    before = {p: p.read_bytes() for p in scenario["store"].directory.rglob("*") if p.is_file()}
    calls = list(scenario["runner"].calls)
    with scenario["store"].locked():
        status = scenario["transaction"].status()
    assert status["evidence_kind"] == "release-transaction-status"
    assert (
        status["releases"][scenario["configs"]["baseline"].config_sha256]
        == scenario["configs"]["baseline"].as_dict()
    )
    assert (
        status["state"]["current"]["slot"]["config_sha256"]
        == scenario["configs"]["baseline"].config_sha256
    )
    assert scenario["runner"].calls == calls
    assert before == {p: p.read_bytes() for p in before}


def test_corrupt_snapshot_or_accepted_evidence_blocks_operational_work(scenario):
    adopt(scenario)
    state = scenario["store"].load()
    path = scenario["store"].config_path(state["current"]["slot"])
    raw = read_json(path)
    raw["port"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="snapshot identity"):
        scenario["store"].load()


def test_machine_identity_and_state_ownership_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout_state.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="native Linux"):
        rollout_state.local_identity("a" * 64)
    path = tmp_path / "unsafe"
    path.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="0700"):
        Store(path, path / "transaction.lock", {})
    path.chmod(0o700)
    with pytest.raises(ValueError, match="stable"):
        Store(path, tmp_path / "bypass.lock", {})


def test_duplicate_json_cannot_hide_missing_identity(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"identity":1,"identity":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)


def test_inherited_child_retains_lock_after_parent_closes_descriptor(scenario, tmp_path):
    ready = tmp_path / "child-ready"
    release_child = tmp_path / "release-child"
    script = (
        "import os,sys,time; from pathlib import Path; "
        "os.fstat(int(sys.argv[1])); Path(sys.argv[2]).touch(); "
        "\nwhile not Path(sys.argv[3]).exists(): time.sleep(0.01)\n"
    )
    child = None
    try:
        with scenario["store"].locked() as descriptor:
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(descriptor), str(ready), str(release_child)],
                pass_fds=(descriptor,),
            )
            # Bounded deterministic synchronization, unrelated to any external target.
            child_ready = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys,time; from pathlib import Path; "
                        "\nfor _ in range(300):\n if Path(sys.argv[1]).exists(): sys.exit(0)\n time.sleep(.01)\n"
                        "sys.exit(1)"
                    ),
                    str(ready),
                ],
                check=False,
                timeout=5,
            )
            assert child_ready.returncode == 0
        with pytest.raises(ValueError, match="lock is busy"), scenario["store"].locked():
            pytest.fail("Orphan helper lost its inherited lock")
    finally:
        release_child.touch()
        if child:
            child.wait(timeout=5)
    with scenario["store"].locked():
        pass


def test_cli_preserves_failed_candidate_exit_even_after_successful_recovery(
    scenario, monkeypatch, tmp_path, capsys
):
    adopt(scenario)
    request = tmp_path / "request.json"
    durable_json(request, scenario["payload"])
    scenario["runner"].failures[("endesa75-candidate", "start")] = "readiness"
    monkeypatch.setattr(rollout, "local_identity", lambda _: scenario["store"].identity)
    monkeypatch.setattr(rollout, "SourceRunner", lambda _: scenario["runner"])
    args = [
        "--state-dir",
        str(scenario["store"].directory),
        "--lock-file",
        str(scenario["store"].lock_file),
        "--machine-id-sha256",
        "d" * 64,
        "deploy",
        "--slot",
        str(scenario["descriptors"]["candidate"]),
        "--expected-current",
        scenario["configs"]["baseline"].config_sha256,
        "--request",
        str(request),
    ]
    assert rollout.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed_recovered"
