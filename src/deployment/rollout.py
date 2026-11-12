"""Manually trigger a target-local release transaction or recover interrupted execution."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import signal
import sys

from src.deployment.config import fingerprint
from src.deployment.rollout_runtime import SourceRunner
from src.deployment.rollout_state import Store, durable_json, local_identity, read_json
from src.deployment.verify import validate_request


class CheckFailed(RuntimeError):
    """An operation failed; its immutable evidence remains in the transaction directory."""


class Transaction:
    """Keep one lock across baseline checks, candidate checks and every recovery operation."""

    def __init__(self, store, runner=None, cancelled=None):
        self.store = store
        self.runner = runner if runner is not None else SourceRunner(store)
        self.cancelled = cancelled if cancelled is not None else lambda: False
        self.lock_fd = None

    def step(self, slot, action, attempt, descriptor, request=None, *, require_pass=True):
        """Persist intent before calling a helper, including an interrupted helper's identity."""
        index = len(list(attempt.glob("*.intent.json")))
        output = attempt / f"{index:02d}-{descriptor}.json"
        durable_json(
            output.with_suffix(".intent.json"),
            {
                "slot": slot,
                "action": action,
                "started_at": datetime.now(UTC).isoformat(),
            },
        )
        try:
            response = self.runner.run(slot, action, request or {}, output, self.lock_fd)
        except Exception as error:  # pylint: disable=broad-exception-caught
            if not output.exists():
                durable_json(
                    output,
                    {
                        "schema_version": 1,
                        "evidence_kind": "release-operation",
                        "action": action,
                        "config_sha256": slot["config_sha256"],
                        "status": "failed",
                        "error_type": type(error).__name__,
                    },
                )
            raise CheckFailed(f"Operation failed; evidence: {output.name}") from error
        if response.get("config_sha256") != slot["config_sha256"] or (
            response.get("action") != action or response.get("status") not in ("passed", "failed")
        ):
            raise CheckFailed("Operation response identity/status mismatch")
        if require_pass and response["status"] != "passed":
            raise CheckFailed(f"Check failed; evidence: {output.name}")
        return response.get("result", {}), self.store.reference(output)

    def check_cancelled(self):
        """Handle SIGINT/TERM between settled operations without abandoning a child or lock."""
        if self.cancelled():
            raise InterruptedError("Release interrupted; restore the accepted baseline")

    def prove(self, slot, receipt, inspection, attempt, label, payload):
        """Promotion and recovery both require an observed forecast for the exact inspection."""
        forecast, proof = self.step(
            slot,
            "forecast",
            attempt,
            label + "-forecast",
            {
                "receipt": receipt,
                "payload": payload,
            },
        )
        config = self.store.config(slot)
        if (
            forecast.get("status") != "passed"
            or forecast.get("expect_stopped") is not False
            or (
                forecast.get("inspection_receipt_sha256") != fingerprint(receipt)
                or forecast.get("request_sha256") != fingerprint(payload)
                or forecast.get("release_manifest_sha256") != config.release["manifest_sha256"]
                or forecast.get("image_reference") != config.release["image_reference"]
                or forecast.get("source_commit") != config.source_commit
                or [row.get("status") for row in forecast.get("checks", [])]
                != [200, 200, 200, 422]
                or any(row.get("passed") is not True for row in forecast.get("checks", []))
            )
        ):
            raise CheckFailed("Forecast evidence does not prove this complete release")
        return {"slot": slot, "inspection": inspection, "forecast": proof}

    @staticmethod
    def expected(state, expected):
        """Require an explicit baseline identity rather than silently using a changed service."""
        if state is None or state["current"]["slot"]["config_sha256"] != expected:
            raise ValueError("Stale or missing expected-current identity")

    def pair(self, baseline, candidate):
        """Both slots operate the same private service while retaining independent resources."""
        left, right = self.store.config(baseline), self.store.config(candidate)
        if (left.target_hostname, left.docker_socket, left.port) != (
            right.target_hostname,
            right.docker_socket,
            right.port,
        ):
            raise ValueError("Slots must target the same host, daemon and private loopback port")
        if left.deployment_id == right.deployment_id or baseline == candidate:
            raise ValueError("Candidate requires a distinct owned container and config")
        for first, second in ((left.checkpoint, right.checkpoint), (left.params, right.params)):
            a, b = Path(first), Path(second)
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("Preserve separate complete model and parameter slots")

    def finish(self, attempt, status, **details):
        """Retain terminal evidence separately from the mutable service-state pointer."""
        value = {
            "schema_version": 1,
            "evidence_kind": "release-transaction",
            "status": status,
            "attempt_id": attempt.name,
            "recorded_at": datetime.now(UTC).isoformat(),
            **details,
        }
        durable_json(attempt / "summary.json", value)
        return value

    def committed(self, attempt, status, state, **details):
        """Record success only after the accepted-state pointer has been persisted."""
        result = {
            "schema_version": 1,
            "evidence_kind": "release-state-commit",
            "status": status,
            "attempt_id": attempt.name,
            "generation": state["generation"],
            "state": self.store.reference(self.store.directory / "state.json"),
            **details,
        }
        durable_json(attempt / "commit.json", result)
        return result

    def adopt(self, descriptor, payload):
        """Register only a running published release after full source/model/HTTP checks."""
        validate_request(payload)
        with self.store.locked() as self.lock_fd:
            if self.store.load() is not None:
                raise ValueError("A baseline is already registered; preserve its state")
            slot = self.store.snapshot(descriptor)
            attempt = self.store.attempt()
            try:
                receipt, inspection = self.step(slot, "inspect", attempt, "baseline-inspection")
                accepted = self.prove(slot, receipt, inspection, attempt, "baseline", payload)
                self.check_cancelled()
                self.finish(attempt, "baseline_verified", accepted=accepted)
                state = {
                    "schema_version": 1,
                    "identity": self.store.identity,
                    "generation": 1,
                    "current": accepted,
                    "previous": None,
                    "pending": None,
                    "last_result": self.store.reference(attempt / "summary.json"),
                }
                self.store.save(state)
                return self.committed(attempt, "adopted", state, accepted=accepted)
            except Exception as error:
                if not (attempt / "summary.json").exists():
                    self.finish(attempt, "rejected", error_type=type(error).__name__)
                raise

    def deploy(self, descriptor, expected, payload, *, drill_stop_before_forecast=False):
        """Journal before stopping the accepted service; failures attempt complete recovery."""
        validate_request(payload)
        with self.store.locked() as self.lock_fd:
            state = self.store.load()
            self.expected(state, expected)
            if state["pending"] is not None:
                raise ValueError("Interrupted transaction requires explicit recover before deploy")
            candidate = self.store.snapshot(descriptor)
            baseline = state["current"]["slot"]
            self.pair(baseline, candidate)
            if any(
                read_json(path).get("slot") == candidate
                for path in self.store.directory.glob(
                    "attempts/*/*candidate-preflight.intent.json"
                )
            ):
                raise ValueError("Use a fresh candidate slot; this identity was already attempted")
            attempt = self.store.attempt()
            durable_json(attempt / "request.json", payload)
            if drill_stop_before_forecast:
                durable_json(attempt / "drill.json", {"stop_candidate_before_forecast": True})
            try:
                # Both source-specific bundles, images and artifacts pass before mutation.
                self.step(candidate, "preflight", attempt, "candidate-preflight")
                receipt, inspection = self.step(
                    baseline, "inspect", attempt, "baseline-inspection"
                )
                self.prove(baseline, receipt, inspection, attempt, "baseline", payload)
                self.check_cancelled()
            except Exception as error:
                self.finish(attempt, "rejected", error_type=type(error).__name__)
                raise
            state["pending"] = {
                "id": attempt.name,
                "baseline": state["current"],
                "candidate": candidate,
            }
            # Failure to persist intent prevents all service mutations.
            self.store.save(state)
            try:
                stopped, _ = self.step(baseline, "quiesce", attempt, "baseline-stop")
                if (
                    stopped.get("running") is not False
                    or stopped.get("clean_shutdown") is not True
                ):
                    raise CheckFailed("Baseline clean shutdown was not proved")
                self.check_cancelled()
                receipt, inspection = self.step(candidate, "start", attempt, "candidate-start")
                self.check_cancelled()
                if drill_stop_before_forecast:
                    self.step(candidate, "quiesce", attempt, "controlled-candidate-stop")
                accepted = self.prove(
                    candidate, receipt, inspection, attempt, "candidate", payload
                )
                self.check_cancelled()
                state.update(
                    current=accepted,
                    previous=state["current"],
                    pending=None,
                    generation=state["generation"] + 1,
                )
                self.finish(
                    attempt, "candidate_verified", accepted=accepted, previous=state["previous"]
                )
                state["last_result"] = self.store.reference(attempt / "summary.json")
                self.store.save(state)
                return self.committed(
                    attempt, "deployed", state, accepted=accepted, previous=state["previous"]
                )
            except Exception as error:  # pylint: disable=broad-exception-caught
                # Reload the durable intent: promotion may have failed while saving its pointer.
                pending_state = self.store.load()
                if pending_state["pending"] is None:
                    raise
                durable_json(
                    attempt / "failure.json",
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "pending": pending_state["pending"],
                    },
                )
                return self.recover_locked(pending_state, payload, type(error).__name__)

    def recover_locked(self, state, payload, reason):
        """Retain failure and recovery receipts, refusing success without a restored forecast."""
        pending = state["pending"]
        attempt = self.store.attempt()
        durable_json(attempt / "recovery.json", {"pending": pending, "reason": reason})
        baseline, candidate = pending["baseline"]["slot"], pending["candidate"]
        try:
            stopped, _ = self.step(
                candidate, "quiesce", attempt, "candidate-stop", require_pass=False
            )
            if stopped.get("running") is not False:
                raise CheckFailed("Candidate quiescence was not proved")
            # Clear an interrupted parser under its exact ownership before another preflight.
            self.step(baseline, "quiesce", attempt, "baseline-quiesce", require_pass=False)
            receipt, inspection = self.step(baseline, "restore", attempt, "baseline-restore")
            restored = self.prove(baseline, receipt, inspection, attempt, "restored", payload)
            self.finish(
                attempt,
                "baseline_verified",
                restored=restored,
                failed_attempt_id=pending["id"],
                reason=reason,
            )
            state.update(
                current=restored,
                pending=None,
                generation=state["generation"] + 1,
                last_result=self.store.reference(attempt / "summary.json"),
            )
            self.store.save(state)
            return self.committed(
                attempt,
                "failed_recovered",
                state,
                restored=restored,
                failed_attempt_id=pending["id"],
                reason=reason,
            )
        except Exception as error:  # Preserve pending intent for a separate authorized retry.
            if not (attempt / "summary.json").exists():
                self.finish(
                    attempt,
                    "recovery_failed",
                    failed_attempt_id=pending["id"],
                    error_type=type(error).__name__,
                )
            raise CheckFailed(
                "Recovery failed; retain pending state and inspect status/evidence"
            ) from error

    def recover(self, expected, payload):
        """Resume recovery after SIGKILL, host loss, broken evidence or failed verification."""
        validate_request(payload)
        with self.store.locked() as self.lock_fd:
            state = self.store.load()
            self.expected(state, expected)
            if state["pending"] is None:
                raise ValueError("No interrupted transaction requires recovery")
            return self.recover_locked(state, payload, "explicit_recovery")

    def status(self):
        """Public read-only interface for #84; no target contact or inference runs here."""
        state = self.store.load()
        slots = [] if state is None else [state["current"]["slot"]]
        if state is not None and state["previous"] is not None:
            slots.append(state["previous"]["slot"])
        if state is not None and state["pending"] is not None:
            slots.append(state["pending"]["candidate"])
        return {
            "schema_version": 1,
            "evidence_kind": "release-transaction-status",
            "state": state,
            "releases": {
                slot["config_sha256"]: self.store.config(slot).as_dict() for slot in slots
            },
            "reports": [
                read_json(p)
                for pattern in ("attempts/*/summary.json", "attempts/*/commit.json")
                for p in sorted(self.store.directory.glob(pattern))
            ],
        }


def main(argv=None):
    """Manual trigger; a private target-local session is an explicit prerequisite."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state-dir", "lock-file"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--machine-id-sha256", required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("adopt", "deploy", "recover", "status"):
        command = commands.add_parser(action)
        if action in ("adopt", "deploy"):
            command.add_argument("--slot", required=True, type=Path)
        if action in ("deploy", "recover"):
            command.add_argument("--expected-current", required=True)
        if action == "deploy":
            command.add_argument("--drill-stop-before-forecast", action="store_true")
        if action != "status":
            command.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    interrupted = []

    def cancel(signum, _frame):
        interrupted.append(signum)

    signals = {s: signal.signal(s, cancel) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        store = Store(args.state_dir, args.lock_file, local_identity(args.machine_id_sha256))
        transaction = Transaction(store, cancelled=lambda: bool(interrupted))
        if args.action == "status":
            result = transaction.status()
        else:
            payload = read_json(args.request)
            if args.action == "adopt":
                result = transaction.adopt(args.slot, payload)
            elif args.action == "deploy":
                result = transaction.deploy(
                    args.slot,
                    args.expected_current,
                    payload,
                    drill_stop_before_forecast=args.drill_stop_before_forecast,
                )
            else:
                result = transaction.recover(args.expected_current, payload)
        print(json.dumps(result, allow_nan=False))
        # Failed candidates stay failures even when baseline recovery succeeded.
        passed = result.get("status") in (None, "adopted", "deployed") or (
            args.action == "recover" and result.get("status") == "failed_recovered"
        )
        return 0 if passed else 1
    except (OSError, ValueError, TypeError, KeyError, CheckFailed, InterruptedError) as error:
        print(
            f"Release transaction failed ({type(error).__name__}); inspect status/evidence",
            file=sys.stderr,
        )
        return 1
    finally:
        for signum, handler in signals.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
