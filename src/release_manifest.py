"""Bind a published CPU image to clean source and bounded release evidence.

This command checks API receipts, downloaded validation artifacts and Docker
inspection JSON. Verification is offline; no command loads a model or deploys.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import yaml

from src.modeling import directional, minimum_functionality
from src.modeling.predict import SAMPLING, ForecastConfig
from src.release_validation import (
    ACCEPTANCE,
    BLOCKS,
    BUDGETS,
    EVIDENCE_PATTERNS,
    MODEL,
    TOTAL_SECONDS,
    check_handoff,
)

ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPOSITORY = "mlops-2627q1-mds-upc/MLOps_Endesa"
VALIDATION_WORKFLOW = ".github/workflows/release-validation.yml"
ARCHIVE_LIMIT = 32 * 1024 * 1024
SHA256 = r"[0-9a-f]{64}"
REPOSITORY = r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/[a-z0-9]+(?:[._/-][a-z0-9]+)*"
VERSION = (
    r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?"
)


class ManifestError(ValueError):
    """A controlled rejection that can be displayed without report contents."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON object key")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    _require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        timeout=10,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).strip()


def _file(directory: Path, name: str) -> Path:
    directory = directory.resolve()
    path = directory / name
    _require(
        path.resolve().is_relative_to(directory.resolve())
        and not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(directory))
        and path.is_file(),
        f"Missing or unsafe evidence/source file: {name}",
    )
    return path


def _number(value, maximum: float, label: str, *, positive: bool = False) -> None:
    _require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (0 < value if positive else 0 <= value)
        and value <= maximum,
        f"Invalid or exceeded budget: {label}",
    )


def source_record(root: Path, source_sha: str) -> dict:
    """Require the caller's exact commit and a clean tracked/untracked checkout."""
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", source_sha)), "Use a full Git source SHA")
    _require(_git(root, "rev-parse", "HEAD") == source_sha, "Git source SHA mismatch")
    _require(
        not _git(root, "status", "--porcelain", "--untracked-files=all"),
        "Source checkout must be clean",
    )
    # Git status can hide assume-unchanged/skip-worktree content. Bind actual bytes
    # to HEAD as well, including Dockerfile and other inputs outside the report.
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", "HEAD"], cwd=root, timeout=10)
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, expected = metadata.split()
        path = _file(root, name.decode("utf-8"))
        _require(
            kind == b"blob" and mode in (b"100644", b"100755"),
            "Require regular tracked source files",
        )
        content = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        _require(actual == expected.decode(), "Tracked source bytes differ from the Git commit")
    return {
        "git_commit": source_sha,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "git_dirty": False,
    }


def _source_hashes(root: Path) -> dict:
    # These are the v1 release-validation report's source/configuration inputs.
    names = [
        "params.yaml",
        "pyproject.toml",
        "uv.lock",
        "dvc.yaml",
        "dvc.lock",
        "data/raw.dvc",
        f"{MODEL}.dvc",
        f"{MODEL}.provenance.json",
        "tests/test_minimum_functionality_model.py",
        VALIDATION_WORKFLOW,
    ]
    names += [name for name in _git(root, "ls-files", "src").splitlines() if name.endswith(".py")]
    return {name: _hash(_file(root, name)) for name in names}


def artifact_record(run: dict, artifact: dict, selection: dict) -> dict:
    """Validate authenticated API responses against explicit run/attempt/artifact pins.

    The workflow obtains these responses through gh api using actions:read. This
    pure checker alone cannot authenticate caller-created metadata.
    """
    for name in ("run_id", "run_attempt", "artifact_id"):
        _require(
            bool(re.fullmatch(r"[1-9][0-9]{0,19}", str(selection[name]))),
            "Use positive numeric Actions identifiers",
        )
    _require(
        bool(re.fullmatch("sha256:" + SHA256, selection["artifact_digest"])),
        "Pin the Actions artifact SHA-256 digest",
    )
    _require(
        run["repository"]["full_name"] == GITHUB_REPOSITORY
        and run["head_repository"]["full_name"] == GITHUB_REPOSITORY
        and run["path"] == VALIDATION_WORKFLOW
        and run["head_branch"] == "main"
        and run["event"] == "workflow_dispatch"
        and run["status"] == "completed"
        and run["conclusion"] == "success"
        and run["head_sha"] == selection["source_sha"]
        and str(run["id"]) == str(selection["run_id"])
        and str(run["run_attempt"]) == str(selection["run_attempt"]),
        "Selected validation run/source/workflow/event/attempt did not pass",
    )
    name = f"release-validation-{selection['run_id']}-{selection['run_attempt']}"
    _require(
        artifact["id"] == int(selection["artifact_id"])
        and artifact["name"] == name
        and artifact["expired"] is False
        and artifact["digest"] == selection["artifact_digest"]
        and artifact["workflow_run"]["id"] == run["id"]
        and artifact["workflow_run"]["repository_id"] == run["repository"]["id"]
        and artifact["workflow_run"]["head_repository_id"] == run["repository"]["id"]
        and artifact["workflow_run"]["head_sha"] == selection["source_sha"]
        and 0 < artifact["size_in_bytes"] <= ARCHIVE_LIMIT
        and datetime.fromisoformat(run["run_started_at"])
        <= datetime.fromisoformat(artifact["created_at"])
        <= datetime.fromisoformat(artifact["updated_at"])
        <= datetime.fromisoformat(run["updated_at"]),
        "Selected artifact is missing, expired, mismatched or outside this run attempt",
    )
    return {
        "repository": GITHUB_REPOSITORY,
        "workflow": VALIDATION_WORKFLOW,
        "run_id": str(run["id"]),
        "run_attempt": str(run["run_attempt"]),
        "id": artifact["id"],
        "name": name,
        "digest": artifact["digest"],
        "url": f"https://github.com/{GITHUB_REPOSITORY}/actions/runs/{run['id']}/attempts/{run['run_attempt']}",
    }


def _archive_members(archive: Path, digest: str) -> list[zipfile.ZipInfo]:
    _require(
        archive.stat().st_size <= ARCHIVE_LIMIT and "sha256:" + _hash(archive) == digest,
        "Downloaded Actions artifact digest mismatch",
    )
    with zipfile.ZipFile(archive) as bundle:
        members = [member for member in bundle.infolist() if not member.is_dir()]
    _require(
        0 < len(members) <= 256 and sum(m.file_size for m in members) <= 100 * 1024 * 1024,
        "Validation artifact exceeds extraction limits",
    )
    _require(len({m.filename for m in members}) == len(members), "Duplicate artifact members")
    for member in members:
        path = Path(member.filename)
        _require(
            not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in member.filename
            and len(path.parts) >= 2
            and re.fullmatch(r"[0-9a-f]{32}", path.parts[0])
            and (member.external_attr >> 16) & 0o170000 != 0o120000,
            "Unsafe validation artifact member",
        )
        relative = Path(*path.parts[1:])
        _require(
            any(
                relative.match(pattern)
                for pattern in (*EVIDENCE_PATTERNS, "*.log", "handoff.json", "handoff.sha256")
            ),
            "Unexpected validation artifact member",
        )
    return members


def ingest(archive: Path, run: dict, artifact: dict, selection: dict, output_dir: Path) -> Path:
    """Check actual API receipts/digest and unpack a bounded, immutable evidence bundle."""
    artifact_record(run, artifact, selection)
    members = _archive_members(archive, selection["artifact_digest"])
    _require(
        len({Path(m.filename).parts[0] for m in members}) == 1,
        "Require one validation run per artifact",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        for member in members:
            target = output_dir / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream, bundle.open(member) as source:
                shutil.copyfileobj(source, stream)
    directory = output_dir / Path(members[0].filename).parts[0]
    shutil.copyfile(archive, directory / "validation-artifact.zip")
    # Retain only API identity/outcome fields; exclude actor text, commit messages,
    # URLs containing redirect signatures and arbitrary metadata fields.
    receipt = {
        "selection": selection,
        "run": {
            key: run[key]
            for key in (
                "id",
                "run_attempt",
                "head_sha",
                "head_branch",
                "path",
                "event",
                "status",
                "conclusion",
                "run_started_at",
                "updated_at",
            )
        },
        "artifact": {
            key: artifact[key]
            for key in (
                "id",
                "name",
                "expired",
                "digest",
                "size_in_bytes",
                "created_at",
                "updated_at",
                "workflow_run",
            )
        },
    }
    receipt["run"].update(
        {
            key: {name: run[key][name] for name in ("full_name", "id")}
            for key in ("repository", "head_repository")
        }
    )
    receipt["artifact"]["workflow_run"] = {
        key: artifact["workflow_run"][key]
        for key in ("id", "repository_id", "head_repository_id", "head_sha")
    }
    (directory / "actions-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return directory


def _model_record(root: Path, identities: dict) -> dict:
    provenance_path = _file(root, f"{MODEL}.provenance.json")
    provenance = _json(provenance_path)
    model = identities["model"]
    _require(
        provenance["schema_version"] == 1
        and re.fullmatch(SHA256, provenance["model_safetensors_sha256"])
        and re.fullmatch(r"[0-9a-f]{32}", provenance["training_run_id"])
        and model["sha256"] == provenance["model_safetensors_sha256"]
        and model["training_run_id"] == provenance["training_run_id"],
        "Checkpoint weights/training provenance mismatch",
    )
    model_dvc = yaml.safe_load((root / f"{MODEL}.dvc").read_text())["outs"][0]
    expected = {
        MODEL: model_dvc["md5"],
        "data/raw": yaml.safe_load((root / "data/raw.dvc").read_text())["outs"][0]["md5"],
    }
    outputs = yaml.safe_load((root / "dvc.lock").read_text())["stages"]["split"]["outs"]
    by_name = {row["path"]: row["md5"] for row in outputs}
    expected.update(
        {f"data/processed/{b}.parquet": by_name[f"data/processed/{b}.parquet"] for b in BLOCKS}
    )
    _require(
        {name: row["md5"] for name, row in identities["artifacts"].items()} == expected,
        "Validation DVC artifact identities mismatch",
    )
    return {
        "path": MODEL,
        "dvc_md5": model_dvc["md5"],
        "weights_sha256": model["sha256"],
        "training_run_id": model["training_run_id"],
        "provenance_sha256": _hash(provenance_path),
    }


def _successful_steps(report: dict) -> list[str]:
    _require(
        type(report["schema_version"]) is int
        and report["schema_version"] == 1
        and report["status"] == "passed"
        and type(report["exit_code"]) is int
        and report["exit_code"] == 0
        and report["acceptance"] == ACCEPTANCE,
        "Release validation must have passed the declared v1 acceptance rule",
    )
    _require(report["budgets_seconds"] == BUDGETS, "Release validation step caps mismatch")
    _number(report["total_budget_seconds"], TOTAL_SECONDS, "total cap", positive=True)
    _number(report["wall_seconds"], report["total_budget_seconds"], "total elapsed")
    names = [step["name"] for step in report["steps"]]
    expected = ["verify", "configure-data", "data", "minimum-functionality", "directional"]
    _require(names in (expected, ["retrieve", *expected]), "Missing/duplicate validation steps")
    for step in report["steps"]:
        _require(
            step["status"] == "passed"
            and type(step["exit_code"]) is int
            and step["exit_code"] == 0,
            f"Validation step did not pass: {step['name']}",
        )
        _number(step["budget_seconds"], BUDGETS[step["name"]], step["name"], positive=True)
        _number(step["wall_seconds"], step["budget_seconds"], step["name"])
    return names


def _detailed_reports(directory: Path, source: dict, model: dict, prediction: dict) -> list[str]:
    data_name = "data-check/reports/validation.json"
    data = _json(_file(directory, data_name))
    _require(
        data["success"] is True
        and data["failed_expectations"] == 0
        and data["evaluated_expectations"] > 0
        and data["failures"] == [],
        "Data validation report did not pass",
    )
    minimum = _json(_file(directory, "minimum-functionality.json"))
    _require(
        minimum["status"] == "completed"
        and minimum["exit_code"] == 0
        and minimum["within_budget"] is True
        and minimum["protocol"]["cases"] == [asdict(c) for c in minimum_functionality.CASES]
        and [case["case_id"] for case in minimum["cases"]]
        == [c.case_id for c in minimum_functionality.CASES]
        and all(case["status"] == "passed" for case in minimum["cases"]),
        "Minimum-functionality evidence is incomplete or failed",
    )
    directional_names = sorted(directory.glob("directional/*/report.json"))
    _require(len(directional_names) == 1, "Require exactly one directional report")
    directional_name = directional_names[0].relative_to(directory).as_posix()
    direction = _json(_file(directory, directional_name))
    plan_name = str(Path(directional_name).with_name("plan.json"))
    _require(
        direction["status"] == "completed"
        and direction["success"] is True
        and direction["control"]["passed"] is True
        and direction["plan"]["cases"] == [asdict(c) for c in directional.CASES]
        and direction["plan"]["minimum_response_fraction"] == directional.MIN_RESPONSE_FRACTION
        and direction["plan"]["control_max_absolute_difference"] == directional.CONTROL_TOLERANCE
        and [case["case_id"] for case in direction["cases"]]
        == [c.case_id for c in directional.CASES]
        and all(case["passed"] is True for case in direction["cases"])
        and direction["plan"] == _json(_file(directory, plan_name)),
        "Directional evidence is incomplete or failed",
    )
    for record, provenance in (
        (minimum, minimum["provenance"]),
        (direction["plan"], direction["plan"]),
    ):
        _require(
            provenance["git_commit"] == source["git_commit"] and provenance["git_dirty"] is False,
            "Detailed validation source mismatch",
        )
        _require(
            record["model"]["source"] == "local"
            and record["model"]["sha256"] == model["weights_sha256"]
            and record["model"]["training_run_id"] == model["training_run_id"],
            "Detailed validation checkpoint mismatch",
        )
        _require(
            all(
                record["settings"][name] == prediction[name]
                for name in ("num_samples", "seed", "device", "cpu_threads")
            )
            and record["sampling"] == SAMPLING,
            "Detailed validation prediction settings mismatch",
        )
    return [data_name, "minimum-functionality.json", directional_name, plan_name]


def candidate(validation_dir: Path, source_sha: str, *, root: Path | None = None) -> dict:
    """Check retained evidence against source; never execute model checks again."""
    root = ROOT if root is None else root
    source = source_record(root, source_sha)
    handoff_path = _file(validation_dir, "handoff.json")
    _require(
        _file(validation_dir, "handoff.sha256").read_text()
        == f"{_hash(handoff_path)}  handoff.json\n",
        "Validation handoff checksum mismatch",
    )
    # Check member paths before the dependency's hash checker reads report files.
    for path in validation_dir.rglob("*"):
        _require(not path.is_symlink(), "Unsafe symlink in validation evidence")
    try:
        handoff = check_handoff(handoff_path, source_commit=source_sha)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise ManifestError(
            "Release-validation handoff rejected; retain all successful retrieval/check reports"
        ) from error
    report = _json(_file(validation_dir, "report.json"))
    steps = _successful_steps(report)
    _require(
        report["source"]["git_commit"] == source_sha
        and report["source"]["git_tree"] == source["git_tree"]
        and report["source"]["git_dirty"] is False
        and report["source"]["sha256"] == _source_hashes(root),
        "Release validation source/configuration mismatch",
    )
    identities = _json(_file(validation_dir, "identities.json"))
    _require(identities == report["identities"], "Retained artifact identity report mismatch")
    model = _model_record(root, identities)
    settings = asdict(ForecastConfig.from_params(root / "params.yaml", cpu_threads=2))
    prediction = {
        name: settings[name]
        for name in ("context_length", "horizon", "num_samples", "seed", "device", "cpu_threads")
    }
    prediction["sampling"] = SAMPLING
    params = yaml.safe_load((root / "params.yaml").read_text())
    params["evaluation"].update(device="cpu", cpu_threads=2)
    executed = _file(validation_dir, "params-cpu2.yaml")
    _require(
        yaml.safe_load(executed.read_text()) == params
        and _hash(executed) == report["executed_params_sha256"],
        "Executed release-validation parameters mismatch",
    )
    names = [
        "report.json",
        "identities.json",
        "params-cpu2.yaml",
        "handoff.json",
        "handoff.sha256",
        "actions-provenance.json",
    ]
    names += [f"{name}.log" for name in steps]
    names += _detailed_reports(validation_dir, source, model, prediction)
    names = sorted(set(names) | handoff["reports_sha256"].keys())
    # Copy only defined GitHub identifiers, never arbitrary environment/command fields.
    receipt = _json(_file(validation_dir, "actions-provenance.json"))
    _require(receipt["selection"]["source_sha"] == source_sha, "Actions receipt source mismatch")
    artifact = artifact_record(receipt["run"], receipt["artifact"], receipt["selection"])
    github = {
        "GITHUB_REPOSITORY": GITHUB_REPOSITORY,
        "GITHUB_RUN_ID": artifact["run_id"],
        "GITHUB_RUN_ATTEMPT": artifact["run_attempt"],
        "GITHUB_SHA": source_sha,
        "GITHUB_WORKFLOW_REF": f"{GITHUB_REPOSITORY}/{VALIDATION_WORKFLOW}@refs/heads/main",
    }
    _require(
        report["source"]["github"] == github, "GitHub run/attempt/workflow attribution mismatch"
    )
    archive = _file(validation_dir, "validation-artifact.zip")
    with zipfile.ZipFile(archive) as bundle:
        for member in _archive_members(archive, artifact["digest"]):
            prefix, relative = member.filename.split("/", 1)
            _require(prefix == report["run_id"], "Artifact validation run directory mismatch")
            _require(
                bundle.read(member) == _file(validation_dir, relative).read_bytes(),
                "Artifact extracted bytes mismatch",
            )
    _require(
        handoff["reports_sha256"]
        == {name: _hash(_file(validation_dir, name)) for name in handoff["reports_sha256"]},
        "Handoff report hashes mismatch",
    )
    return {
        "source": source,
        "model": model,
        "configuration": {**handoff["configuration"], "prediction": prediction},
        "validation": {
            "status": "passed",
            "run_id": report["run_id"],
            "report_sha256": _hash(validation_dir / "report.json"),
            "handoff_sha256": _hash(handoff_path),
            "executed_params_sha256": _hash(executed),
            "evidence_sha256": {name: _hash(_file(validation_dir, name)) for name in names},
            "github": github,
            "artifact": artifact,
            "data": handoff["data"],
            "source_sha256": report["source"]["sha256"],
        },
    }


def _image(repository: str, tag: str, digest: str, architecture: str, source_sha: str) -> dict:
    host = repository.split("/")[0]
    _require(
        bool(re.fullmatch(REPOSITORY, repository))
        and ("." in host or ":" in host or host == "localhost"),
        "Use an explicit lowercase registry/repository without credentials or a tag",
    )
    _require(
        bool(re.fullmatch(VERSION, tag)),
        "Use a version tag such as v1.0.0; mutable aliases are rejected",
    )
    _require(
        bool(re.fullmatch("sha256:" + SHA256, digest)), "Require an immutable sha256 image digest"
    )
    _require(architecture in ("amd64", "arm64"), "Architecture must be amd64 or arm64")
    return {
        "repository": repository,
        "tag": tag,
        "digest": digest,
        "reference": f"{repository}@{digest}",
        "os": "linux",
        "architecture": architecture,
        "revision": source_sha,
    }


def check_image(
    inspection: Path, *, repository: str, tag: str, architecture: str, source_sha: str
) -> dict:
    """Check built image metadata before the workflow is allowed to push it."""
    inspected = json.loads(inspection.read_text())
    _require(isinstance(inspected, list) and len(inspected) == 1, "Inspect exactly one image")
    image = inspected[0]
    _require(
        image["Architecture"] == architecture
        and image["Os"] == "linux"
        and image["Config"]["Labels"]["org.opencontainers.image.revision"] == source_sha
        and image["Config"]["Labels"]["org.opencontainers.image.source"]
        == f"https://github.com/{GITHUB_REPOSITORY}"
        and f"{repository}:{tag}" in image["RepoTags"],
        "Inspected image source/platform/version mismatch",
    )
    return image


def create_manifest(
    validation_dir: Path,
    source_sha: str,
    inspection: Path,
    *,
    repository: str,
    tag: str,
    architecture: str,
    root: Path | None = None,
) -> dict:
    """Use a post-push Docker inspection, including its registry digest and revision."""
    evidence = candidate(validation_dir, source_sha, root=root)
    image = check_image(
        inspection,
        repository=repository,
        tag=tag,
        architecture=architecture,
        source_sha=source_sha,
    )
    references = [ref for ref in image["RepoDigests"] if ref.startswith(repository + "@")]
    _require(len(references) == 1, "Require exactly one published repository digest")
    record = _image(repository, tag, references[0].split("@", 1)[1], architecture, source_sha)
    return {"schema_version": 1, **evidence, "image": record}


def verify_manifest(
    manifest: dict,
    validation_dir: Path,
    image_reference: str,
    architecture: str,
    *,
    root: Path | None = None,
) -> dict:
    """Read-only verification for deployment consumers using an expected digest."""
    _require(
        type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
        "Unsupported manifest schema",
    )
    evidence = candidate(validation_dir, manifest["source"]["git_commit"], root=root)
    image = manifest["image"]
    expected = _image(
        image["repository"],
        image["tag"],
        image["digest"],
        architecture,
        evidence["source"]["git_commit"],
    )
    _require(expected["reference"] == image_reference, "Expected image digest mismatch")
    _require(
        manifest == {"schema_version": 1, **evidence, "image": expected},
        "Release manifest identity/evidence mismatch",
    )
    return manifest


def main() -> None:
    """Check before building, create after push, or verify without modifying files."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_command = commands.add_parser("ingest")
    for name in ("run-metadata", "artifact-metadata", "archive", "output-dir"):
        ingest_command.add_argument("--" + name, required=True, type=Path)
    for name in ("source-sha", "run-id", "run-attempt", "artifact-id", "artifact-digest"):
        ingest_command.add_argument("--" + name, required=True)
    for name in ("check", "create", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--validation-dir", required=True, type=Path)
        if name == "verify":
            command.add_argument("--manifest", required=True, type=Path)
            command.add_argument("--image-reference", required=True)
        else:
            command.add_argument("--source-sha", required=True)
            command.add_argument("--repository", required=True)
            command.add_argument("--version", required=True)
        command.add_argument("--architecture", required=True, choices=("amd64", "arm64"))
        if name == "check":
            command.add_argument("--image-inspect", type=Path)
        if name == "create":
            command.add_argument("--image-inspect", required=True, type=Path)
            command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            selection = {
                key: getattr(args, key)
                for key in (
                    "source_sha",
                    "run_id",
                    "run_attempt",
                    "artifact_id",
                    "artifact_digest",
                )
            }
            directory = ingest(
                args.archive,
                _json(args.run_metadata),
                _json(args.artifact_metadata),
                selection,
                args.output_dir,
            )
            result = {"validation_dir": str(directory)}
        elif args.command == "verify":
            result = verify_manifest(
                _json(args.manifest), args.validation_dir, args.image_reference, args.architecture
            )
        else:
            _image(
                args.repository,
                args.version,
                "sha256:" + "0" * 64,
                args.architecture,
                args.source_sha,
            )
            if args.command == "check":
                result = candidate(args.validation_dir, args.source_sha)
                if args.image_inspect:
                    check_image(
                        args.image_inspect,
                        repository=args.repository,
                        tag=args.version,
                        architecture=args.architecture,
                        source_sha=args.source_sha,
                    )
            else:
                result = create_manifest(
                    args.validation_dir,
                    args.source_sha,
                    args.image_inspect,
                    repository=args.repository,
                    tag=args.version,
                    architecture=args.architecture,
                )
                # Never overwrite a release record or earlier failed execution evidence.
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    except ManifestError as error:
        parser.exit(1, f"Release manifest rejected: {error}\n")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
        yaml.YAMLError,
        IndexError,
        AttributeError,
    ) as error:
        # Do not echo untrusted reports, Docker JSON, credentials or exception values.
        parser.exit(
            1,
            f"Release manifest rejected ({type(error).__name__}); check source, image and complete validation evidence.\n",
        )


if __name__ == "__main__":
    main()
