"""Build local draft PDFs from TeX and versioned evidence, without inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "delivery"
CANDIDATE = "f9b5896af998331494175733c557e8dc5dd12cc7"
EVIDENCE = (
    "monitoring/quality/validation-subset.json",
    "docs/evidence/serving-benchmark-2026-11-02.json",
    "monitoring/scenarios/local-evidence-2026-11-12.json",
)
DOCUMENTS = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/final-delivery.md",
    "docs/replication.md",
    "docs/requirements.md",
    "docs/DATASET_CARD.md",
    "docs/MODEL_CARD.md",
    "docs/rubric-evidence.md",
    "docs/api.md",
    "docs/container.md",
    "docs/release-validation.md",
    "docs/image-publication.md",
    "docs/virtech-deployment.md",
    "docs/deployment-release.md",
    "docs/monitoring-service.md",
    "docs/monitoring-scenarios.md",
    "docs/replay.md",
    "docs/drift.md",
    "docs/forecast-quality.md",
)


def digest(path: Path) -> str:
    """Return the SHA-256 of a bounded document or evidence input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str], *, cwd: Path = ROOT) -> str:
    """Run a read-only metadata or validation command."""
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def write_chart_data(output: Path) -> None:
    """Derive every measured chart mark from the versioned records."""
    quality, benchmark, scenarios = (json.loads((ROOT / name).read_text()) for name in EVIDENCE)
    states = quality["by_state"]
    if len(states) != 5 or any(row["forecast_count"] != 30 for row in states):
        raise ValueError("Coverage figure requires the declared five-state subset")
    with (output / "coverage.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "state", "coverage"))
        for index, row in enumerate(states):
            writer.writerow((index, row["state"], row["selected"]["empirical_coverage"] * 100))
    with (output / "latency.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "discovery", "confirmation"))
        for index, phase in enumerate(("warm_c1", "warm_c2")):
            values = [
                benchmark["runs"][run]["phases"][phase] for run in ("discovery", "confirmation")
            ]
            if any(row["attempts"] != 20 or row["errors"] != 0 for row in values):
                raise ValueError("Latency figure requires the declared 20-request phases")
            writer.writerow((index, *(row["p95_seconds"] for row in values)))
    outage = next(
        row for row in scenarios["records"] if row["run"] == "docker-final-service-unavailable-01"
    )
    if outage["status"] != "passed" or outage["engine"] != "docker":
        raise ValueError("Outage figure requires the observed successful Docker procedure")
    start = outage["observed_signal"]["fault_start"]["elapsed_seconds"]
    firing = outage["alerts"]["firing"]["elapsed_seconds"]
    clearing = outage["alerts"]["clearing"]["elapsed_seconds"]
    recovery = outage["recovery"]["elapsed_seconds"]
    values = {
        "OutageStart": start,
        "OutageFire": firing,
        "OutageClear": clearing,
        "OutageHttp": recovery,
        "OutageDelay": firing - start,
    }
    (output / "data-values.tex").write_text(
        "".join(f"\\newcommand{{\\{key}}}{{{value:.2f}}}\n" for key, value in values.items())
    )


def package_sources(output: Path) -> Path:
    """Package only editable documents and compact versioned figure inputs."""
    paths = sorted(
        set(SOURCE.glob("*.tex"))
        | set(SOURCE.glob("*.py"))
        | set(SOURCE.glob("*.txt"))
        | set(SOURCE.glob("*.md"))
        | {ROOT / name for name in (*EVIDENCE, *DOCUMENTS)}
    )
    target = output / "endesa-delivery-sources-DRAFT.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            member = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(), date_time=(2026, 11, 12, 0, 0, 0)
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, path.read_bytes())
    return target


def build(output: Path) -> dict:
    """Build sequentially and retain logs, source identities and structural checks."""
    if not output.is_relative_to(ROOT / "temporary"):
        raise ValueError("Output must be below this source root's ignored temporary/")
    for tool in ("pdflatex", "pdfinfo", "pdftotext", "qpdf"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required installed tool is unavailable: {tool}")
    output.mkdir(parents=True, exist_ok=True)
    write_chart_data(output)
    env = os.environ.copy()
    env["TEXINPUTS"] = str(SOURCE) + os.pathsep + str(output) + os.pathsep
    env["SOURCE_DATE_EPOCH"] = "1794484800"
    env["FORCE_SOURCE_DATE"] = "1"
    checked = []
    for source_name, stem, limit in (
        ("report.tex", "endesa-final-report-DRAFT", 30),
        ("presentation.tex", "endesa-final-presentation-DRAFT", 16),
    ):
        for attempt in (1, 2):
            with (output / f"{stem}-pass-{attempt}.txt").open("w") as handle:
                subprocess.run(
                    [
                        "pdflatex",
                        "-no-shell-escape",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-jobname=" + stem,
                        str(SOURCE / source_name),
                    ],
                    cwd=output,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=True,
                )
        pdf = output / (stem + ".pdf")
        final_log = (output / f"{stem}-pass-2.txt").read_text()
        if any(marker in final_log for marker in ("Overfull", "undefined references")):
            raise ValueError(f"Unresolved layout/reference diagnostic: {pdf.name}")
        structure = command(["qpdf", "--check", str(pdf)])
        info = command(["pdfinfo", str(pdf)])
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)[1])
        if pages > limit or (source_name == "presentation.tex" and pages != 16):
            raise ValueError(f"Unexpected page count: {pdf.name}: {pages}")
        extracted = output / (stem + ".txt")
        subprocess.run(["pdftotext", "-layout", str(pdf), str(extracted)], check=True, timeout=60)
        if "DRAFT" not in extracted.read_text():
            raise ValueError("Required draft status is missing from extracted PDF")
        checked.append(
            {
                "file": pdf.name,
                "pages": pages,
                "sha256": digest(pdf),
                "qpdf": structure,
                "pdfinfo": info,
            }
        )
    archive = package_sources(output)
    source_commit = None
    dirty = None
    if (ROOT / ".git").exists():
        source_commit = command(["git", "rev-parse", "HEAD"])
        dirty = command(["git", "status", "--porcelain"]).splitlines()
    source_paths = [
        *SOURCE.glob("*.tex"),
        *SOURCE.glob("*.py"),
        *SOURCE.glob("*.txt"),
        *SOURCE.glob("*.md"),
        *(ROOT / name for name in (*EVIDENCE, *DOCUMENTS)),
    ]
    manifest = {
        "status": "DRAFT",
        "implementation_candidate": CANDIDATE,
        "document_source_commit": source_commit,
        "document_source_dirty": dirty,
        "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(source_paths)},
        "pdfs": checked,
        "source_archive": {"file": archive.name, "sha256": digest(archive)},
        "limits": [
            "No model/data retrieval, inference, training or service execution",
            "No registry publication, deployment, submission or human approval",
            "PDF structure/page counts checked; visual review and rehearsal separate",
        ],
    }
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "temporary" / "issue-85" / "artifacts"
    )
    arguments = parser.parse_args()
    result = build(arguments.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "pdfs": [{"file": p["file"], "pages": p["pages"]} for p in result["pdfs"]],
            },
            indent=2,
        )
    )
