import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_transition_thresholds(interval: int, budget: int) -> tuple[int, ...]:
    interval = int(interval)
    budget = int(budget)
    if interval < 1 or budget < 1:
        raise ValueError("Snapshot interval and transition budget must be positive")
    thresholds = list(range(interval, budget + 1, interval))
    if not thresholds or thresholds[-1] != budget:
        thresholds.append(budget)
    return tuple(thresholds)


def _relative_file(root: Path, path: Path) -> Path:
    path = path.resolve()
    try:
        return path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Snapshot policy must be written inside {root}: {path}") from exc


def load_snapshot_record(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    record_path = directory / "snapshot.json" if directory.is_dir() else directory
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read snapshot record: {record_path}") from exc
    required = {
        "schema_version",
        "backend",
        "algorithm",
        "requested_transition",
        "accepted_transition",
        "completed_episodes",
        "policy",
        "files",
        "created_at_utc",
    }
    if set(record) != required or record.get("schema_version") != 1:
        raise ValueError(f"Malformed transition snapshot record: {record_path}")
    if int(record["accepted_transition"]) < int(record["requested_transition"]):
        raise ValueError(f"Snapshot precedes its requested transition: {record_path}")
    policy = record.get("policy")
    files = record.get("files")
    if not isinstance(policy, dict) or set(policy) != {"path", "sha256"}:
        raise ValueError(f"Malformed policy entry in {record_path}")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Snapshot contains no files: {record_path}")
    seen = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError(f"Malformed file entry in {record_path}")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative in seen:
            raise ValueError(f"Unsafe or duplicate snapshot path in {record_path}")
        seen.add(relative)
        candidate = record_path.parent / relative
        if not candidate.is_file():
            raise ValueError(f"Snapshot file is missing: {candidate}")
        if candidate.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Snapshot file size changed: {candidate}")
        if sha256_file(candidate) != str(item["sha256"]):
            raise ValueError(f"Snapshot checksum changed: {candidate}")
    policy_path = Path(str(policy["path"]))
    if policy_path not in seen:
        raise ValueError(f"Primary policy is absent from snapshot files: {record_path}")
    if str(policy["sha256"]) != next(
        str(item["sha256"]) for item in files if Path(str(item["path"])) == policy_path
    ):
        raise ValueError(f"Primary policy checksum disagrees with file inventory: {record_path}")
    return record


class TransitionSnapshotPublisher:
    """Publish an immutable policy snapshot at each crossed transition threshold."""

    def __init__(
        self,
        root: str | Path,
        *,
        interval: int,
        budget: int,
        backend: str,
        algorithm: str,
    ):
        # Keep callback paths unambiguous even when the CLI receives a relative root.
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.interval = int(interval)
        self.budget = int(budget)
        self.backend = str(backend)
        self.algorithm = str(algorithm)
        if not self.backend or not self.algorithm:
            raise ValueError("Snapshot backend and algorithm cannot be empty")
        self.thresholds = expected_transition_thresholds(self.interval, self.budget)
        staging = list(self.root.glob(".transitions-*.tmp-*"))
        if staging:
            raise RuntimeError(
                "Incomplete transition snapshot staging directory found: "
                + ", ".join(str(path) for path in staging)
            )
        self.published = self._load_published_prefix()

    def _load_published_prefix(self) -> set[int]:
        found = {}
        for directory in self.root.glob("transitions-*"):
            if not directory.is_dir():
                continue
            try:
                threshold = int(directory.name.removeprefix("transitions-"))
            except ValueError as exc:
                raise RuntimeError(f"Malformed snapshot directory: {directory}") from exc
            if threshold not in self.thresholds:
                raise RuntimeError(f"Unexpected snapshot threshold: {directory}")
            record = load_snapshot_record(directory)
            if int(record["requested_transition"]) != threshold:
                raise RuntimeError(f"Snapshot directory and record disagree: {directory}")
            if record["backend"] != self.backend or record["algorithm"] != self.algorithm:
                raise RuntimeError(f"Snapshot identity changed: {directory}")
            found[threshold] = record
        ordered = [threshold for threshold in self.thresholds if threshold in found]
        if ordered != list(self.thresholds[: len(ordered)]):
            raise RuntimeError("Transition snapshots must form a contiguous prefix")
        previous_accepted = -1
        for threshold in ordered:
            accepted = int(found[threshold]["accepted_transition"])
            if accepted < previous_accepted:
                raise RuntimeError("Snapshot accepted-transition counters are not monotonic")
            previous_accepted = accepted
        return set(ordered)

    @property
    def complete(self) -> bool:
        return len(self.published) == len(self.thresholds)

    def due(self, accepted_transition: int) -> tuple[int, ...]:
        accepted = min(int(accepted_transition), self.budget)
        return tuple(
            threshold
            for threshold in self.thresholds
            if threshold <= accepted and threshold not in self.published
        )

    def publish_due(
        self,
        accepted_transition: int,
        completed_episodes: int,
        save_policy: Callable[[Path, int], str | Path],
    ) -> list[dict[str, Any]]:
        records = []
        for threshold in self.due(accepted_transition):
            temporary = self.root / f".transitions-{threshold}.tmp-{uuid.uuid4().hex}"
            destination = self.root / f"transitions-{threshold}"
            temporary.mkdir()
            try:
                policy_path = Path(save_policy(temporary, threshold))
                if not policy_path.is_absolute():
                    policy_path = temporary / policy_path
                policy_relative = _relative_file(temporary, policy_path)
                files = []
                for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
                    relative = path.relative_to(temporary)
                    files.append(
                        {
                            "path": relative.as_posix(),
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                    )
                if not files:
                    raise RuntimeError("Snapshot callback did not write any files")
                policy_item = next(
                    (item for item in files if item["path"] == policy_relative.as_posix()),
                    None,
                )
                if policy_item is None:
                    raise RuntimeError("Snapshot callback did not return a written policy file")
                record = {
                    "schema_version": 1,
                    "backend": self.backend,
                    "algorithm": self.algorithm,
                    "requested_transition": threshold,
                    "accepted_transition": int(accepted_transition),
                    "completed_episodes": int(completed_episodes),
                    "policy": {
                        "path": policy_relative.as_posix(),
                        "sha256": policy_item["sha256"],
                    },
                    "files": files,
                    "created_at_utc": utc_now(),
                }
                (temporary / "snapshot.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                try:
                    temporary.rename(destination)
                except FileExistsError as exc:
                    raise RuntimeError(f"Snapshot already exists: {destination}") from exc
                load_snapshot_record(destination)
                self.published.add(threshold)
                records.append(record)
            except Exception:
                # Keep incomplete staging evidence. A later resume refuses it explicitly.
                raise
        return records
