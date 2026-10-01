"""Private entrypoint for one isolated AutoML trial subprocess."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


class _DiscardResultChannel:
    """Worker-compatible channel; durable result JSON is the IPC contract."""

    def send(self, result) -> None:  # noqa: ANN001 - worker protocol is intentionally structural
        return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m torch_like.automl._trial_process PAYLOAD.pkl")

    payload_path = Path(args[0])
    with payload_path.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - trusted, parent-created local IPC payload

    # Import after unpickling starts the child with the smallest possible module
    # surface while still resolving the dataclasses embedded in the payload.
    from torch_like.automl.task import _cv_trial_worker, _trial_worker

    worker = _cv_trial_worker if "folds" in payload else _trial_worker
    worker(payload, _DiscardResultChannel())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
