from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=True)
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
