"""Content identities, independent of environment slots, episodes and labels."""
from dataclasses import dataclass
import hashlib
import json
import math


def canonical(value):
    def normalized(v):
        if v is None or isinstance(v, (str, bool)):
            return v
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                raise ValueError("Nonfinite task content")
            return int(v) if int(v) == v else v
        if isinstance(v, (tuple, list)):
            return [normalized(x) for x in v]
        if isinstance(v, dict) and all(isinstance(k, str) for k in v):
            return {k: normalized(x) for k, x in v.items()}
        raise ValueError("Task content must be finite JSON")
    return json.dumps(normalized(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    content_json: str

    @classmethod
    def from_content(cls, content):
        if not isinstance(content, dict) or not content:
            raise ValueError("Task content must be a nonempty semantic object")
        forbidden = {"env_id", "environment_id", "episode_id", "task_id", "run_id", "seed", "label"}
        if forbidden.intersection(content):
            raise ValueError("Bookkeeping and stochastic seed fields are not task content")
        return cls(canonical(content))

    def __post_init__(self):
        if canonical(json.loads(self.content_json)) != self.content_json:
            raise ValueError("Noncanonical task content")

    @property
    def content(self):
        return json.loads(self.content_json)

    @property
    def content_sha256(self):
        return hashlib.sha256(self.content_json.encode()).hexdigest()

    @property
    def task_id(self):
        return "task:" + self.content_sha256


class TaskPool:
    def __init__(self, tasks, support_version):
        self.tasks = tuple(tasks)
        if not self.tasks or not support_version or not all(isinstance(t, TaskSpec) for t in self.tasks):
            raise ValueError("Nonempty typed task pool and support version required")
        self.support_version = support_version
        self.ids = tuple(t.task_id for t in self.tasks)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate semantic task content; aliases are not new tasks")
        self.index = {key: i for i, key in enumerate(self.ids)}
        self.by_id = {t.task_id: t for t in self.tasks}
        self.support_sha256 = digest(self.manifest())

    def manifest(self):
        return {"schema_version": 1, "support_version": self.support_version,
                "tasks": [{"task_id": t.task_id, "content_sha256": t.content_sha256,
                           "content": t.content} for t in self.tasks]}
