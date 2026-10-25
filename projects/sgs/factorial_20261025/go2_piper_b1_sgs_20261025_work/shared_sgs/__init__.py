"""B1/B2 shared SGS, controller independent and CPU only."""
from .catalog import TaskSpec, TaskPool
from .feedback import T1_RULE_VERSION, t1_episode_labels
from .session import EpisodeAssignment, TerminalFeedback, SamplerSession, FixedEvaluation

__version__ = "1.0.0-rc1"
__all__ = ["TaskSpec", "TaskPool", "EpisodeAssignment", "TerminalFeedback",
           "SamplerSession", "FixedEvaluation", "T1_RULE_VERSION", "t1_episode_labels"]
