"""Checkpoint the environment only after native stepping initialized its buffers."""


def capture_environment(completed, raw, logger, capture_fn):
    assert isinstance(completed, int) and completed >= 0
    # Before the first PPO update there is no learned state to resume. Preserve
    # initialization separately and recreate from its recorded seed on failure;
    # do not fabricate native step buffers or execute an extra physics step.
    if completed == 0:
        return None
    return capture_fn(raw, logger)
