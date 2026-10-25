"""Stable exact-task identity and evidence-bound physical admission.

Admission binds the task parameters, implementation, robot files and actual
geometry/kinematic/physical evidence. It is never inferred from a profile label.
"""
import hashlib
import json
from pathlib import Path

RUNTIME_KEYS = {'num_envs', 'seed', 'evaluation', 'device', 'log_dir', 'episode_cap', 'steps',
                'admission', 'admitted', 'validation', 'validation_path'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def task_identity(metadata):
    # Only the top-level run seed is excluded. Nested seeds can define actual
    # terrain/reset geometry and must remain part of the exact task identity.
    spec = {key: value for key, value in metadata.items() if key not in RUNTIME_KEYS}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()
    return key, spec


def check_admission(path, key, *, source_files, asset_files):
    admission_path = Path(path)
    record = json.loads(admission_path.read_text())
    if record.get('admission_key') != key:
        raise ValueError('Admission does not match the exact task parameters')
    for name in ('geometry_pass', 'kinematics_pass', 'physics_pass'):
        if record.get(name) is not True:
            raise ValueError('Training blocked: admission requires ' + name)
    bindings = record.get('bindings', {})
    for group, current_files in [('sources', source_files), ('assets', asset_files)]:
        if not current_files:
            raise ValueError('Admission requires nonempty current ' + group)
        recorded = bindings.get(group, {})
        if set(recorded) != set(current_files):
            raise ValueError('Admission ' + group + ' file set differs from the current run')
        for name, current_path in current_files.items():
            if not Path(current_path).is_file():
                raise ValueError('Admission dependency is missing: ' + name)
            if recorded[name] != file_sha256(current_path):
                raise ValueError('Admission is stale: ' + group + '/' + name)
    evidence = record.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Admission requires actual evidence files and SHA256 hashes')
    for item in evidence:
        if not isinstance(item, dict) or not item.get('path') or not item.get('sha256'):
            raise ValueError('Malformed admission evidence entry')
        evidence_path = Path(item['path'])
        if not evidence_path.is_absolute():
            evidence_path = admission_path.parent / evidence_path
        if not evidence_path.is_file():
            raise ValueError('Admission evidence is missing: ' + str(evidence_path))
        if file_sha256(evidence_path) != item['sha256']:
            raise ValueError('Admission evidence hash mismatch: ' + str(evidence_path))
    return record
