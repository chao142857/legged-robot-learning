"""Validate immutable archived evidence with an explicit local path map."""
import hashlib, json, os
from pathlib import Path
BASE = Path(os.environ['Z1_PRIVATE_RESTORE_ROOT'])
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate():
    manifest = json.loads((BASE/'payload.manifest.json').read_text())
    # Manifest structure is checked in setup; critical runtime hashes also match passed physical evidence.
    ops = BASE/'workspace/ops'
    asset = BASE/'workspace/basic-locomotion-isaaclab/robot_model/custom/z1_team_v6/Z1_team/Z1_team.usda'
    for item in manifest['files']:
        if '/z1_team_v6/' in item['file']:
            path = BASE/item['file']
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['file']
    for name in ('z1-team-physics.audit.json', 'z1-team-controls.audit.json'):
        audit = json.loads((BASE/'reference'/name).read_text())
        assert audit['passed'] and audit['self_collision_enabled']
        assert sha(asset) == audit['asset_sha256']
        assert sha(ops/'z1_team_cfg.py') == audit['config_source_sha256']
        assert sha(ops/'z1_cfg.py') == audit['shared_z1_config_sha256']
    cp = BASE/'checkpoints/cp5700/model_5700.pt'
    assert sha(cp) == 'ba964a31120661a3f8139583b5b7f9865df2764b369bf5ea71592e2ac0329134'
    return {'passed': True, 'asset': str(asset), 'checkpoint': str(cp),
            'asset_sha256': sha(asset), 'checkpoint_sha256': sha(cp),
            'mapping_only': True, 'no_physical_revalidation_claim': True}
if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
