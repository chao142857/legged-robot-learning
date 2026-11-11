"""Bind the already fixed pools and audit sources without fitting to outcomes."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from common import ROOT, PROTOCOL, digest, file_sha, write
from teacher_v02 import Teacher, V02_CONFIG, array_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-index', type=int)
    args=parser.parse_args()
    sources=json.loads((ROOT/'sources.json').read_text())['sources']
    if args.source_index is None:
        assert len(sources)==8 and {s['seed'] for s in sources}=={43,44}
        parent=Path('/data/go2_piper_nsgs_v0_20261106')
        old=json.loads((parent/'prepared/manifest.json').read_text())
        (ROOT/'prepared').mkdir(exist_ok=False)
        for name in ('command_features.npy','feature_normalization.npz','wbc_development.npz'):
            shutil.copy2(parent/'prepared'/name, ROOT/'prepared'/name)
            assert file_sha(ROOT/'prepared'/name)==file_sha(parent/'prepared'/name)
        binding=dict(old['binding'], version=V02_CONFIG['version'], v02_config_sha256=digest(V02_CONFIG))
        assert binding['protocol_sha256']==digest(PROTOCOL)
        write(ROOT/'prepared/manifest.json', dict(binding=binding, parent_binding=old['binding'],
              parent_manifest_sha256=file_sha(parent/'prepared/manifest.json'),
              fixed_development_pool_reused=True, formal_test_excluded_from_teacher=True,
              source_order=[dict(seed=s['seed'],updates=s['updates'],sha256=s['sha256']) for s in sources]))
        (ROOT/'parent').mkdir(exist_ok=False)
        shutil.copy2(parent/'runs/fresh-seed45/configuration.json',ROOT/'parent/reference_configuration.json')
        static=np.load(ROOT/'prepared/command_features.npy',allow_pickle=False)
        teacher=Teacher(static,binding)
        (ROOT/'teachers').mkdir(exist_ok=False)
        torch.save(dict(teacher=teacher.state(),purpose='Frozen Fresh Teacher; created before any branch result.'),ROOT/'teachers/fresh.pt')
        write(ROOT/'teachers/initial.json',dict(config=V02_CONFIG,sha256=file_sha(ROOT/'teachers/fresh.pt'),
              seed=V02_CONFIG['teacher_seed'],command_features_sha256=file_sha(ROOT/'prepared/command_features.npy')))
        checked=set()
        for s in sources:
            assert file_sha(s['path'])==s['sha256']
            assert file_sha(s['configuration_path'])==s['configuration_sha256']
            for f in s['feedback_files']:
                if f['path'] not in checked:
                    assert file_sha(f['path'])==f['sha256']
                    checked.add(f['path'])
        print('V02_PREPARE_PASS',flush=True)
    else:
        from fork_v02 import prepare_source
        s=sources[args.source_index]
        assert file_sha(s['path'])==s['sha256']
        out=ROOT/'starts'/f"seed{s['seed']}-cp{s['updates']:05d}"
        out.mkdir(parents=True,exist_ok=False)
        receipt=prepare_source(Path(s['path']),[Path(f['path']) for f in s['feedback_files']],
            out/'prepared.pt',sampler_seed=204800+args.source_index,student_seed=s['seed'])
        write(out/'source.json',s)
        write(out/'preparation.json',receipt)
        print('V02_SOURCE_PREPARED '+str(out),flush=True)


if __name__=='__main__': main()
