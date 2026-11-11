"""Place only new transfer outputs on the local data partition, via stable links."""
import json
from pathlib import Path

def prepare_output(args,root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1]
    config=root/'STORAGE.json'
    if args.mode!='transfer' or not config.exists():
        args.out.mkdir(parents=True,exist_ok=False);return
    storage=json.loads(config.read_text());destination=Path(storage['data_root'])
    assert args.out.parent==root/'runs' and args.dataset.parent==root/'experience'
    real_out=destination/'runs'/args.out.name;real_dataset=destination/'experience'/args.dataset.name
    assert not args.out.exists() and not args.out.is_symlink()
    assert not args.dataset.exists() and not args.dataset.is_symlink()
    assert not real_out.exists() and not real_dataset.exists()
    real_out.mkdir(parents=True,exist_ok=False);real_dataset.parent.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.dataset.parent.mkdir(parents=True,exist_ok=True)
    args.out.symlink_to(real_out,target_is_directory=True)
    args.dataset.symlink_to(real_dataset,target_is_directory=True)
    # train() creates its new dataset exactly once, on the data partition.
    args.dataset=real_dataset
