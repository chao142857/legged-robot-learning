"""Source binding for the publication's independent directory layout."""
import hashlib
from pathlib import Path


def release_source_hashes():
    root = Path(__file__).resolve().parent
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.json', '.sh')
        and 'results' not in p.parts and '__pycache__' not in p.parts
    }
