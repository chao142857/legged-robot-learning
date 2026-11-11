"""Import the one approved archive incrementally; release only SHA-verified states."""
from pathlib import Path
import hashlib,json,tarfile,time,traceback
ROOT=Path(__file__).resolve().parent
DATA=Path('/media/alex/Sharedata/go2_piper_nsgs_v02_matched_seed48_20261108')
EXPECTED_BYTES=428464362
EXPECTED_SHA='9fd501b688c052cbf54c71608dc5aa9ee65e09099387a282c768c673c0e8d267'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

class GrowingReader:
    def __init__(self,p):self.f=p.open('rb');self.offset=0;self.deadline=time.monotonic()+86400
    def read(self,n):
        n=min(n,EXPECTED_BYTES-self.offset);chunks=[];count=0
        while count<n:
            b=self.f.read(n-count)
            if not b:
                if time.monotonic()>self.deadline:raise TimeoutError('Archive transmission incomplete')
                time.sleep(.2);continue
            chunks.append(b);count+=len(b);self.offset+=len(b)
        return b''.join(chunks)

def main():
    receipts=json.loads((ROOT/'source_export_manifest.json').read_text())['sources']
    expected={f"seed{s['student_seed']}-cp{s['updates']:05d}":s for s in receipts}
    archive=DATA/'matched_sources.tar.gz'
    while not archive.exists() and not archive.with_suffix(archive.suffix+'.partial').exists():time.sleep(1)
    opened=archive if archive.exists() else archive.with_suffix(archive.suffix+'.partial')
    reader=GrowingReader(opened);released=set()
    with tarfile.open(fileobj=reader,mode='r|gz') as t:
        for m in t:
            name=Path(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts
            if name.name=='source_export_manifest.json' and len(name.parts)==1:
                assert json.loads(t.extractfile(m).read())==json.loads((ROOT/'source_export_manifest.json').read_text());continue
            assert len(name.parts)==3 and name.parts[0]=='starts' and name.parts[1] in expected
            assert name.parts[2] in ('prepared.pt','source.json','progress_row.json','export_receipt.json')
            dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            temporary=dest.with_suffix(dest.suffix+'.importing')
            with temporary.open('xb') as f:
                src=t.extractfile(m)
                for b in iter(lambda:src.read(1<<20),b''):f.write(b)
            assert temporary.stat().st_size==m.size;temporary.rename(dest)
            label=name.parts[1];r=expected[label];folder=ROOT/'starts'/label
            if label in released or not all((folder/n).exists() for n in ('prepared.pt','source.json','progress_row.json','export_receipt.json')):continue
            assert (folder/'prepared.pt').stat().st_size==r['exported_bytes']
            assert sha(folder/'prepared.pt')==r['exported_prepared_sha256']
            assert sha(folder/'progress_row.json')==r['progress_row_sha256']
            assert json.loads((folder/'export_receipt.json').read_text())==r
            original=json.loads((folder/'source.json').read_text())
            assert original['sha256']==r['original_source_checkpoint_sha256']
            # Retain original source metadata and expose the relocated prepared file.
            sources=json.loads((ROOT/'sources.json').read_text())['sources']
            mapped=[s for s in sources if (s['seed'],s['updates'])==(r['student_seed'],r['updates'])];assert len(mapped)==1
            assert mapped[0]['original_source']==original
            (folder/'original_source.json').write_text(json.dumps(original,indent=2)+'\n')
            (folder/'source.json').write_text(json.dumps(mapped[0],indent=2)+'\n')
            marker=dict(passed=True,checkpoint_sha256=r['exported_prepared_sha256'],all_source_files_complete=True)
            (folder/'IMPORT_READY.json').write_text(json.dumps(marker,indent=2)+'\n')
            released.add(label);print('SOURCE_SHA_READY',label,flush=True)
    assert len(released)==8
    while reader.offset<EXPECTED_BYTES:reader.read(min(1<<20,EXPECTED_BYTES-reader.offset))
    # The open descriptor survives the uploader's atomic .partial -> final rename.
    reader.f.seek(0);h=hashlib.sha256()
    for b in iter(lambda:reader.f.read(1<<20),b''):h.update(b)
    assert h.hexdigest()==EXPECTED_SHA
    (ROOT/'IMPORT_COMPLETE.json').write_text(json.dumps(dict(passed=True,archive_sha256=h.hexdigest(),archive_bytes=EXPECTED_BYTES,sources=8),indent=2)+'\n')
    print('ALL_IMPORTS_AND_BUNDLE_SHA_PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (ROOT/'IMPORT_FAULT.txt').write_text(traceback.format_exc());raise
