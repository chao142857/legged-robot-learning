"""Verify migrated source identities, documentation targets and published numbers."""
from pathlib import Path
import ast,hashlib,json,re,statistics,sys
r=Path(__file__).resolve().parents[1]
def read(p):return json.loads((r/p).read_text())
errors=[]
def check(ok,message):
 if not ok:errors.append(message)
for row in read('docs/source-map.json'):
 p=r/row['path']
 check(p.is_file(),'Missing mapped file: '+row['path'])
 if p.is_file():check(hashlib.sha256(p.read_bytes()).hexdigest()==row['published_sha256'],'Changed mapped file: '+row['path'])
for p in r.rglob('*.md'):
 if '.git' in p.parts:continue
 for url in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  if url.startswith(('http://','https://','mailto:','#')):continue
  target=url.split('#')[0]
  check(not target.startswith('/'),'Absolute documentation link: '+str(p.relative_to(r))+': '+url)
  check((p.parent/target).exists(),'Broken documentation target: '+str(p.relative_to(r))+': '+url)
for folder in ('src','experiments','tests','tools'):
 for p in (r/folder).rglob('*.py'):
  try:ast.parse(p.read_text(),filename=str(p))
  except SyntaxError as e:errors.append(str(e))
d=read('results/cleanmeta8/curve_data.json');check(d['student_seeds']==[42,47,48,49],'Seed identity')
source=read('results/cleanmeta8/source_inputs/go2_piper_cleanmeta8_mean_seed47_48_49_20261110/curve_data.json')
for seed in ('47','48','49'):check(d['seeds'][seed]==source['seeds'][seed],'Three-seed source mismatch '+seed)
seed42=read('results/cleanmeta8/source_inputs/go2_piper_cleanmeta8_seed42_20261111_work/completion_verified.json')
for method,arm in [('fresh','fresh'),('cleanmeta8','clean-meta8')]:
 rows=seed42['runs'][arm]['tests'];check([x['joint_success_rate'] for x in rows]==d['seeds']['42'][method],'Seed42 source mismatch '+method)
u=read('results/cleanmeta8/source_inputs/go2_piper_cleanmeta8_seed42_20261111_work/uniform_baseline.json')
check([x['joint_success_rate'] for x in u['points']]==d['seeds']['42']['uniform'],'Uniform seed42 source')
for method,expected in [('uniform','31.88'),('fresh','33.30'),('cleanmeta8','33.50')]:
 mean=statistics.mean(d['seeds'][str(s)][method][-1] for s in d['student_seeds'])
 check(f'{100*mean:.2f}'==expected,'Mean mismatch '+method)
 print(method,format(100*mean,'.8f')+'%')
for row in read('results/cleanmeta8/source_reconciliation.json'):
 p=Path(row['recovered_path']);local=r/'results/cleanmeta8/source_inputs'/p.parent.name/p.name
 check(hashlib.sha256(local.read_bytes()).hexdigest()==row['original']['sha256'],'Recovered source hash')
for arm,expected in [('uniform',.345703125),('frozen-base',.310546875),('frozen-recovery',.35546875)]:
 p=f'results/cl1/transfer70158/evaluations/{arm}-seed70158-to10000-a01-10000-test/result.json';x=read(p)
 check(x['passed'] and x['update']==10000 and x['episodes']==1024 and x['kind']=='test','CL1 endpoint identity '+arm)
 check(x['joint_success_rate']==expected,'CL1 value '+arm)
 check(x['protocol_sha256']=='93067793642c17e12618129548914f7b440edce267ad20d4df34b8a19c855fad','CL1 protocol '+arm)
 check(x['pool_sha256']=='d48b5850285e26b9cec447a0bb4bd6c84493afec5efa0c7725ddcbacbd714afa','CL1 pool '+arm)
if errors:
 print('\n'.join(errors));sys.exit(1)
print('Source hashes, Python syntax, Markdown targets and numerical evidence verified.')
