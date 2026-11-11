import hashlib,json,time,numpy as np
from common import ROOT,Pool,write
rows=[]
for kind in ('flat','wbc'):
    a=Pool(kind,'train');b=Pool(kind,'evaluation')
    def ids(pool):return {hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest() for x,y in zip(pool.ee,pool.velocity)}
    training=ids(a);evaluation=ids(b)
    assert len(training)==32768 and len(evaluation)==1024 and not training&evaluation
    assert np.all((a.ee>=0)&(a.ee<1)) and np.all((a.velocity>=0)&(a.velocity<1))
    rows.append(dict(kind=kind,train_sequences=len(training),eval_sequences=len(evaluation),overlapping_sequences=0,train_sha256=a.sha,eval_sha256=b.sha,passed=True))
write(ROOT/'audit/pool_independence.json',dict(checked_unix=time.time(),results=rows))
print(json.dumps(rows),flush=True)
