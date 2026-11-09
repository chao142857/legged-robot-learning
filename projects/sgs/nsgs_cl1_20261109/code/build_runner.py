"""Deterministic, count-checked adaptation of the verified native runner."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
source=root/'code/train_v02.py';s=source.read_text();changes=[]
def change(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new);changes.append(old)
change('from teacher_v02 import Teacher','from cl1_proxy import Teacher')
change('teacher = Teacher(static, binding)','teacher = Teacher(static, binding, out, args.student_seed)')
change("phase = 'initialized'","phase = 'initialized'\n    latest_row = None")
change("source_checkpoint_sha256=source_hash, teacher=teacher.state() if teacher else None,","source_checkpoint_sha256=source_hash, latest_row=latest_row, teacher=teacher.state() if teacher else None,")
change('result = teacher.decide(dynamic, g, engine.version+1,','teacher.dev = dev_before\n            result = teacher.decide(dynamic, g, engine.version+1,')
change("decision = info['decision']","decision = info['decision']\n                latest_row = info.get('latest_row')")
change("        preserve_decision()\n        phase = 'ready'", "        if args.resume:\n            decision = new_decision(latest_row)\n            adapter.owner.engine.install(decision['probability'], decision['version'])\n            adapter.owner.cycle_version_exposure = {}\n        preserve_decision()\n        phase = 'ready'")
change("                dev_before = dev_after","                dynamic_after, global_after = global_state(row, dev_after)\n                teacher.complete(decision, dynamic_after, global_after, dev_after, completed)\n                dev_before = dev_after")
change("if not branch and (completed % 1000 == 0 or completed == args.target):","if not branch and completed % 1000 == 0:")
change('        def recorded_log(*a, **kw):\n            native_log', '        def recorded_log(*a, **kw):\n            nonlocal latest_row\n            native_log')
change("            emit(out/'updates.jsonl', row)","            latest_row = row\n            emit(out/'updates.jsonl', row)")
(root/'code/train_cl1.py').write_text(s)
(root/'RUNNER_ADAPTATION.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(s.encode()).hexdigest(),count_checked_replacements=changes),indent=2))
