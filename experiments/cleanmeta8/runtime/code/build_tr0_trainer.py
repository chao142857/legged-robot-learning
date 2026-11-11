"""Generate an isolated local-continuation adapter from the audited native runner.

Only lifecycle/measurement code changes: a frozen Teacher may continue a full
Student state for 500 updates, refresh q every100 and request development at
+100/+300/+500. PPO, rewards, actions, observations and native env cfg are copied.
The original file is never edited. All replacements require an exact match.
"""
from pathlib import Path
import argparse,hashlib,json

def build(source,destination):
 text=Path(source).read_text();changes=[]
 def replace(old,new):
  nonlocal text
  assert text.count(old)==1,old[:140]
  text=text.replace(old,new);changes.append(dict(old=old,new=new))
 replace("parser.add_argument('--smoke', action='store_true')", "parser.add_argument('--smoke', action='store_true')\nparser.add_argument('--tr0', action='store_true')\nparser.add_argument('--source-row',type=Path)\nparser.add_argument('--baseline-dev',type=Path)\nparser.add_argument('--repeat-seed',type=int)")
 replace('    assert branch or args.source is None','    assert branch or args.source is None or args.tr0')
 replace('    out, dataset = args.out, args.dataset',"    if args.tr0:\n        assert not branch and (args.source or args.resume) and args.teacher_in\n        assert args.source_row and args.baseline_dev\n    out, dataset = args.out, args.dataset")
 old="""            else:
                assert info['v02_preparation']['source_checkpoint_sha256']
                assert adapter.owner.engine.version == -1
                decision = dict(version=0, probability=probability.copy(), distribution=distribution_stats(probability),
                    probability_sha256=file_sha(args.probability), decision_source=str(args.decision_source),
                    decision_source_sha256=file_sha(args.decision_source) if args.decision_source else None,
                    teacher_frozen=True, source_checkpoint_sha256=source_hash)
                previous_transitions = adapter.owner.transitions.copy()
                adapter.owner.engine.install(probability, 0)
                adapter.owner.cycle_version_exposure = {}"""
 new="""            else:
                assert info['v02_preparation']['source_checkpoint_sha256']
                assert adapter.owner.engine.version == -1
                if args.tr0:
                    assert args.smoke or args.target-completed == 500
                    receipt['original_global_rng_sha256'] = tree_hash(rng_state())
                    if args.repeat_seed is not None:
                        import random
                        random.seed(args.repeat_seed)
                        np.random.seed(args.repeat_seed)
                        torch.manual_seed(args.repeat_seed)
                        torch.cuda.manual_seed_all(args.repeat_seed)
                        adapter.owner.engine.rng = np.random.default_rng(args.repeat_seed+1000)
                    receipt['repeat_seed'] = args.repeat_seed
                    receipt['execution_global_rng_sha256'] = tree_hash(rng_state())
                    # Evaluation runs separately; it does not change Student RNG.
                    measured_baseline = evaluate(frozen_actor(), 'development')
                    dev_before = json.loads(args.baseline_dev.read_text())
                    assert dev_before['passed'] and dev_before['kind'] == 'development'
                    assert dev_before['update'] == completed
                    assert dev_before['actor_sha256'] == measured_baseline['actor_sha256'] == tensor_hash(actor.state_dict())
                    assert dev_before['pool_sha256'] == binding['development_pool_sha256']
                    write(out/'baseline_restore_noise.json',dict(fixed_baseline=dev_before,
                        repeated_baseline=measured_baseline,scope='Fixed predeclared baseline conditions every Teacher; repeated evaluation diagnoses reset/PhysX variation.'))
                    row = json.loads(args.source_row.read_text())
                    assert row['completed_updates'] == completed
                    decision = new_decision(row)
                    assert decision['version'] == 0
                    probability = decision['probability']
                    write(out/'starting_student_state.json',dict(actor_sha256=tensor_hash(actor.state_dict()),
                        critic_sha256=tensor_hash(critic.state_dict()),optimizer_sha256=tree_hash(runner.alg.optimizer.state_dict()),
                        source_checkpoint_sha256=source_hash,source_update=completed,repeat_seed=args.repeat_seed,
                        original_global_rng_sha256=receipt['original_global_rng_sha256'],
                        execution_global_rng_sha256=receipt['execution_global_rng_sha256']))
                else:
                    decision = dict(version=0, probability=probability.copy(), distribution=distribution_stats(probability),
                        probability_sha256=file_sha(args.probability), decision_source=str(args.decision_source),
                        decision_source_sha256=file_sha(args.decision_source) if args.decision_source else None,
                        teacher_frozen=True, source_checkpoint_sha256=source_hash)
                previous_transitions = adapter.owner.transitions.copy()
                adapter.owner.engine.install(probability, 0)
                adapter.owner.cycle_version_exposure = {}"""
 replace(old,new)
 old="""                dev_after = evaluate(actor_path, 'development')
                assert dev_after['initial_sha256'] == dev_before['initial_sha256']
                outcome.update(dev_before=dev_before['continuous_joint_score'], dev_after=dev_after['continuous_joint_score'],
                    dev_strict_success=dev_after['joint_success_rate'])
                dev_before = dev_after
                assert tree_hash(teacher.state()) == teacher_identity"""
 new="""                if not args.tr0 or args.smoke or completed-base_update in (100,300,500):
                    dev_after = evaluate(actor_path, 'development')
                    assert dev_after['initial_sha256'] == dev_before['initial_sha256']
                    outcome.update(dev_before=dev_before['continuous_joint_score'], dev_after=dev_after['continuous_joint_score'],
                        dev_strict_success=dev_after['joint_success_rate'])
                    dev_before = dev_after
                else:
                    outcome['development_not_scheduled'] = True
                assert tree_hash(teacher.state()) == teacher_identity"""
 replace(old,new)
 replace("            if not branch and (completed % 1000 == 0 or completed == args.target):", "            if not branch and not args.tr0 and (completed % 1000 == 0 or completed == args.target):")
 replace("        print('NSGS_V02_TRAIN_PASS '+str(completed), flush=True)","        print('NSGS_TR0_LOCAL_TRAIN_PASS '+str(completed), flush=True)")
 compile(text,str(destination),'exec');destination=Path(destination);assert not destination.exists()
 destination.write_text(text)
 receipt=dict(source=str(source),source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
  generated=str(destination),generated_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),replacements=changes,
  unchanged_student='PPO/model/native cfg/reward/observations/action and normalization loading unchanged')
 destination.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();print(json.dumps(build(a.source,a.destination)))
