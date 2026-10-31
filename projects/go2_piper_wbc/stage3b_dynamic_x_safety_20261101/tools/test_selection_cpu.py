"""Contract counterexamples for blind-test selection, with no real study I/O."""
import pathlib, sys, types, tempfile

# The native report runner is unnecessary for these in-memory counterexamples.
sys.modules['stage3_pipeline'] = types.SimpleNamespace(report=lambda *_: None)
import stage3b_selection as selector


def fixture(entries, hard_labels=()):
    nodes=[]; rows=[]
    for label, experiment, seed, recipe, count, gain, passed in entries:
        cp=dict(checkpoint='/CPU_FIXTURE/'+label+'.pt', checkpoint_sha256=label)
        nodes.append(dict(label=label, evaluation_prefix=label, checkpoint=cp,
            recipe_id=recipe, training_seed=seed, experiment_id=experiment,
            additional_actual_updates=count, actual_checkpoint_iteration=1800+count,
            decision=dict(original_all_development_gates=passed)))
        for dev in (6181031,6281031):
            for protocol in ('A','T0','T1','X'):
                rows.append(dict(label=label+'-'+protocol+'-'+str(dev), protocol=protocol,
                    seed=dev, checkpoint_sha256=label, joint_limit=int(label in hard_labels),
                    nonfoot=0, strict_safe=64, n=64, complete=64,
                    retention_checks=dict(original_health=True), X_after1s=dict(synchronous_gain=gain)))
    selector.read=lambda _:nodes
    selector.report=lambda *_:dict(rows=rows)
    with tempfile.TemporaryDirectory(prefix='CPU-only-selection-fixture-') as tmp:
        root=pathlib.Path(tmp); (root/'CANDIDATE_NODES.json').touch()
        return selector.selection(types.SimpleNamespace(p=root))


qualified=[('A100','A',41,'recipe',100,.4,True),('A200','A',41,'recipe',200,.5,True)]
single_better=[('B100','B',42,'recipe',100,.2,False),('B200','B',42,'recipe',200,.9,True)]
selected=fixture(qualified+single_better)
assert selected['candidate']['labels']==['A200'] and selected['adjacent']['labels']==['A100']
assert selected['candidate']['consecutive_development_pass']

# A predecessor with an actual hard event cannot qualify a pair, even when a
# native aggregate gate is marked true. Original final checks remain separate.
selected=fixture(qualified+single_better,hard_labels=('A100',))
assert selected['candidate']['labels']==['B200']
assert not selected['candidate']['consecutive_development_pass']

# Similar experiment labels must never join different training seeds/recipes.
for changed in [('different_seed',42,'recipe'),('different_recipe',41,'other')]:
    name,seed,recipe=changed
    selected=fixture([qualified[0],('A200','A',seed,recipe,200,.9,True)])
    assert not selected['candidate']['consecutive_development_pass']
    assert selected['adjacent'] is None
print('CPU selection contract PASS: consecutive gates, actual hard-event exclusion, same seed/recipe; no GPU or real-study writes')
