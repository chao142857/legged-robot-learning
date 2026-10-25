"""Atomic composition of the owners' module-state APIs, without physics claims."""
import copy

def capture_module_state(session, controller, metrics, *, protocol_sha256, dependency_lock_sha256):
    return dict(schema_version=1,run_id=session.run_id,support_sha256=session.pool.support_sha256,
        protocol_sha256=protocol_sha256,dependency_lock_sha256=dependency_lock_sha256,
        sampler=session.state_dict(),controller=controller.state_dict(),metrics=metrics.state_dict(),
        restore_scope='modules only; native command/action buffers and simulator require separate restore or censor/reset')

def restore_module_state(state, session, controller, metrics, *, protocol_sha256, dependency_lock_sha256):
    expected={'schema_version':1,'run_id':session.run_id,'support_sha256':session.pool.support_sha256,
              'protocol_sha256':protocol_sha256,'dependency_lock_sha256':dependency_lock_sha256}
    if any(state.get(k)!=v for k,v in expected.items()):raise ValueError('Combined module state binding mismatch')
    backup=(session.state_dict(),controller.state_dict(),metrics.state_dict())
    try:
        session.load_state_dict(copy.deepcopy(state['sampler']))
        controller.load_state_dict(copy.deepcopy(state['controller']))
        metrics.load_state_dict(copy.deepcopy(state['metrics']))
    except Exception:
        session.load_state_dict(backup[0]);controller.load_state_dict(backup[1]);metrics.load_state_dict(backup[2])
        raise
