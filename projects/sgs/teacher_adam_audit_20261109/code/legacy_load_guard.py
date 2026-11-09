"""Explicit opt-in for old V0/V0.1 APIs; never monkeypatch running processes."""
import copy
from contextlib import contextmanager

@contextmanager
def independent_optimizer_loads(optimizer):
    had = 'load_state_dict' in optimizer.__dict__
    override = optimizer.__dict__.get('load_state_dict')
    native = optimizer.load_state_dict
    optimizer.load_state_dict = lambda state: native(copy.deepcopy(state))
    try:
        yield
    finally:
        if had: optimizer.__dict__['load_state_dict'] = override
        else: optimizer.__dict__.pop('load_state_dict', None)

def isolated_teacher_class(base):
    """Protect checkpoint input ownership and one-shot rollback in V0/V0.1.

    Their single-step rejection mechanism has no repeated-backtrack bug.
    This guard also prevents a later optimizer step mutating a caller checkpoint.
    """
    class Isolated(base):
        def load(self, state):
            return super().load(copy.deepcopy(state))
        def learn(self, *args, **kwargs):
            before = self.state()
            gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.net.parameters()]
            try:
                with independent_optimizer_loads(self.optimizer):
                    result = super().learn(*args, **kwargs)
            except BaseException:
                self.load(before)
                for p, g in zip(self.net.parameters(), gradients): p.grad = g
                raise
            if result.get('step_rejected'):
                for p, g in zip(self.net.parameters(), gradients): p.grad = g
            return result
    return Isolated
