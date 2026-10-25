import os
import random
import numpy as np
import torch
from recipe_v1.telemetry import RecipeTelemetry
from telemetry import plain,write_json,sha,model_sha


class APlusTelemetry(RecipeTelemetry):
    checkpoints={1400,1600,1800}

    def __init__(self,runner,out,monitor,parent):
        super().__init__(runner,out,monitor)
        self.update_count=1200;self.parent=str(parent)

    def save(self,filename):
        target=self.out/filename
        if target.exists():raise FileExistsError(target)
        saved=self.runner.alg.save()
        saved.update(iter=self.update_count-1,infos=dict(completed_updates=self.update_count,
            initialization='continue own V1 CP1200 actor/critic/distribution/Adam/adaptive LR; no reinitialization',
            parent_checkpoint=self.parent,additional_completed_updates=self.update_count-1200,
            environment_steps=self.update_count*self.env.num_envs*24,task_suite=plain(self.env.cfg.task_suite),
            learning_rate=self.runner.alg.learning_rate,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
            rng_python=random.getstate(),rng_numpy=np.random.get_state()))
        tmp=target.with_suffix('.partial');torch.save(saved,tmp);os.replace(tmp,target)
        write_json(target.with_suffix('.json'),dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,
            completed_updates=self.update_count,actor_sha256=model_sha(self.runner.alg.actor),critic_sha256=model_sha(self.runner.alg.critic),
            optimizer_state_entries=len(self.runner.alg.optimizer.state)))

    def reward(self,*a,**kw):
        result=super().reward(*a,**kw)
        with torch.no_grad():
            term=self.env.command_manager.get_term('base_velocity');row=self.steps[-1]
            for kind,name in [(0,'zero'),(1,'low'),(2,'medium')]:
                row['aplus/'+name+'_fraction']=(term.kind==kind).float().mean()
            for name,axis,sign in [('forward',0,1),('backward',0,-1),('left',2,1),('right',2,-1)]:
                for kind,level in [(1,'low'),(2,'medium')]:
                    cmd=term.command;actual=torch.cat((self.env.scene['robot'].data.root_lin_vel_b[:,:2],self.env.scene['robot'].data.root_ang_vel_b[:,2:3]),-1)
                    w=((term.kind==kind)&(term.desired[:,axis]*sign>.1)&((cmd-term.desired).abs().amax(-1)<1e-5)).float()
                    row['aplus/'+level+'/'+name+'/samples']=w.sum()
                    row['aplus/'+level+'/'+name+'/cross_sum']=(w*cmd[:,axis]*actual[:,axis]).sum()
                    row['aplus/'+level+'/'+name+'/command_sq_sum']=(w*cmd[:,axis].square()).sum()
        return result
