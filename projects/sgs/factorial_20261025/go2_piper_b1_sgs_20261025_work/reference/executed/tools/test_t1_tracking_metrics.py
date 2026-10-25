import unittest
import numpy as np
from t1_tracking_metrics import tracking_metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.p=dict(ee_center=[.45,0,.5],base_segment_s=6.,base_linear_speed=.2,base_yaw_rate=.15)
        t=np.arange(1000)*.02;phase=t*np.pi/4
        xyz=np.array(self.p['ee_center'])+np.stack((.02*np.sin(phase),.03*np.cos(phase),-.02*np.sin(phase)),-1)
        commands=np.zeros((1000,4,3))
        for i in range(4):
            group=(i+(t/6).astype(int))%4
            commands[:,i,0]=np.where(group==0,.2,np.where(group==1,-.2,0))
            commands[:,i,2]=np.where(group==2,.15,np.where(group==3,-.15,0))
        self.d={'valid':np.ones((1000,4),bool),'ee_cmd':np.tile(xyz[:,None],(1,4,1)),
            'ee_actual_mixed':np.tile(xyz[:,None],(1,4,1)),'base_cmd':commands,'base_actual':commands.copy(),
            'age_steps':np.tile(np.arange(1,1001)[:,None],(1,4))}

    def test_perfect_track(self):
        x=tracking_metrics(self.d,self.p)['windows']['all']
        self.assertAlmostEqual(x['ee']['mse_improvement_vs_center'],1)
        for v in x['directions'].values():self.assertAlmostEqual(v['command_plateau']['gain'],1)

    def test_center_and_zero_are_not_tracking(self):
        self.d['ee_actual_mixed'][:]=self.p['ee_center'];self.d['base_actual'][:]=0
        x=tracking_metrics(self.d,self.p)['windows']['all']
        self.assertAlmostEqual(x['ee']['mse_improvement_vs_center'],0)
        self.assertLess(x['ee']['position_rmse_m'],.05)
        for v in x['directions'].values():self.assertEqual(v['command_plateau']['gain'],0)
        self.assertLess(x['directions']['left']['command_plateau']['rmse'],.2)

    def test_invalid_postreset_values_excluded(self):
        self.d['valid'][500:]=False
        a=tracking_metrics(self.d,self.p)
        self.d['ee_actual_mixed'][500:]=123
        self.d['base_actual'][500:]=456
        self.assertEqual(a,tracking_metrics(self.d,self.p))

    def test_empty_direction_is_not_success(self):
        self.d['valid'][:,1:]=False;self.d['valid'][200:]=False
        x=tracking_metrics(self.d,self.p)['windows']['all']
        self.assertEqual(x['directions']['right']['command_plateau'],{'samples':0})


if __name__=='__main__':unittest.main()
