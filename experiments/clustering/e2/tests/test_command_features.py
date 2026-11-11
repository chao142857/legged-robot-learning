import pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'code'))
from command_features import features,timeline,balanced_partition,euler_matrix,quat_matrix

class FeatureTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17);self.ee=rng.random((64,4,7),dtype=np.float32);self.vel=rng.random((64,4,5),dtype=np.float32);self.ref=dict(height=.42,quaternion=[1,0,0,0])
    def test_excludes_unused_slots_and_timers(self):
        a,_=features(self.ee,self.vel,self.ref);ee=self.ee.copy();vel=self.vel.copy()
        ee[:,3]=.99;ee[:,2,6]=.01;vel[:,3]=.99;vel[:,:,4]=.99
        b,_=features(ee,vel,self.ref);np.testing.assert_array_equal(a,b)
    def test_standing_marker_only_and_zero_velocity(self):
        vel=self.vel.copy();vel[:,:2,3]=.01;a,meta=features(self.ee,vel,self.ref)
        vel[:,:2,3]=.09;b,_=features(self.ee,vel,self.ref);np.testing.assert_array_equal(a,b)
        vel[:,:2,:3]=.99;b,_=features(self.ee,vel,self.ref)
        start=sum(v for k,v in list(meta['block_dimensions'].items())[:6]);np.testing.assert_array_equal(a[:,:start],b[:,:start])
        np.testing.assert_array_equal(a[:,start+3:],b[:,start+3:]);self.assertGreater(float(np.linalg.norm(a[:,start:start+3]-b[:,start:start+3])),0)
    def test_quaternion_antipodes_same_rotation(self):
        q=np.array([.5,.5,.5,.5]);np.testing.assert_array_equal(quat_matrix(q),quat_matrix(-q))
        R=euler_matrix(np.array([0.,0.,np.pi]));self.assertAlmostEqual(float(np.linalg.norm(R-np.eye(3))**2),8)
    def test_autoreset_final_velocity_slot2_is_effective(self):
        trace,starts=timeline(self.ee,self.vel,True)
        self.assertTrue(np.all(trace[:,-1,1]==2));self.assertTrue(np.all(starts[:,1,2]==999))
        a,_=features(self.ee,self.vel,self.ref);vel=self.vel.copy();vel[:,2,:3]=1-vel[:,2,:3];b,_=features(self.ee,vel,self.ref)
        self.assertGreater(float(np.linalg.norm(a-b)),0)
    def test_switch_order_not_a_bag_of_commands(self):
        a,_=features(self.ee,self.vel,self.ref);ee=self.ee.copy();ee[:,[0,1]]=ee[:,[1,0]];b,_=features(ee,self.vel,self.ref)
        self.assertGreater(float(np.linalg.norm(a-b)),1)
    def test_native_timer_float32_and_capacity(self):
        trace,starts=timeline(self.ee,self.vel,False)
        self.assertTrue(np.all(starts[:,1,1]==500));self.assertTrue(np.all(starts[:,1,2]==1000))
        self.assertLessEqual(int(trace[:,:,0].max()),2);self.assertLessEqual(int(trace[:,:,1].max()),1)
        X,_=features(self.ee,self.vel,self.ref);a=balanced_partition(X,8,42);b=balanced_partition(X,8,42)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(np.bincount(a),[8]*8)

if __name__=='__main__':unittest.main(verbosity=2)
