from host import ROOT,run
import subprocess
run('prepare-seed70159',[str(ROOT/'code/prepare_new_seed.py')],gpu=False)
subprocess.run(['python3','-u',str(ROOT/'gen_fixed_controller.py')],check=True)
