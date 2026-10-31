"""Standalone scientific SVG from the compact measured physics table."""
import argparse,hashlib,json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(data):
    rows=data['physics'];fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    colors={'control':'#4263eb','coverage':'#e67700','entropyLow':'#2b8a3e'}
    initial={(r['protocol'],r['seed']):r for r in rows if r['case'].startswith('frozen1800-')}
    series={}
    for r in rows:
        if r['case'].startswith(('frozen1800-','final-')):continue
        match=re.fullmatch(r'(control|coverage|entropyLow)(\d+)',r['case'].split('-')[0])
        if match:series.setdefault((match[1],r['seed'],r['protocol']),[]).append((int(match[2]),r))
    panels=[('X','X signed gain, native after1s',lambda r:r['X_after1s']['synchronous_gain'],.35),
        ('X','X correlation, native after1s',lambda r:r['X_after1s']['correlation'],.5),
        ('X','Actual X amplitude after1s (mm)',lambda r:1000*r['X_after1s']['actual_fitted_amplitude_m'],8.),
        ('A','A strict safety, entire20s',lambda r:r['strict_safe']/r['n'],.95),
        ('X','X EE position RMSE, entire20s (cm)',lambda r:100*r['EE_full20s_m'],5.),
        ('X','X base XY RMSE, entire20s (m/s)',lambda r:r['XY_full20s_mps'],.15)]
    for ax,(protocol,title,value,gate) in zip(axes.flat,panels):
        for (branch,seed,p),entries in series.items():
            if p!=protocol:continue
            points=sorted(entries)
            if (p,seed) in initial:points=[(0,initial[(p,seed)])]+points
            ax.plot([n for n,r in points],[value(r) for n,r in points],marker='o',ms=3,
                color=colors[branch],linestyle='-' if seed==8101 else '--',label=branch+' / '+str(seed))
        ax.axhline(gate,color='#c92a2a',ls=':',lw=1)
        if 'amplitude' in title:ax.axhline(20,color='#555555',ls='--',lw=.8,label='target20mm')
        ax.set_title(title,fontsize=10);ax.set_xlabel('New PPO updates from shared CP1800');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Original PPO + Uniform | warm-start train seed21 | separate evaluation-seed lines\nMeasured checkpoints; original task and strict-safety gates unchanged',fontsize=11)
    return fig

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output',required=True);c=a.parse_args();path=pathlib.Path(c.input);raw=path.read_bytes();fig=plot(json.loads(raw));fig.savefig(c.output,format='svg',metadata={'Description':'Input SHA256 '+hashlib.sha256(raw).hexdigest()});plt.close(fig)
