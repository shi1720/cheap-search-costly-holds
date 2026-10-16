from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from holdmarket import Market,success
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,
 'axes.titlesize':11,'legend.fontsize':8.5,'figure.dpi':160,'savefig.bbox':'tight',
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.alpha':.18,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#195D92';ORANGE='#C16B16';GREEN='#287A61';GRAY='#727A83';PURPLE='#765292'
def get(name):return json.loads((ROOT/'results'/name).read_text())
def save(fig,name):
 fig.savefig(ROOT/'figures'/f'{name}.pdf')
 fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180)
 plt.close(fig)

def main():
 fig,ax=plt.subplots(figsize=(7,2.45));ax.set(xlim=(0,10),ylim=(0,3.2));ax.axis('off')
 boxes=[(.1,1.1,1.8,1.,'Discover\nnonbinding quotes'),(2.5,1.1,1.8,1.,'Request $k$\nexclusive holds'),(5,1.1,1.8,1.,'Decide while\ncapacity is held'),(7.6,1.1,2.2,1.,'One transaction\nrelease alternatives')]
 for i,(x,y,w,h,label) in enumerate(boxes):
  p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#F2F6F9' if i!=2 else '#FCF0E1',edgecolor=BLUE if i!=2 else ORANGE,linewidth=1.3);ax.add_patch(p)
  ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
  if i<len(boxes)-1:ax.add_patch(FancyArrowPatch((x+w+.1,y+.5),(boxes[i+1][0]-.12,y+.5),arrowstyle='-|>',mutation_scale=13,color=GRAY))
 ax.text(5.9,.65,'Occupancy per mandate: $q\\,k\\tau_k$',ha='center',color=ORANGE)
 ax.text(5.9,.25,'Common decision clock can synchronize suppliers',ha='center',color=GRAY,fontsize=9)
 ax.text(5,2.7,'Search cost and capacity-time are different economic quantities',ha='center',fontweight='bold',fontsize=11)
 save(fig,'mechanism')

 sim=get('simulation_summary.json');fig,axes=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
 M=np.geomspace(2,1024,250);q=1/(1+M)
 axes[0].loglog(M,q,color=ORANGE,label='Common release: exact')
 axes[0].loglog(M,M*0+.5,color=BLUE,label='Independent release: exact')
 axes[0].loglog(M,1-(1-q)**M,color=GRAY,ls='--',label='Independent availability')
 for release,color in [('common_exp',ORANGE),('independent_exp',BLUE)]:
  rs=[r for r in sim if r['kind']=='full' and r['release']==release]
  for ax,metric in zip(axes,['success','q']):
   ax.errorbar([r['M'] for r in rs],[r[metric] for r in rs],
    yerr=[[r[metric]-r[metric+'_low'] for r in rs],[r[metric+'_high']-r[metric] for r in rs]],fmt='o',color=color,ms=4,capsize=2)
 axes[1].loglog(M,q,color=GRAY,label='Both release laws: exact')
 for ax in axes:ax.set_xlabel('Number of suppliers $M$')
 axes[0].set_ylabel('Probability of any successful hold');axes[1].set_ylabel('One-supplier idle probability $q$')
 axes[0].set_title('(a) Distinct successful mandates');axes[1].set_title('(b) Matched marginal capacity')
 axes[0].legend(loc='lower left');axes[1].legend(loc='upper right')
 save(fig,'full_fanout')

 curves=get('equilibrium_curves.json');central=get('central.json')
 fig,axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
 for slope,color in [(0,GRAY),(.01,GREEN),(.05,BLUE),(.2,ORANGE)]:
  rs=[r for r in curves if r['slope']==slope and r['cost']>=.001 and r['mean_k']<511.9]
  rs=sorted(rs,key=lambda r:r['cost'],reverse=True)
  for ax,metric in zip(axes,['mean_k','success']):
   ax.plot([r['cost'] for r in rs],[r[metric] for r in rs],color=color,lw=1.7,label=f'$a={slope:g}$')
 for r in central:
  for e in r['equilibria']:
   for ax,metric in zip(axes,['mean_k','success']):ax.plot(r['cost'],e[metric],'o',color=BLUE,ms=4)
 for ax in axes:
  ax.set_xscale('log');ax.invert_xaxis();ax.set_xlabel('Per-attempt cost $c$ (cheaper $\\rightarrow$)')
 axes[0].set_ylabel('Equilibrium mean breadth');axes[1].set_ylabel('Equilibrium completion probability')
 axes[0].set_yscale('log');axes[0].set_title('(a) Private expansion');axes[1].set_title('(b) Consequences for trade')
 axes[0].legend();axes[1].set_ylim(0,1.03)
 save(fig,'economics')

 fig,axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
 ks=np.arange(1,33);pred=[float(success(k,1/(1+k*(.2+.05*(k-1))))) for k in ks]
 for ax,M in zip(axes,[128,2048]):
  ax.plot(ks,pred,color=GRAY,ls='--',label='Independent availability')
  for release,color,marker in [('common_exp',BLUE,'o'),('common_fixed',ORANGE,'s')]:
   rs=[r for r in sim if r['kind']=='validation' and r['M']==M and r['release']==release]
   ax.errorbar([r['k'] for r in rs],[r['success'] for r in rs],
    yerr=[[r['success']-r['success_low'] for r in rs],[r['success_high']-r['success'] for r in rs]],
    fmt=marker+'-',lw=1.2,ms=4,color=color,capsize=2,label=release.replace('common_','Common '))
  ax.set(title=f'$M={M}$: main finite horizon',xlabel='Reservation breadth $k$',ylim=(.15,.97))
 axes[0].set_ylabel('Successful mandates / arrivals');axes[1].legend(loc='lower left')
 save(fig,'finite_validation')

 diagnostics=get('fixed_horizon_runs.json')
 fig,ax=plt.subplots(figsize=(5.7,2.8),layout='constrained')
 from scipy.stats import t as student_t
 for M,color in [(512,GREEN),(2048,ORANGE)]:
  xs=[];ys=[];errs=[]
  for warmup in [20000,200000,2000000]:
   values=np.array([r['success'] for r in diagnostics if r['M']==M and r['warmup']==warmup])
   xs.append(warmup);ys.append(values.mean());errs.append(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
  ax.errorbar(xs,ys,yerr=errs,fmt='o-',color=color,capsize=3,label=f'$M={M}$')
 ax.axhline(float(success(32,1/57)),color=GRAY,ls='--',label='Independent-availability benchmark')
 ax.set(xscale='log',xlabel='Warm-up arrivals before 500,000 counted arrivals',ylabel='Successful mandates / arrivals',ylim=(.29,.45))
 ax.legend(loc='lower right')
 save(fig,'horizon_diagnostic')
 rows=[]
 for M in [512,2048]:
  for warmup in [20000,200000,2000000]:
   values=np.array([r['success'] for r in diagnostics if r['M']==M and r['warmup']==warmup])
   half=student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
   rows.append(f"{M} & {warmup:,} & {values.mean():.4f} & [{values.mean()-half:.4f}, {values.mean()+half:.4f}] " + chr(92)*2)
 (ROOT/'paper'/'horizon_rows.tex').write_text('\n'.join(rows)+'\n')

 rows=[]
 for r in central:
  e=r['equilibria'][0];p=r['planner']
  rows.append(f"{r['cost']:.3f} & {e['mean_k']:.2f} & {e['success']:.4f} & {e['welfare']:.4f} & {p['mean_k']:.0f} & {p['success']:.4f} & {p['welfare']:.4f}" + chr(92)*2)
 (ROOT/'paper'/'central_rows.tex').write_text('\n'.join(rows)+'\n')
 rows=[]
 for s in [0.,.5]:
  for k in [3,12,32]:
   r=next(r for r in sim if r['M']==2048 and r['k']==k and r['release']=='common_exp' and r['service']==s)
   rows.append(f"{s:.1f} & {k} & {r['prediction']:.4f} & {r['success']:.4f} & [{r['success_low']:.4f}, {r['success_high']:.4f}]" + chr(92)*2)
 (ROOT/'paper'/'finite_rows.tex').write_text('\n'.join(rows)+'\n')
 summary=get('run_summary.json')
 text=(f"The main run contains {summary['runs']:,} runs and {summary['counted_arrivals']/1e6:.1f} million counted arrivals "
       f"({summary['including_warmup']/1e6:.0f} million including warm-up). Across the 231 finite CTMC cases, "
       f"the largest stationarity residual is ${summary['max_stationarity_residual']:.2g}$ and the largest marginal-balance error is "
       f"${summary['max_exact_balance_error']:.2g}$. For common exponential release at $M=2048$, "
       "the largest absolute success discrepancy across the four tested breadths is 0.0016. "
       "The deterministic-release control is substantially different at breadth 32: the main-horizon success mean "
       "is 0.2494, compared with benchmark 0.4324, despite nearly unchanged mean availability. "
       "This motivates the separate horizon diagnostic below; these fixed-clock runs should not be interpreted as demonstrated stationary samples.\n")
 # Use proper scientific notation in LaTeX.
 text=text.replace('1.4e-15',r'1.4\times10^{-15}').replace('4.4e-16',r'4.4\times10^{-16}')
 (ROOT/'paper'/'results_paragraph.tex').write_text(text)

if __name__=='__main__':main()
