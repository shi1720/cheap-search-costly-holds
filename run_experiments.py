"""Reproduce all paper calculations; retained outputs are machine-readable."""
from pathlib import Path
import sys,json,hashlib,csv,dataclasses,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from holdmarket import Market,equilibria,planner,toll_at,finite_ctmc,simulate,success

def clean(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    raise TypeError(type(obj))

def save(name,x):
    (ROOT/'results'/name).write_text(json.dumps(x,indent=2,default=clean)+'\n')

def worker(job):
    kind,M,k,seed,release,arrivals,warmup,service=job
    tau=(lambda k:1.) if kind=='full' else (lambda k:.2+.05*(k-1))
    r=simulate(M,[k],[1.],1.,tau,seed,arrivals,warmup,release,service)
    r.update(kind=kind,k=k)
    r.pop('trace')
    return r

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['kind'],r['M'],r['k'],r['release'],r['service']),[]).append(r)
    summaries=[]
    for (kind,M,k,release,s),rs in groups.items():
        tau=1. if kind=='full' else .2+.05*(k-1)
        m=Market(base=tau,slope=0,K=k,service=s)
        p=np.eye(k+1)[k];q=m.availability(p)
        pred=1/(1+(M if release=='common_exp' else 1)) if kind=='full' else float(success(k,q))
        result=dict(kind=kind,M=M,k=k,release=release,service=s,n=len(rs),prediction=pred,q_prediction=q)
        for metric in ['success','q','accepted_per_request']:
            a=np.array([r[metric] for r in rs]);half=student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
            result[metric]=float(a.mean());result[metric+'_low']=float(a.mean()-half);result[metric+'_high']=float(a.mean()+half)
        summaries.append(result)
    return summaries

def main():
    start=time.time();plan=json.loads((ROOT/'experiment_plan.json').read_text())
    save('provenance.json',dict(plan_sha256=hashlib.sha256((ROOT/'experiment_plan.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((ROOT/'src'/'holdmarket.py').read_bytes()).hexdigest(),
        data_kind='synthetic event-driven computational experiments; no marketplace observations',
        python=sys.version,plan=plan))
    central=[]
    for c in plan['central_costs']:
        m=Market(cost=c,K=128)
        es=equilibria(m);opt=planner(m,tolerance=1e-7)
        opt['toll']=toll_at(m,opt['p'])
        u=m.utility(opt['q'],opt['toll']);opt['toll_deviation_gap']=float(max(u)-opt['p']@u)
        central.append(dict(cost=c,equilibria=es,planner=opt))
    save('central.json',central)
    curves=[]
    for slope in [0.,.01,.05,.2]:
        for c in np.geomspace(.0001,.1,70):
            m=Market(cost=float(c),slope=slope,K=512)
            for e in equilibria(m):
                curves.append(dict(slope=slope,cost=c,**{k:v for k,v in e.items() if k!='p'}))
    save('equilibrium_curves.json',curves)
    asymptotic=[]
    for beta in [.5,1.,2.]:
        for k in [16,64,256,1024,4096,16384,65536]:
            a=.05;lam=1.;q=1/(1+lam*a*k**(beta+1));P=float(success(k,q))
            c=q*(1-q)**(k-.5)
            kscale=(1/(lam*a*c))**(1/(beta+1))
            Pscale=(1/(lam*a))**(1/(beta+1))*c**(beta/(beta+1))
            asymptotic.append(dict(beta=beta,k=k,q=q,cost=c,success=P,k_ratio=k/kscale,success_ratio=P/Pscale))
    save('asymptotics.json',asymptotic)
    exact=[]
    for M in plan['small_exact']['M']:
        for k in range(1,M+1):
            for rate in plan['small_exact']['rates']:exact.append(finite_ctmc(M,k,rate))
    save('finite_exact.json',exact)
    print('Analytical calculations finished',flush=True)
    v=plan['validation'];jobs=[]
    for M in v['M']:
        for k in v['k']:
            for release in v['releases']:
                for seed in v['seeds']:jobs.append(('validation',M,k,seed,release,v['arrivals'],v['warmup'],0.))
    f=plan['full_fanout']
    for M in f['M']:
        for release in f['releases']:
            for seed in v['seeds']:jobs.append(('full',M,M,seed,release,f['arrivals'],f['warmup'],0.))
    f=plan['postcommit']
    for k in f['k']:
        for seed in v['seeds']:jobs.append(('postcommit',f['M'],k,seed,'common_exp',f['arrivals'],f['warmup'],f['service']))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,row in enumerate(ex.map(worker,jobs,chunksize=2)):
            rows.append(row)
            if (i+1)%48==0:print(f'{i+1}/{len(jobs)} runs complete',flush=True)
    save('simulation_runs.json',rows);save('simulation_summary.json',summarize(rows))
    save('run_summary.json',dict(runs=len(rows),counted_arrivals=sum(r['arrivals'] for r in rows),
         including_warmup=sum(j[5]+j[6] for j in jobs),elapsed_seconds=time.time()-start,
         exact_cases=len(exact),max_stationarity_residual=max(r['residual'] for r in exact),
         max_exact_balance_error=max(r['q_error'] for r in exact)))
    print('Completed',time.time()-start,flush=True)

if __name__=='__main__': main()
