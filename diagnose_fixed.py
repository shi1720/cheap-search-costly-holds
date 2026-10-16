"""Exploratory horizon check prompted by the frozen main run's fixed-clock gap.

This script is explicitly post hoc; it does not alter the main experiment plan.
"""
from run_experiments import worker,save
from concurrent.futures import ProcessPoolExecutor
import time

if __name__=='__main__':
    jobs=[]
    for M in [512,2048]:
        for warmup in [20000,200000,2000000]:
            for seed in range(202801,202807):
                jobs.append(('diagnostic',M,32,seed,'common_fixed',500000,warmup,0.))
    start=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,r in enumerate(ex.map(worker,jobs)):
            r['warmup']=jobs[i][6];rows.append(r)
            save('fixed_horizon_runs.json',rows)
            if (i+1)%6==0:print(i+1,len(jobs),time.time()-start,flush=True)
    save('fixed_horizon_summary.json',dict(runs=len(rows),counted_arrivals=sum(r['arrivals'] for r in rows),
        including_warmup=sum(j[5]+j[6] for j in jobs),elapsed_seconds=time.time()-start,
        purpose='Exploratory post-hoc diagnosis of fixed-clock startup and horizon sensitivity'))
