"""Independent read-only verification of saved result aggregates and provenance."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/'results'/name).read_text())

def main():
 provenance=read('provenance.json')
 for name,field in [('experiment_plan.json','plan_sha256'),('src/holdmarket.py','source_sha256')]:
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==provenance[field],name
 runs=read('simulation_runs.json');summary=read('simulation_summary.json');totals=read('run_summary.json')
 assert len(runs)==840==totals['runs']
 assert sum(r['arrivals'] for r in runs)==103200000==totals['counted_arrivals']
 assert len(set((r['kind'],r['M'],r['k'],r['release'],r['service'],r['seed']) for r in runs))==840
 for s in summary:
  r=[r for r in runs if all(r[k]==s[k] for k in ['kind','M','k','release','service'])]
  assert len(r)==24==s['n']
  for metric in ['success','q','accepted_per_request']:
   a=np.array([x[metric] for x in r]);mu=float(sum(a)/len(a))
   half=float(t.ppf(.975,len(a)-1)*np.sqrt(sum((a-mu)**2)/(len(a)-1)/len(a)))
   assert abs(s[metric]-mu)<1e-12
   assert abs(s[metric+'_low']-(mu-half))<1e-12
   assert abs(s[metric+'_high']-(mu+half))<1e-12
 exact=read('finite_exact.json')
 assert len(exact)==231 and max(x['q_error'] for x in exact)<1e-12 and max(x['residual'] for x in exact)<1e-12
 for r in exact:
  if r['k']==r['M']:assert abs(r['success']-r['q'])<1e-12
 central=read('central.json')
 assert central[0]['equilibria'][0]['mean_k']==12 and central[1]['equilibria'][0]['mean_k']==32
 for r in central:
  assert r['planner']['gap']<1.0001e-7 and r['planner']['toll_deviation_gap']<1e-9
 diag=read('fixed_horizon_runs.json')
 assert len(diag)==36 and sum(x['arrivals'] for x in diag)==18000000
 assert len(set((x['M'],x['warmup'],x['seed']) for x in diag))==36
 print('PASS: provenance, 840 main runs, 36 diagnostic runs, aggregation and intervals, 231 CTMC cases, central policies and toll checks.')

if __name__=='__main__':main()
