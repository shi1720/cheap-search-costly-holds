import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from holdmarket import *
from fractions import Fraction as R
from itertools import combinations


def labeled_stationary(M,k):
    """Independent exact rational oracle using groups of labeled bitmasks."""
    choices=[sum(1<<i for i in c) for c in combinations(range(M),k)]
    states=[()];idx={():0};rows=[]
    for state in states:
        occupied=0
        for mask in state: occupied |=mask
        tr={}
        for mask in choices:
            new=mask & ~occupied
            if new:
                nxt=tuple(sorted((*state,new)))
                tr[nxt]=tr.get(nxt,R(0))+R(M,len(choices))
        for j in range(len(state)):
            nxt=state[:j]+state[j+1:]
            tr[nxt]=tr.get(nxt,R(0))+1
        for nxt in tr:
            if nxt not in idx:idx[nxt]=len(states);states.append(nxt)
        rows.append(tr)
    n=len(states);Q=[[R(0) for _ in range(n)] for _ in range(n)]
    for i,tr in enumerate(rows):
        for nxt,rate in tr.items(): Q[i][idx[nxt]]+=rate;Q[i][i]-=rate
    A=[[Q[j][i] for j in range(n)]+[R(0)] for i in range(n)]
    A[-1]=[R(1)]*n+[R(1)]
    for col in range(n):
        pivot=next(r for r in range(col,n) if A[r][col])
        A[col],A[pivot]=A[pivot],A[col]
        den=A[col][col];A[col]=[x/den for x in A[col]]
        for r in range(n):
            if r!=col and A[r][col]:
                mult=A[r][col];A[r]=[a-mult*b for a,b in zip(A[r],A[col])]
    pi=[row[-1] for row in A]
    assert sum(pi)==1 and min(pi)>=0
    P=R(0);q=R(0)
    for state,mass in zip(states,pi):
        occupied=0
        for mask in state:occupied|=mask
        P+=mass*R(sum(bool(mask & ~occupied) for mask in choices),len(choices))
        q+=mass*R(M-occupied.bit_count(),M)
    return q,P,len(states)


def test_exact_rational_labeled_oracle():
    for M in range(1,6):
        for k in range(1,M+1):
            q,P,n=labeled_stationary(M,k)
            out=finite_ctmc(M,k)
            assert q==R(1,1+k)
            assert abs(out['success']-float(P))<1e-11
            assert out['residual']<1e-10


def test_full_fanout_and_singleton():
    for M in range(1,15):
        for rate in [.2,1,3]:
            assert abs(finite_ctmc(M,M,rate)['success']-1/(1+rate*M))<1e-12
            assert abs(finite_ctmc(M,1,rate)['success']-1/(1+rate))<1e-12


def test_equilibrium_completeness_and_deviations():
    cases=0
    for rate in [.2,1,3]:
        for base in [.05,.2,1.]:
            for slope in [0,.03,.2]:
                for cost in [.001,.01,.05,.2]:
                    for service in [0,.5]:
                        m=Market(rate=rate,base=base,slope=slope,cost=cost,K=32,service=service)
                        es=equilibria(m)
                        assert es
                        for e in es:
                            p=e['p'];u=m.utility(e['q'])
                            assert max(u)-p@u<1e-8
                            support=np.flatnonzero(p>1e-8)
                            assert len(support)<=2
                            assert max(support)-min(support)<=1
                            assert abs(sum(p)-1)<1e-10
                        cases+=1
    assert cases==216


def test_global_planner_bound_and_toll():
    rng=np.random.default_rng(711)
    for service in [0,.5]:
        for c in [.001,.03]:
            m=Market(K=16,cost=c,service=service)
            sol=planner(m,tolerance=2e-6)
            assert sol['gap']<2.01e-6
            for _ in range(100):
                p=rng.dirichlet(np.ones(m.K+1))
                assert m.metrics(p)['welfare']<=sol['upper_bound']+1e-9
            u=m.utility(sol['q'],toll_at(m,sol['p']))
            # A certified objective gap is not an exact KKT certificate.
            assert max(u)-sol['p']@u<1e-3


def test_balance_and_simulator_boundary():
    for release in ['common_exp','common_fixed','independent_exp']:
        r=simulate(1,[1],[1],1,lambda k:1.,18,arrivals=100000,warmup=5000,release=release)
        assert abs(r['success']-.5)<.02
        assert abs(r['q']-.5)<.02
    # No attempts create no holds; the outside option is fully supported.
    r=simulate(4,[0],[1],1,lambda k:0.,1,arrivals=100)
    assert r['success']==0 and r['q']==1


def test_multiple_equilibrium_roots_are_retained():
    m=Market(rate=.001,base=.001,slope=1.,cost=.001,K=8)
    es=equilibria(m)
    assert len(es)==3
    assert es[0]['mean_k']==1 and es[-1]['mean_k']==2
    assert 1<es[1]['mean_k']<2
