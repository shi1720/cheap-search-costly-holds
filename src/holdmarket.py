"""Reservation markets: independent-availability economics and finite exact checks.

Time is in arbitrary consistent units; arrivals are per capacity unit per time.
No LLM outputs, observed shopper behavior, or field estimates are simulated here.
"""
from dataclasses import dataclass
from itertools import product
from math import comb
import heapq
import numpy as np
from scipy.optimize import brentq, linprog
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def success(k, q):
    return -np.expm1(np.asarray(k) * np.log1p(-np.minimum(q, 1-1e-16))) if q < 1 else np.asarray(k > 0, dtype=float)


@dataclass(frozen=True)
class Market:
    rate: float = 1.0
    base: float = 0.2
    slope: float = 0.05
    beta: float = 1.0
    value: float = 1.0
    cost: float = 0.001
    K: int = 64
    service: float = 0.0
    delay_cost: float = 0.0

    @property
    def ks(self):
        return np.arange(self.K+1)

    def tau(self, k):
        k = np.asarray(k)
        return np.where(k > 0, self.base+self.slope*np.maximum(k-1, 0)**self.beta, 0.)

    def h(self, k):
        return np.asarray(k)*self.tau(k)

    def availability(self, p):
        h = np.dot(p, self.h(self.ks))
        if not self.service:
            return 1/(1+self.rate*h)
        return brentq(lambda q: q*(1+self.rate*h)+self.rate*self.service*np.dot(p, success(self.ks,q))-1, 0., 1.)

    def utility(self, q, toll=0.):
        P = success(self.ks,q)
        return self.value*P-self.cost*self.ks-self.delay_cost*self.tau(self.ks)-toll*(q*self.h(self.ks)+self.service*P)

    def metrics(self, p):
        q = self.availability(p)
        P = float(np.dot(p, success(self.ks,q)))
        return dict(q=q, success=P, mean_k=float(np.dot(p,self.ks)),
                    hold_time=float(np.dot(p,self.h(self.ks))),
                    welfare=self.rate*float(np.dot(p,self.utility(q))))


def equilibria(m):
    """All symmetric equilibria, untolled, with zero delay cost.

    Strict concavity in integer fanout implies support on <=2 adjacent actions.
    Each adjacent indifference equation has at most two roots in q.
    """
    if m.delay_cost:
        raise ValueError('enumeration theorem assumes zero direct delay cost')
    out=[]
    def add(p):
        q=m.availability(p)
        u=m.utility(q)
        if np.max(u)-np.dot(p,u)>1e-8:
            raise AssertionError('not an equilibrium')
        if not any(np.max(np.abs(p-r['p']))<1e-7 for r in out):
            out.append(dict(p=p,**m.metrics(p)))
    for k in m.ks:
        p=np.eye(m.K+1)[k]
        u=m.utility(m.availability(p))
        if np.max(u)-u[k] <= 1e-10:
            add(p)
    c=m.cost/m.value
    for k in range(m.K):
        peak=1/(k+1)
        f=lambda q: q*(1-q)**k-c
        roots=[]
        if k==0:
            if 0<c<1: roots=[c]
        elif 0<c<=peak*(1-peak)**k:
            if abs(f(peak))<1e-14: roots=[peak]
            else: roots=[brentq(f,0,peak,xtol=1e-14),brentq(f,peak,1,xtol=1e-14)]
        for q in roots:
            # F(q,p)=0 is affine in mixing probability at fixed availability.
            a=q*(1+m.rate*m.h(k))+m.rate*m.service*success(k,q)-1
            b=m.rate*(q*(m.h(k+1)-m.h(k))+m.service*(success(k+1,q)-success(k,q)))
            if b<=0: continue
            x=-a/b
            if 1e-9<x<1-1e-9:
                p=np.zeros(m.K+1);p[k]=1-x;p[k+1]=x
                add(p)
    return sorted(out,key=lambda x:x['mean_k'])


def fixed_q_lp(m,q):
    """Best feasible randomized planner at specified availability (generic LP)."""
    P=success(m.ks,q)
    occ=m.rate*(q*m.h(m.ks)+m.service*P)
    reward=m.rate*(m.value*P-m.cost*m.ks-m.delay_cost*m.tau(m.ks))
    sol=linprog(-reward,A_eq=np.array([np.ones(m.K+1),occ]),b_eq=[1,1-q],bounds=(0,None),method='highs')
    if not sol.success:return None
    return dict(p=sol.x,q=q,welfare=-sol.fun,dual=sol.eqlin.marginals.tolist())


def planner(m,tolerance=1e-6,max_nodes=100000):
    """Global planner with an explicit interval upper bound on welfare.

    On [l,u], rewards are bounded by their value at u. Balance is relaxed to
    F(l,p)<=0<=F(u,p). No assertion of globality rests on a local optimizer.
    """
    def bound(l,u):
        Plo=success(m.ks,l);Phi=success(m.ks,u)
        lo=m.rate*(l*m.h(m.ks)+m.service*Plo)
        hi=m.rate*(u*m.h(m.ks)+m.service*Phi)
        reward=m.rate*(m.value*Phi-m.cost*m.ks-m.delay_cost*m.tau(m.ks))
        sol=linprog(-reward,A_ub=[lo,-hi],b_ub=[1-l,-(1-u)],
                    A_eq=[np.ones(m.K+1)],b_eq=[1],bounds=(0,None),method='highs')
        return -sol.fun if sol.success else -np.inf
    lower=m.availability(np.eye(m.K+1)[m.K]);upper=1.
    best=None
    for k in m.ks:
        p=np.eye(m.K+1)[k];met=m.metrics(p)
        if best is None or met['welfare']>best['welfare']:best=dict(p=p,**met)
    heap=[(-bound(lower,upper),0,lower,upper)];nodes=0;seq=1
    while heap and -heap[0][0]>best['welfare']+tolerance and nodes<max_nodes:
        negub,_,l,u=heapq.heappop(heap);mid=(l+u)/2;nodes+=1
        sol=fixed_q_lp(m,mid)
        if sol and sol['welfare']>best['welfare']:
            best=dict(p=sol['p'],**m.metrics(sol['p']))
        for a,b in ((l,mid),(mid,u)):
            ub=bound(a,b)
            if ub>best['welfare']+tolerance:
                heapq.heappush(heap,(-ub,seq,a,b));seq+=1
    ub=max(best['welfare'], -heap[0][0] if heap else best['welfare']+tolerance)
    best.update(upper_bound=ub,gap=ub-best['welfare'],nodes=nodes)
    return best


def toll_at(m,p):
    q=m.availability(p)
    dq=np.where(m.ks>0,m.ks*(1-q)**np.maximum(m.ks-1,0),0)
    return m.rate*m.value*np.dot(p,dq)/(1+m.rate*np.dot(p,m.h(m.ks)+m.service*dq))


def finite_ctmc(M,k,rate=1.,tau=1.):
    """Exact finite common-exponential-release system for fixed fanout.

    State n[j-1] = number of active mandates holding j slots. Only reachable
    states are built. A mandate samples k distinct slots, takes every idle one,
    and releases its entire accepted set after one Exp(mean=tau) clock.
    """
    states=[(0,)*k]; index={states[0]:0}; rows=[];pos=0
    denominator=comb(M,k)
    while pos<len(states):
        state=states[pos];busy=sum((j+1)*n for j,n in enumerate(state));free=M-busy
        transitions={}
        for j in range(1,min(k,free)+1):
            if k-j>busy:continue
            prob=comb(free,j)*comb(busy,k-j)/denominator
            nxt=list(state);nxt[j-1]+=1;nxt=tuple(nxt)
            transitions[nxt]=transitions.get(nxt,0)+rate*M*prob
        for j,n in enumerate(state):
            if n:
                nxt=list(state);nxt[j]-=1;nxt=tuple(nxt)
                transitions[nxt]=transitions.get(nxt,0)+n/tau
        for nxt in transitions:
            if nxt not in index:index[nxt]=len(states);states.append(nxt)
        rows.append(transitions);pos+=1
    Q=lil_matrix((len(states),len(states)))
    for i,tr in enumerate(rows):
        for nxt,r in tr.items():Q[i,index[nxt]]+=r
        Q[i,i]-=sum(tr.values())
    Q=Q.tocsr();A=Q.T.tolil();A[-1,:]=np.ones(len(states));b=np.zeros(len(states));b[-1]=1
    pi=spsolve(A.tocsr(),b)
    if np.min(pi)<-1e-8:raise AssertionError('negative stationary mass')
    free=np.array([M-sum((j+1)*n for j,n in enumerate(st)) for st in states])
    succ=np.array([1-(comb(M-int(f),k)/denominator if M-int(f)>=k else 0.) for f in free])
    q=float(pi@free/M);P=float(pi@succ)
    return dict(M=M,k=k,states=len(states),q=q,success=P,
                independent_success=float(success(k,q)),
                residual=float(np.max(np.abs(pi@Q))),
                mass_error=float(abs(sum(pi)-1)),
                q_error=float(abs(q-1/(1+rate*k*tau))))


def simulate(M,ks,probs,rate,tau_fn,seed,arrivals=100000,warmup=10000,
             release='common_exp',service=0.,record=False):
    """Event-driven physical-slot simulator, distinct from CTMC count states.

    Holding clocks are independent across mandates. `independent_exp` changes
    release clocks to independent across holds: an explicit negative control.
    """
    rng=np.random.default_rng(seed)
    busy=np.zeros(M,dtype=bool);events=[];t=0.;seq=0
    hits=holds=attempts=0;trace=[];free_sum=0.
    for n in range(arrivals+warmup):
        t+=rng.exponential(1/(rate*M))
        while events and events[0][0]<=t:
            _,_,slots=heapq.heappop(events)
            if not np.all(busy[slots]):raise AssertionError('double release')
            busy[slots]=False
        k=int(rng.choice(ks,p=probs))
        slots=rng.choice(M,size=k,replace=False) if k else np.empty(0,dtype=int)
        accepted=slots[~busy[slots]]
        if n>=warmup:
            hits+=bool(len(accepted));holds+=len(accepted);attempts+=k;free_sum+=1-busy.mean()
        if len(accepted):
            busy[accepted]=True
            tau=float(tau_fn(k))
            if release=='independent_exp':
                for slot in accepted:
                    heapq.heappush(events,(t+rng.exponential(tau),seq,np.array([slot])));seq+=1
            else:
                duration=rng.exponential(tau) if release=='common_exp' else tau
                if service:
                    winner=int(rng.choice(accepted));losers=accepted[accepted!=winner]
                    if len(losers):heapq.heappush(events,(t+duration,seq,losers));seq+=1
                    heapq.heappush(events,(t+duration+rng.exponential(service),seq,np.array([winner])));seq+=1
                else:
                    heapq.heappush(events,(t+duration,seq,accepted));seq+=1
        if record and n>=warmup and len(trace)<200:
            trace.append(dict(arrival=n-warmup,time=t,k=k,accepted=len(accepted),busy=int(busy.sum())))
    return dict(M=M,seed=seed,release=release,service=service,arrivals=arrivals,
                success=hits/arrivals,q=free_sum/arrivals,
                accepted_per_request=holds/arrivals,attempts_per_request=attempts/arrivals,trace=trace)
