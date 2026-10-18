"""High-precision deterministic check of the beta=1 pure-equilibrium welfare limit.

This is an algebra check, not additional market or simulation evidence.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent

def main():
    rows=[]
    with localcontext() as ctx:
        ctx.prec=80
        for k in [32,128,512,2048,8192,32768,65536]:
            q=Decimal(1)/(Decimal(1)+Decimal(k)**2/Decimal(20))
            r=Decimal(1)-q
            cost=q*r**k/r.sqrt()
            probability=1-r**k
            welfare=probability-Decimal(k)*cost
            rows.append(dict(k=k,cost=str(cost),success=str(probability),
                             welfare=str(welfare),welfare_ratio=str(2*welfare/probability**2)))
    (ROOT/'results/asymptotic_welfare_check.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert all(Decimal(r['welfare'])>0 for r in rows)
    assert abs(Decimal(rows[-1]['welfare_ratio'])-1)<Decimal('.001')
    print('PASS: high-precision beta=1 welfare asymptotics, seven constructive pure-equilibrium costs at 80-digit precision.')

if __name__=='__main__':main()
