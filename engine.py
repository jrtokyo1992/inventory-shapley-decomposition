"""Counterfactual inventory evaluation and Shapley allocation (single SKU/location)."""
import itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    weight: float = 0.6
    q_min: int = 0
    q_max: int = 120
    initial_inventory: float = 20
    initial_pipeline: tuple = () # (arrival_period, quantity)


def simulate(demand, order_periods, quantities, leads, cfg: Config):
    demand = np.asarray(demand, dtype=float)
    periods = np.asarray(order_periods, dtype=int)
    quantities = np.asarray(quantities, dtype=float)
    leads = np.asarray(leads, dtype=int)
    T = len(demand)
    arrivals = np.zeros(T)
    for t, q in cfg.initial_pipeline:
        if 0 <= int(t) < T:
            arrivals[int(t)] += float(q)
    for t, q, lead in zip(periods, quantities, leads):
        a = int(t + lead)
        if 0 <= a < T:
            arrivals[a] += q
    remaining = float(cfg.initial_inventory)
    records = []
    for t in range(T):
        opening = remaining
        available = opening + arrivals[t]
        sales = min(available, demand[t])
        lost = demand[t] - sales
        remaining = max(0.0, available - sales)
        records.append((t, opening, arrivals[t], demand[t], sales, lost, remaining, int(lost > 1e-9)))
    df = pd.DataFrame(records, columns=['period','opening_inventory','receipts','demand','fulfilled_demand','lost_demand','ending_inventory','stockout'])
    avg_d = max(float(np.mean(demand)), 1e-8)
    oos = float(df.stockout.mean())
    inv = float(df.ending_inventory.mean())
    holding = inv / avg_d
    return {'loss':cfg.weight*oos+(1-cfg.weight)*holding,'stockout_rate':oos,'avg_inventory':inv,'normalized_inventory':holding,'trajectory':df}


def generate_references(demand, periods, obs_q, planned_leads, cfg, n_candidates=300, keep=8, seed=42):
    """Best-found feasible policies, NOT a global optimum. Optimization assumes planned leads."""
    rng = np.random.default_rng(seed)
    obs_q = np.asarray(obs_q, dtype=int)
    low, high = cfg.q_min, cfg.q_max
    candidates = {tuple(np.clip(obs_q, low, high))}
    expected = np.asarray([np.sum(demand[t:min(len(demand), t + max(1, int(np.median(np.diff(periods))) if len(periods)>1 else 4))]) for t in periods])
    for base in (expected, expected*0.8, expected*1.2, np.full(len(periods),np.mean(demand)*max(1,np.median(np.diff(periods)) if len(periods)>1 else 4))):
        candidates.add(tuple(np.clip(np.rint(base),low,high).astype(int)))
    for _ in range(n_candidates):
        mode = rng.integers(0,3)
        if mode == 0: q=obs_q + rng.normal(0,max(3,np.mean(demand)*2),len(obs_q))
        elif mode == 1: q=expected * rng.uniform(.6,1.4) + rng.normal(0,max(2,np.mean(demand)*.5),len(obs_q))
        else: q=rng.uniform(low,high,len(obs_q))
        candidates.add(tuple(np.clip(np.rint(q),low,high).astype(int)))
    scored=[]
    for candidate in candidates:
        y=simulate(demand,periods,candidate,planned_leads,cfg)['loss']
        scored.append((y,candidate))
    scored.sort(key=lambda z:z[0])
    return scored[:keep]


def select_references(scored, obs_q, k, weighting='distance', distance_power=2.):
    obs=np.asarray(obs_q,float)
    rows=[]
    for y,q in scored:
        d=np.linalg.norm(np.abs((np.array(q)-obs)/np.maximum(1,abs(obs))),ord=distance_power)
        rows.append({'policy':np.array(q,dtype=int),'loss':y,'distance':d})
    selected=sorted(rows,key=lambda r:r['distance'])[:k]
    if weighting=='distance':
        inv=1/(np.array([r['distance'] for r in selected])+1e-6)
        ws=inv/inv.sum()
    else: ws=np.ones(len(selected))/len(selected)
    for r,w in zip(selected,ws):r['weight']=float(w)
    return selected


def allocate(demand, periods, obs_q, actual_leads, planned_leads, references, cfg, M=200, seed=42, include_delays=True, exact=False):
    """Coalition game: replace order decisions and delay realizations separately.
    All agents share same demand path. For each reference, Shapley telescopes to
    loss(reference orders, planned lead) - loss(observed orders, actual lead).
    """
    rng=np.random.default_rng(seed)
    n=len(obs_q)
    active_orders=[i for i in range(n) if any(int(r['policy'][i])!=int(obs_q[i]) for r in references)]
    active_delays=[i for i in range(n) if include_delays and int(actual_leads[i])!=int(planned_leads[i])]
    players=[('order',i) for i in active_orders]+[('delay',i) for i in active_delays]
    labels=[f'Order {i+1} (t={periods[i]})' if kind=='order' else f'Delay {i+1} (t={periods[i]})' for kind,i in players]
    if not players:
        return pd.DataFrame(columns=['factor','type','order_index','contribution','se']), {'gap':0.,'residual':0.,'cache_size':0,'n_players':0,'by_reference':[]}
    n_players=len(players)
    if exact and n_players>10:raise ValueError('Exact enumeration supports at most 10 changing factors.')
    weighted_phi=np.zeros(n_players)
    weighted_var=np.zeros(n_players)
    total_cache=0; by_ref=[]
    for ref_id, ref in enumerate(references):
        ref_q=np.asarray(ref['policy'],int)
        cache={}
        def value(mask):
            if mask not in cache:
                q=np.array(obs_q,dtype=int,copy=True)
                leads=np.array(actual_leads,dtype=int,copy=True)
                for j,(kind,i) in enumerate(players):
                    if mask & (1<<j):
                        if kind=='order':q[i]=ref_q[i]
                        else:leads[i]=planned_leads[i]
                cache[mask]=simulate(demand,periods,q,leads,cfg)['loss']
            return cache[mask]
        if exact:
            from math import factorial
            phi=np.zeros(n_players)
            for mask in range(1<<n_players):
                size=mask.bit_count()
                for j in range(n_players):
                    if not mask&(1<<j):
                        wt=factorial(size)*factorial(n_players-size-1)/factorial(n_players)
                        phi[j]+=wt*(value(mask|(1<<j))-value(mask))
            se=np.zeros(n_players)
        else:
            draws=[]
            for _ in range(M):
                perm=rng.permutation(n_players)
                pair=[]
                for path in (perm,perm[::-1]):
                    cur=0; start=value(0); delta=np.zeros(n_players)
                    for j in path:
                        nxt=cur|(1<<int(j)); end=value(nxt)
                        delta[j]=end-start
                        cur=nxt;start=end
                    pair.append(delta)
                draws.append(np.mean(pair,axis=0))
            draws=np.array(draws)
            phi=draws.mean(axis=0)
            se=draws.std(axis=0,ddof=1)/np.sqrt(M) if M>1 else np.zeros(n_players)
        weight=ref['weight']
        weighted_phi+=weight*phi
        weighted_var+=(weight*se)**2
        full_mask=(1<<n_players)-1
        by_ref.append({'reference':ref_id+1,'weight':weight,'observed_loss':value(0),'reference_loss':value(full_mask),'gap':value(0)-value(full_mask)})
        total_cache+=len(cache)
    output=pd.DataFrame([{'factor':label,'type':kind.title(),'order_index':i+1,'contribution':-weighted_phi[j], 'se':np.sqrt(weighted_var[j]),'observed_order':int(obs_q[i]),'reference_order_weighted':float(sum(r['weight']*r['policy'][i] for r in references)),'actual_lead':int(actual_leads[i]),'planned_lead':int(planned_leads[i])} for j,(label,(kind,i)) in enumerate(zip(labels,players))])
    gap=sum(r['weight']*r['gap'] for r in by_ref)
    return output, {'gap':gap,'residual':float(output.contribution.sum()-gap),'cache_size':total_cache,'n_players':n_players,'by_reference':by_ref}
