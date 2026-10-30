import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from engine import Config, simulate, generate_references, select_references, allocate

st.set_page_config(page_title='Inventory Decision Attribution',page_icon='📦',layout='wide')
st.title('📦 Inventory Decision Attribution')
st.caption('Counterfactual Shapley decomposition of ordering decisions and delivery delays · one SKU × one location')

@st.cache_data
def demo():
    rng=np.random.default_rng(2026)
    days=np.arange(28)
    demand=np.maximum(0,rng.poisson(15+3*np.sin(days*2*np.pi/7))).astype(int)
    orders=pd.DataFrame({'order_id':[f'PO-{i+1:02d}' for i in range(7)],'period':np.arange(0,28,4),'quantity':[55,65,45,80,55,70,50],'planned_lead':[2]*7,'actual_lead':[2,3,2,4,2,3,2]})
    return pd.DataFrame({'period':days,'demand':demand}),orders

st.sidebar.header('Objective & assumptions')
weight=st.sidebar.slider('Stockout weight',0.,1.,.6,.05)
st.sidebar.caption(f'Overstock / inventory weight: {1-weight:.0%}')
initial=st.sidebar.number_input('Initial on-hand inventory',min_value=0.,value=30.,step=1.)
min_q=st.sidebar.number_input('Min order quantity',min_value=0,value=0)
max_q=st.sidebar.number_input('Max order quantity',min_value=int(min_q),value=max(120,int(min_q)))
st.sidebar.header('Search & Shapley')
n_candidates=st.sidebar.slider('Reference search candidates',50,2000,400,50)
k=st.sidebar.slider('Number of reference policies',1,8,3)
weighting=st.sidebar.selectbox('Reference weighting',['distance','simple'])
M=st.sidebar.slider('Permutation pairs per reference',25,1000,200,25)
include_delays=st.sidebar.checkbox('Attribute delivery deviations',True)
exact=st.sidebar.checkbox('Use exact Shapley (≤ 10 changing factors)',False)
st.sidebar.caption('Random search returns best-found feasible benchmarks, not a proven global optimum. Results use the realized demand path (perfect hindsight).')

with st.expander('1 · Upload and review input data',expanded=True):
    st.write('Upload **CSV** files. Demand: `period, demand`. Orders: `order_id, period, quantity, planned_lead, actual_lead`. Periods are integer indices starting at 0. Planned/actual leads are integer numbers of periods.')
    a,b=st.columns(2)
    with a:dem_file=st.file_uploader('Demand / sales CSV',type='csv',key='demand')
    with b:ord_file=st.file_uploader('Orders & delivery CSV',type='csv',key='orders')
    use_demo=st.checkbox('Use example data when files are not uploaded',value=True)
    if use_demo:demand_df, orders_df=demo()
    else:demand_df,orders_df=None,None
    if dem_file is not None:demand_df=pd.read_csv(dem_file)
    if ord_file is not None:orders_df=pd.read_csv(ord_file)
    if demand_df is not None and orders_df is not None:
        a,b=st.columns(2)
        with a:st.markdown('**Demand**');demand_df=st.data_editor(demand_df,num_rows='dynamic',hide_index=True,key='edit_dem')
        with b:st.markdown('**Orders**');orders_df=st.data_editor(orders_df,num_rows='dynamic',hide_index=True,key='edit_ord')
        with st.expander('Optional: observed inventory reconciliation'):
            inv_file=st.file_uploader('Inventory CSV (period, ending_inventory)',type='csv')
            st.caption('Used to compare simulator-predicted inventory with recorded inventory; not used to overwrite demand or infer missing transactions.')
    else:
        st.info('Upload both datasets, or enable the example data.');st.stop()

try:
    required_d={'period','demand'}
    required_o={'order_id','period','quantity','planned_lead','actual_lead'}
    if not required_d.issubset(demand_df) or not required_o.issubset(orders_df):raise ValueError(f'Required columns: demand {sorted(required_d)}, orders {sorted(required_o)}')
    dd=demand_df.copy().sort_values('period')
    oo=orders_df.copy().sort_values('period')
    for col in ['period','demand']:dd[col]=pd.to_numeric(dd[col],errors='raise')
    for col in ['period','quantity','planned_lead','actual_lead']:oo[col]=pd.to_numeric(oo[col],errors='raise')
    if dd['period'].duplicated().any() or oo['order_id'].duplicated().any():raise ValueError('Duplicate demand periods or order IDs.')
    if len(dd)==0 or len(oo)==0:raise ValueError('Both datasets must be nonempty.')
    if not np.array_equal(dd.period.to_numpy(),np.arange(len(dd))):raise ValueError('Demand periods must be contiguous integers 0 through T-1.')
    for col in ['period','quantity','planned_lead','actual_lead']:
        if not np.allclose(oo[col],np.rint(oo[col])):raise ValueError(f'Orders {col} must contain integer values.')
    if (dd.demand<0).any() or (oo[['period','quantity','planned_lead','actual_lead']]<0).any().any():raise ValueError('Negative demand, orders, periods or leads are not supported.')
    if (oo.period>=len(dd)).any():raise ValueError('An order period falls beyond the demand horizon.')
    if (oo.quantity>max_q).any() or (oo.quantity<min_q).any():raise ValueError('Observed orders violate specified order bounds; expand the constraints or fix input.')
    if max_q<min_q:raise ValueError('Maximum order less than minimum.')
    if exact and 2*len(oo)>10:st.info('Exact mode requires at most 10 changing factors; the run will check the number actually changed.')
except Exception as exc:
    st.error(f'Input validation failed: {exc}');st.stop()

cfg=Config(weight=weight,q_min=int(min_q),q_max=int(max_q),initial_inventory=initial)
demand=dd.demand.to_numpy(dtype=float)
periods=oo.period.to_numpy(dtype=int)
obs_q=oo.quantity.to_numpy(dtype=int)
planned=oo.planned_lead.to_numpy(dtype=int)
actual=oo.actual_lead.to_numpy(dtype=int)

if not st.button('▶ Run analysis',type='primary',use_container_width=True):
    st.info('Set inputs and objective weights, then click Run analysis.');st.stop()

with st.spinner('Searching reference order policies and computing Shapley values…'):
    obs=simulate(demand,periods,obs_q,actual,cfg)
    scored=generate_references(demand,periods,obs_q,planned,cfg,n_candidates, max(k,8),42)
    refs=select_references(scored,obs_q,k,weighting)
    attr, diagnostics=allocate(demand,periods,obs_q,actual,planned,refs,cfg,M,42,include_delays,exact)
    ref_losses=[simulate(demand,periods,r['policy'],planned,cfg) for r in refs]
    mean_ref=sum(r['weight']*v['loss'] for r,v in zip(refs,ref_losses))
    ref_stockout=sum(r['weight']*v['stockout_rate'] for r,v in zip(refs,ref_losses))
    ref_inv=sum(r['weight']*v['avg_inventory'] for r,v in zip(refs,ref_losses))
    if not include_delays:
        # Explicitly note endpoint: attribution holds actual delays fixed.
        gap=diagnostics['gap']
    else:gap=obs['loss']-mean_ref

st.subheader('2 · Performance summary')
a,b,c,d=st.columns(4)
a.metric('Observed loss',f"{obs['loss']:.4f}")
b.metric('Weighted reference loss',f'{mean_ref:.4f}')
c.metric('Observed − reference',f'{obs["loss"]-mean_ref:+.4f}')
d.metric('Attributable improvement',f'{diagnostics["gap"]:+.4f}')
if not include_delays:st.warning('Delay attribution disabled: Shapley endpoint uses **actual delays** with reference quantities. The displayed reference performance uses **planned delays**; these are different counterfactuals.')
if obs['loss']-mean_ref<0:st.warning('Selected reference ensemble is worse than the observed trajectory under this combined scenario. Positive improvement is not guaranteed.')
st.caption('Loss = stockout weight × share of periods with unmet demand + inventory weight × mean ending inventory / mean demand. Lower is better.')
metrics=pd.DataFrame({'Metric':['Stockout frequency','Average ending inventory'], 'Observed':[obs['stockout_rate'],obs['avg_inventory']],'Reference':[ref_stockout,ref_inv]})
st.dataframe(metrics,hide_index=True,use_container_width=True)

st.subheader('3 · Factor attribution')
if len(attr):
    attr['order_id']=attr['order_index'].map(lambda i:oo.iloc[i-1].order_id)
    attr=attr.sort_values('contribution',ascending=False)
    a,b=st.columns([2,1])
    with a:
        chart=alt.Chart(attr).mark_bar().encode(x=alt.X('contribution:Q',title='Contribution to improvement (positive = beneficial correction)'),y=alt.Y('factor:N',sort='-x',title=None),color=alt.Color('type:N'),tooltip=['order_id','factor','type',alt.Tooltip('contribution:Q',format='.5f'),alt.Tooltip('se:Q',format='.5f')]).properties(height=max(250,27*len(attr)))
        st.altair_chart(chart,use_container_width=True)
    with b:
        type_sum=attr.groupby('type',as_index=False)['contribution'].sum()
        st.markdown('**Contribution by factor type**')
        st.dataframe(type_sum,hide_index=True,use_container_width=True)
        st.metric('Sum − target gap',f'{diagnostics["residual"]:+.2e}')
        st.caption(f'Changing factors: {diagnostics["n_players"]} · Cached coalition scenarios: {diagnostics["cache_size"]}')
    st.dataframe(attr,hide_index=True,use_container_width=True)
    st.download_button('Download factor attributions',attr.to_csv(index=False).encode(),'shapley_attributions.csv','text/csv')
else:st.info('No order decisions or delivery delays differ from the selected reference.')

st.subheader('4 · Observed and reference inventory trajectories')
traj=pd.DataFrame({'period':obs['trajectory'].period,'Observed':obs['trajectory'].ending_inventory})
for i,(r,_) in enumerate(zip(refs,ref_losses),1):
    traj[f'Reference {i}']=simulate(demand,periods,r['policy'],planned,cfg)['trajectory'].ending_inventory.to_numpy()
st.line_chart(traj.set_index('period'))
if inv_file is not None:
    try:
        inv_df=pd.read_csv(inv_file)
        if not {'period','ending_inventory'}.issubset(inv_df):raise ValueError('Inventory file needs period, ending_inventory.')
        check=inv_df[['period','ending_inventory']].merge(obs['trajectory'][['period','ending_inventory']],on='period',how='inner',suffixes=('_recorded','_simulated'))
        check['difference']=check.ending_inventory_recorded-check.ending_inventory_simulated
        st.markdown('**Inventory reconciliation**')
        st.metric('Max absolute reconciliation difference',f'{check.difference.abs().max():.3f}' if len(check) else 'N/A')
        st.dataframe(check,hide_index=True)
    except Exception as exc:st.error(f'Inventory reconciliation failed: {exc}')

st.subheader('5 · Reference policies & audit')
reference_rows=[]
for i,r in enumerate(refs):
    reference_rows.append({'reference':i+1,'weight':r['weight'],'distance':r['distance'],'loss':r['loss'],**{str(oo.iloc[j].order_id):int(q) for j,q in enumerate(r['policy'])}})
st.dataframe(pd.DataFrame(reference_rows),hide_index=True,use_container_width=True)
st.dataframe(pd.DataFrame(diagnostics['by_reference']),hide_index=True,use_container_width=True)
if abs(diagnostics['residual'])>1e-8:st.error('Shapley efficiency reconciliation failed. Check the simulation.')
else:st.success('Shapley contributions reconcile to the stated counterfactual gap.')
st.download_button('Download observed trajectory',obs['trajectory'].to_csv(index=False),'observed_trajectory.csv','text/csv')
st.info('Interpretation: this is hindsight counterfactual attribution, not a causal estimate from a randomized experiment. The best-found random-search benchmark is not certified globally optimal. Mixed coalition combinations can violate cross-order constraints not modeled here. Demand is assumed fully observed and inventory accounting is deterministic.')
