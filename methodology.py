"""Human-readable documentation of the implemented computational pipeline."""
import streamlit as st


def show_methodology():
    st.title('📘 Methodology & Computation')
    st.caption('A detailed guide to the calculations performed by this version of Inventory Decision Attribution.')
    st.info('**Implementation scope:** one SKU, one location, integer-period review/order dates, full receipts, fixed realized demand, and deterministic inventory accounting. The optimization is a retrospective best-found random search, not a proof of global optimality.')

    st.markdown(r'''### Contents
1. [Input data and modeling assumptions](#input-data-and-modeling-assumptions)
2. [Inventory simulation](#inventory-simulation)
3. [Evaluation objective](#evaluation-objective)
4. [Constructing feasible reference policies](#constructing-feasible-reference-policies)
5. [Choosing and weighting benchmarks](#choosing-and-weighting-benchmarks)
6. [Shapley allocation to orders and delays](#shapley-allocation-to-orders-and-delays)
7. [Monte Carlo computation and diagnostics](#monte-carlo-computation-and-diagnostics)
8. [Reading results and worked example](#reading-results-and-worked-example)
9. [Limits and reproducibility](#limits-and-reproducibility)
''')
    st.divider()
    st.header('1. Input data and modeling assumptions', anchor='input-data-and-modeling-assumptions')
    st.markdown(r'''Upload a **demand file** with `period, demand` and an **orders file** with
`order_id, period, quantity, planned_lead, actual_lead`.

- **Period:** integer time index, beginning at 0; demand observations must cover every period through T−1.
- **Demand:** nonnegative demand per period, assumed to be *true unconstrained demand*. Observed sales alone can underestimate demand during stockouts.
- **Order quantity:** nonnegative integer quantity of each placed order, subject to user-entered per-order minimum and maximum.
- **Lead time:** nonnegative integer periods. An order entered at time t with lead L arrives at the **start** of period t+L, before demand is served.
- **Initial inventory:** starting on-hand stock before period-0 arrivals. The engine also supports an initial pipeline internally, but this version of the UI does not expose it.

The data is held fixed for all counterfactuals. The uploaded optional inventory file is only used to **reconcile** the simulated historical trajectory; it does not change demand, arrivals, or optimization. No orders arrive partially, and no backorders are carried over. Demand not filled during its period is lost.''')
    st.header('2. Inventory simulation', anchor='inventory-simulation')
    st.markdown('Let $D_t$ be demand, $I_{t-1}$ the preceding period\'s ending inventory, and $R_t$ the receipts arriving at the start of period $t$. For order $j$, let $t_j$ denote placement time, $q_j$ its quantity, and $L_j$ its lead.')
    st.latex(r'R_t=\sum_{j:\,t_j+L_j=t}q_j\quad\text{(+ any initial pipeline receipts)}')
    st.latex(r'A_t=I_{t-1}+R_t')
    st.latex(r'S_t=\min(A_t,D_t),\qquad U_t=\max(D_t-A_t,0)')
    st.latex(r'I_t=\max(A_t-D_t,0),\qquad Z_t=\mathbf{1}\{U_t>0\}')
    st.markdown('The simulator records opening stock, receipts, demand, fulfilled demand, lost demand, ending inventory, and a binary stockout indicator each period. Orders arriving after the final period do not affect the evaluation window. Every candidate or Shapley coalition is **re-simulated from the same starting inventory**; the app does not simply edit historical inventory observations.')

    st.header('3. Evaluation objective', anchor='evaluation-objective')
    st.markdown('The user changes the **stockout weight** $w$ in the sidebar. The inventory weight is exactly $1-w$. Lower objective values are preferred.')
    st.latex(r'\mathrm{OOS}=\frac{1}{T}\sum_{t=0}^{T-1} Z_t')
    st.latex(r'\overline{I}=\frac{1}{T}\sum_{t=0}^{T-1}I_t,\qquad \overline{D}=\frac{1}{T}\sum_{t=0}^{T-1}D_t')
    st.latex(r'J(q,L)=w\,\mathrm{OOS}+(1-w)\,\frac{\overline I}{\max(\overline D,10^{-8})}')
    st.markdown(r'''**Important:** OOS is the *fraction of periods with any unmet demand*, not the fraction of units lost. The inventory term is **average ending inventory normalized by average demand**. It is not the amount of inventory above a safety-stock threshold. The denominator floor of $10^{-8}$ avoids division by zero for zero-demand data.

Increasing $w$ emphasizes avoiding stockout periods. With $w=1$ only stockout frequency matters; with $w=0$ only normalized ending inventory matters. There is no separate per-order procurement cost, per-unit shortage cost, or delay penalty in this objective.''')
    st.header('4. Constructing feasible reference policies', anchor='constructing-feasible-reference-policies')
    st.markdown(r'''The app searches for alternative **order-quantity vectors** $q^*=(q^*_1,\dots,q^*_n)$, keeping the uploaded order dates fixed and assuming **planned lead times** for search scoring. It does **not** optimize order dates or lead times.

**Implemented search procedure (`engine.generate_references`):**

1. Start with the observed order vector, clipped to the specified quantity bounds.
2. Add four heuristic starting points: demand summed over an approximate review window; 80% and 120% of that estimate; and a constant quantity based on average demand and the review interval. The interval is the median difference between order dates (or 4 for one order).
3. Generate the specified number of random proposals using three mechanisms: perturb observed quantities, perturb demand-based quantities, or sample uniformly within the allowable range.
4. Round all proposals to integers, clip each quantity to `[q_min, q_max]`, remove duplicate vectors, and compute $J(q, L^{planned})$ for every unique candidate.
5. Sort candidates by objective value (ascending) and **retain the top `max(K, 8)`** candidates, where $K$ is the requested number of final benchmarks.

This is **random search with heuristic seeds**, not a genetic algorithm, dynamic programming solution, or exhaustive optimizer. A candidate is feasible with respect to the *implemented per-order bounds*. Aggregate capacity, storage capacity, supplier allocation, minimum order multiples, and multi-location coupling are not implemented. Thus, “optimal” in dashboard descriptions means *best found among the searched candidates*.''')
    st.latex(r'\mathcal C_{best}=\operatorname{Top}_{\max(K,8)}\left\{J(q,L^{planned}):q\in\mathcal C\right\}')
    st.header('5. Choosing and weighting benchmarks', anchor='choosing-and-weighting-benchmarks')
    st.markdown(r'''**Selection is not simply the top K lowest-loss candidates.** The engine first retains the best-performing candidate shortlist, then selects the **K closest to observed quantities** by normalized Euclidean distance. The distance exponent is fixed at 2 in the current UI.''')
    st.latex(r'd(q,q^{obs})=\left[\sum_{j=1}^{n}\left|\frac{q_j-q^{obs}_j}{\max(1,|q^{obs}_j|)}\right|^2\right]^{1/2}')
    st.markdown(r'''Two weighting options are available:

- **`simple`:** each selected reference has weight $1/K$.
- **`distance`:** references closer to observed orders receive larger weights via inverse distance. A small stabilizer $10^{-6}$ is added to distance before inversion.

Weights sum to 1. This produces a **mixture of reference scenarios**, not a single policy built by rounding a weighted-average order vector.''')
    st.latex(r'\omega_k=\frac{(d_k+10^{-6})^{-1}}{\sum_{h=1}^{K}(d_h+10^{-6})^{-1}}\quad\text{(distance weighting)}')
    st.latex(r'J_{ref}=\sum_{k=1}^K\omega_k J(q^{(k)},L^{planned})')
    st.markdown('Because $J$ is nonlinear, averaging reference **scores** is generally not equivalent to evaluating a weighted-average quantity vector. The app deliberately uses the weighted average of the *separately evaluated* reference losses.')

    st.header('6. Shapley allocation to orders and delays', anchor='shapley-allocation-to-orders-and-delays')
    st.markdown(r'''The observed endpoint is $J(q^{obs},L^{actual})$. A Shapley **player** is either (a) one order decision that differs from a selected reference in at least one benchmark, or (b) one delivery lead-time deviation, if **Attribute delivery deviations** is checked.

For each selected reference $k$, define a coalition $S$ of factors already corrected. The coalition scenario replaces:

- $q_j^{obs}$ by $q_j^{(k)}$ if its **Order j** player belongs to $S$;
- $L_j^{actual}$ by $L_j^{planned}$ if its **Delay j** player belongs to $S$;
- all other factors remain at observed values.

The same realized demand and simulation rules apply in every coalition. This retains nonlinear interactions between order timing, arrival timing, stockouts, and ending inventory. A “delay” player means a **lead-time deviation from the plan**, including an early arrival, not necessarily a positive late delay.''')
    st.latex(r'v_k(S)=J\big(q^{(k,S)},L^{(S)}\big)')
    st.latex(r'\phi_{j,k}=\sum_{S\subseteq N\setminus\{j\}}\frac{|S|!(|N|-|S|-1)!}{|N|!}\left[v_k(S\cup\{j\})-v_k(S)\right]')
    st.markdown(r'The mathematical $\phi$ is a **change in loss** when a factor is corrected. The dashboard reports its negative, called *contribution to improvement*:')
    st.latex(r'C_j=-\sum_{k=1}^{K}\omega_k\phi_{j,k}')
    st.markdown(r'''**Positive** contribution: correcting the factor tends to reduce the objective, accounting for interactions. **Negative** contribution: correcting the factor tends to increase the objective. Both signs can occur even when the total scenario improves.

**If delay attribution is enabled:** all differing delays and order quantities are eligible players, so the full coalition for reference $k$ reaches $(q^{(k)},L^{planned})$.

**If delay attribution is disabled:** only order changes are players, and delays remain **actual**, including at the full coalition. The Shapley target is then $J(q^{obs},L^{actual})-\sum_k\omega_k J(q^{(k)},L^{actual})$, **not** the separate dashboard reference KPI evaluated under planned delays. The app displays a warning in this case.''')

    st.header('7. Monte Carlo computation and diagnostics', anchor='monte-carlo-computation-and-diagnostics')
    st.markdown(r'''**Monte Carlo mode (default).** For every reference, the engine draws $M$ random permutations of the changing players. It walks through each permutation from the empty coalition, adding one correction at a time, and records the change in loss caused by each addition. It also evaluates the **reverse permutation** and averages the two (antithetic pairing). The $M$ pair-averaged contributions produce a per-factor estimate and a standard error.''')
    st.latex(r'\widehat\phi_{j,k}=\frac{1}{M}\sum_{m=1}^M\frac{\Delta_{j,k,m}^{forward}+\Delta_{j,k,m}^{reverse}}{2}')
    st.latex(r'\widehat{SE}_{j,k}=\frac{\mathrm{sd}(\Delta_{j,k,1}^{pair},\ldots,\Delta_{j,k,M}^{pair})}{\sqrt M}')
    st.markdown(r'''The code combines benchmark standard errors as $\sqrt{\sum_k(\omega_k\widehat{SE}_{j,k})^2}$. This treats the benchmark estimates as independent conditional on the selected references. These are **Monte Carlo sampling errors**, not confidence intervals for real-world causal effects or uncertainty from input data.

**Exact mode:** for at most **10 changing players**, compute the factorial-weighted Shapley sum over all subsets, rather than sample permutations. The reported numerical SE is zero because there is no Monte Carlo sampling (not because inputs are known without uncertainty).

**Caching:** within each reference, a coalition's loss is computed once and cached under its bit-mask. Cache-size diagnostics count the number of distinct coalition evaluations summed across references.

**Efficiency audit:** because each permutation telescopes from the observed scenario to its reference endpoint, estimated contributions should sum to the reported Shapley target, up to floating-point error.''')
    st.latex(r'\sum_j C_j\;\approx\;J(q^{obs},L^{actual})-\sum_k\omega_k J(q^{(k)},L^{endpoint})')
    st.markdown('Here $L^{endpoint}=L^{planned}$ when delay attribution is on, and $L^{endpoint}=L^{actual}$ otherwise. The residual displayed in the dashboard is **sum of contributions minus this target**. A small residual verifies accounting consistency but **does not** validate that the reference search found a global optimum.')

    st.header('8. Reading results and worked example', anchor='reading-results-and-worked-example')
    st.markdown(r'''**Illustrative arithmetic (not the app demo output):** Suppose the observed loss is 0.50 and the weighted reference loss is 0.40. A decomposition might attribute an improvement of +0.07 to correcting order quantities and +0.03 to restoring planned lead times. Then $0.07+0.03=0.10=0.50-0.40$. Individual factors can still have negative contributions.

In the actual dashboard:

- **Observed loss:** calculated from the uploaded quantities and actual leads.
- **Weighted reference loss:** average of best-found reference losses scored at planned leads.
- **Observed − reference:** difference between those two displayed KPIs.
- **Attributable improvement:** the endpoint difference *actually used in Shapley*, which differs from the displayed KPI gap if delay attribution is disabled.
- **Factor attribution:** sorted player-level contributions with Monte Carlo SE, order ID, observed quantity, weighted reference quantity, and lead times.
- **Trajectories:** observed stock path plus a separate path for each selected reference; there is no single averaged reference trajectory.
- **Reference table:** one row per reference, including its selection weight, normalized distance, objective score, and individual quantities.
- **Audit table:** per-reference observed/reference endpoint losses, weights, and gaps.''')

    st.header('9. Limits and reproducibility', anchor='limits-and-reproducibility')
    st.markdown(r'''This implementation is a **retrospective what-if attribution tool**. It does not prove causation from an experiment, estimate the demand that would have been observed under different inventory availability, or produce a deployable forward-looking ordering policy. The reference search has hindsight access to the full realized demand series. Delivery timing is changed counterfactually without re-estimating supplier behavior. Parameters are deterministic once inputs and seeds are fixed (candidate search seed 42; Shapley seed 42).

The code is structured as:

- `engine.py` → `simulate`: inventory accounting and objective;
- `engine.py` → `generate_references`: candidate generation and scoring;
- `engine.py` → `select_references`: proximity selection and weights;
- `engine.py` → `allocate`: coalition simulations, Shapley values, standard errors, and audit;
- `app.py` → data input, validation, controls, plots, CSV exports;
- `methodology.py` → this explanatory page.

These explanations describe the **actual Streamlit implementation**, which is a streamlined single-location adaptation of the uploaded research code. Its random search, fixed order dates, delay-player extension, and lack of aggregate-capacity constraints should not be confused with features of a more general mathematical model. For the published research context, see the [Computers article](https://www.mdpi.com/2073-431X/15/6/375).''')
    st.success('Return to **Analysis dashboard** using the navigation menu in the sidebar.')
