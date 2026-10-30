# Inventory Decision Attribution — Streamlit MVP

This app adapts the user's multi-location inventory simulation, objective function, random-search references, distance-based benchmark selection, and antithetic permutation Shapley estimator into a **single-SKU, single-location** dashboard. It additionally attributes planned-vs-actual lead-time deviations.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Inputs

- `demand.csv`: `period,demand` (integer periods 0…T−1; unconstrained demand, not merely sales when there are stockouts).
- `orders.csv`: `order_id,period,quantity,planned_lead,actual_lead` (period/lead columns are nonnegative integers; one unique ID per order).
- Optional `inventory.csv`: `period,ending_inventory` (used for auditing only).
- Set initial on-hand inventory and max order quantity in the sidebar. Existing pre-horizon pipeline and partial receipts are not yet supported.

The app provides editable example data if you have not uploaded files. The algorithm includes each changed order quantity and each changed lead-time realization as a separate player. Its coalition value starts from observed orders/actual leads and switches the selected players to reference quantities/planned leads. The Shapley sum equals observed loss minus the **weighted mean of the individual reference losses**; do not evaluate a rounded weighted-average reference policy and expect its nonlinear loss to match this mean.

**Limits:** Best-found stochastic search does not guarantee a global optimum. The app uses a realized demand path (perfect hindsight), no supplier/shared-capacity constraints, no split deliveries, no ordering while goods are already in transit before t=0, and no multi-location coupling. Its objective is stockout-period frequency plus normalized average ending inventory, not units of lost demand. If only sales are available and stockouts occurred, demand is censored and cannot be substituted without an estimate.

## Methodology page

Use **Navigation → Methodology & computation** in the sidebar for a detailed explanation of the actual objective, simulation, reference search and selection, lead-time counterfactuals, Shapley estimation, uncertainty, endpoints, and interpretation. The page is implemented in `methodology.py`.
