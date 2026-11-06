# MVP vs HRP NIFTY 50 comparison

This Streamlit app compares two long-only, fully invested portfolios using daily **Close** prices from Yahoo Finance:

- **Minimum Variance Portfolio (MVP):** constrained covariance-matrix optimization.
- **Hierarchical Risk Parity (HRP):** correlation-distance clustering, quasi-diagonalization, and recursive bisection.

The supplied default basket contains 10 liquid, large NIFTY 50 names. It is deliberately a practical default rather than a claim that these are a dynamically verified current top-10 ranking. Yahoo Finance symbols and the basket can be changed in the sidebar.

## Dates and no-look-ahead design

- Training window: **1 Jan 2025–30 Apr 2026**
- Test window: **1 May 2026–31 Oct 2026**

Weights are fitted from training Close-to-Close returns only and then held unchanged throughout the test period. The first eligible test return uses the close immediately before the test date as its denominator, which correctly captures the boundary Close-to-Close return without allowing test returns into either weight calculation.

The app asks Yahoo Finance for data through 31 October 2026. Since 31 October 2026 is a Saturday, the last expected market date is 30 October. It reports the actual returned range and warns when Yahoo Finance does not supply the complete expected range. Tickers that have any missing Close prices in the requested period are excluded rather than imputed.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit. Internet access is required while downloading price data.

## Outputs

The application shows annualized return, annualized volatility, Sharpe ratio, maximum drawdown, portfolio-value and risk/return charts, allocation pies/tables, and CSV downloads. The annual risk-free rate is editable and defaults to 0%, so the displayed Sharpe ratio is explicit about its assumption.
