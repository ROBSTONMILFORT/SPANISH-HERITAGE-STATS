# Hispanic Heritage Historical Data Explorer

A Streamlit classroom app for exploring official World Bank historical
indicators and learning polynomial modeling, rates of change,
derivatives, extrema, and concavity.

## Files

-   `app.py` --- self-contained Streamlit app.
-   `requirements.txt` --- minimal, unpinned runtime dependencies.
-   `sources.md` --- data-source links, indicator definitions, and
    methodology notes.

## Local setup

Python 3.10 or newer is recommended.

``` bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually
`http://localhost:8501`).

## Deploy on GitHub and Streamlit Community Cloud

1.  Create a GitHub repository.
2.  Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md`
    to the repository root.
3.  Visit Streamlit Community Cloud and connect your GitHub account.
4.  Create an app, select the repository and branch, and set the main
    file path to `app.py`.
5.  Deploy. Review the app logs if dependency installation or startup
    fails.

The app does not require API keys. It makes public requests to the World
Bank Indicators API, so network access is required for live data.

## Deploy on Render

Create a **Python Web Service** connected to the GitHub repository.

**Build command**

``` bash
pip install -r requirements.txt
```

**Start command** (single line)

``` bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Set the service to use a Python version compatible with the
dependencies. Check Render logs for build or startup errors.

## Verification checklist

-   [ ] Confirm the source link includes the selected country code and
    indicator code.
-   [ ] Compare several observations against the official World Bank
    indicator page.
-   [ ] Try an indicator with sparse coverage and confirm actual row
    count and years are shown.
-   [ ] Edit a value and confirm the report labels it a user-modified
    scenario.
-   [ ] Press **Reset to source** and confirm the original table
    returns.
-   [ ] Try duplicate years, non-integer years, and non-numeric values
    in the editor.
-   [ ] Try a polynomial degree that exceeds the number of distinct
    observations.
-   [ ] Confirm the centered equation, in-sample R², MAE, residuals, and
    observation count update.
-   [ ] Calculate the same year singly and in a batch; the estimates
    should match up to floating-point rounding.
-   [ ] Test invalid custom years, reversed ranges, and API
    failure/no-network behavior.
-   [ ] Change country, indicator, degree, sample interval, forward
    horizon, comparison country, and printer mode; check that outputs
    respond.
-   [ ] Open CSV, Excel, and PDF exports and confirm they reflect the
    current selection.
-   [ ] In the Mathematics Lab, test `x**2 - 4*x + 3`, `1/x`, `log(x)`,
    and `sin(x)` over intervals that include or avoid their domain
    restrictions.
-   [ ] Check derivatives, tangent/secant lines, critical points,
    extrema, and inflection points against hand calculations for simple
    functions.
-   [ ] Deploy and verify live data access, downloads, and app behavior
    in the hosted environment.

## Troubleshooting

-   **World Bank request timeout / API error:** check internet access,
    retry, and consult the app logs. The app does not invent replacement
    values.
-   **Not enough distinct years:** lower the polynomial degree or select
    an indicator with more observations.
-   **Custom function parse error:** use Python-style mathematical
    notation such as `x**2`, `exp(-x)`, `log(x)`, `sqrt(x)`, or
    `sin(x)`. Use `**` for powers.
-   **Blank parts of a graph:** undefined, non-real, infinite, or
    extreme values are masked to avoid connecting through
    discontinuities.
-   **Excel/PDF download fails:** check that dependencies installed
    successfully and inspect deployment logs.
-   **A calculus result looks surprising:** numerical root finding is
    approximate and can miss roots. Verify against a graph and symbolic
    algebra; do not treat a candidate root as an extremum or inflection
    point without the relevant sign change.

## Mathematics and data assumptions

-   The data request is current year minus 70 through current year minus
    1, inclusive. Actual non-null coverage can be shorter.
-   The fitted polynomial uses centered time `t = year - base_year`.
-   R² and MAE are in-sample unless explicitly labeled chronological
    holdout.
-   Polynomial extrapolation is uncertain and is not a validated
    forecast interval.
-   The interactive contextual functions are educational examples and do
    not establish real-world causation.
