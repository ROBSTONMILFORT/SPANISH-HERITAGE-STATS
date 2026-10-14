"""
Hispanic Heritage Historical Data Explorer
Run with: python -m streamlit run app.py
World Bank API data are retrieved live; no historical observations are fabricated.
"""
from __future__ import annotations

import io
import math
import time
import re
from datetime import date, datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    PageBreak
)

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Expected years a newborn would live if current age-specific mortality rates applied."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people (net)", "Number of immigrants minus emigrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP."),
}
CONTEXTS = {
    "Business": ("Revenue", "US dollars", "Profit", "US dollars", "A business metric over time."),
    "Engineering": ("Time", "seconds", "Position", "meters", "A moving system's position over time."),
    "Medical": ("Time", "hours", "Concentration", "mg/L", "A measured concentration over time; not medical advice."),
    "Sports science": ("Time", "seconds", "Speed", "m/s", "An athlete's performance measure over time."),
    "Biology": ("Time", "days", "Population", "organisms", "A population or colony over time."),
    "Bioengineering": ("Time", "hours", "Signal", "arbitrary units", "A biological or engineered signal over time."),
    "Economic forecasting": ("Year", "year", "Economic indicator", "indicator units", "A mathematical scenario, not a forecast guarantee."),
    "Weather": ("Time", "hours", "Temperature", "°C", "A temperature trend over time."),
    "Custom": ("x", "x-units", "f(x)", "y-units", "A user-defined context."),
}
SUGGESTED = {
    "Business": "a*x**2 + b*x + c",
    "Engineering": "a*x**2 + b*x + c",
    "Medical": "a*exp(-k*x) + c",
    "Sports science": "a*sin(k*x) + c",
    "Biology": "a*exp(k*x)",
    "Bioengineering": "a*sin(k*x) + b",
    "Economic forecasting": "a*x**2 + b*x + c",
    "Weather": "a*sin(k*x) + b",
    "Custom": "a*x**2 + b*x + c",
}
PARAMS = {"a": 1.0, "b": 0.0, "c": 0.0, "k": 0.2}

def wb_fetch(country_code: str, indicator_code: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, Optional[str]]:
    """Fetch yearly observations from the official World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    try:
        response = requests.get(url, params={"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observations for this selection."
        rows = [{"year": int(item["date"]), "value": item["value"]} for item in payload[1] if item.get("date")]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["year", "value"]), "The API response contained no year records."
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        return frame, None if not frame.empty else "The API returned records, but all indicator values were null."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "World Bank request timed out. Try again later."
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"

def validate_data(frame: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    out = frame.copy()
    if not {"year", "value"}.issubset(out.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must have year and value columns."]
    years = pd.to_numeric(out["year"], errors="coerce")
    values = pd.to_numeric(out["value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Years must be whole-number integers.")
    if years.duplicated().any():
        errors.append("Years must be unique; remove duplicate years.")
    if values.isna().any():
        errors.append("Values must be numeric and cannot be blank.")
    if indicator_name == "Life expectancy at birth" and ((values <= 0) | (values > 125)).any():
        errors.append("Life expectancy values must be greater than 0 and at most 125.")
    if indicator_name == "Unemployment, national estimate" and ((values < 0) | (values > 100)).any():
        errors.append("Unemployment percentages must be between 0 and 100.")
    if indicator_name == "Intentional homicide rate" and (values < 0).any():
        errors.append("Homicide rates cannot be negative.")
    if indicator_name == "Crude birth rate" and (values < 0).any():
        errors.append("Crude birth rates cannot be negative.")
    out = pd.DataFrame({"year": years, "value": values})
    if not errors:
        out = out.astype({"year": int, "value": float}).sort_values("year").reset_index(drop=True)
    return out, errors

def fit_polynomial(frame: pd.DataFrame, degree: int):
    if len(frame) < degree + 1 or frame["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid observations.")
    base_year = int(frame["year"].min())
    t = frame["year"].to_numpy(dtype=float) - base_year
    y = frame["value"].to_numpy(dtype=float)
    coef = np.polyfit(t, y, degree)
    pred = np.polyval(coef, t)
    residuals = y - pred
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(residuals ** 2) / ss_tot) if ss_tot else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    return base_year, coef, pred, residuals, r2, mae

def poly_equation(coef, base_year, y_name="y"):
    terms = []
    degree = len(coef) - 1
    for i, c in enumerate(coef):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        if power == 0:
            part = f"{c:.8g}"
        elif power == 1:
            part = f"{c:.8g}·t"
        else:
            part = f"{c:.8g}·t^{power}"
        terms.append(part)
    return f"{y_name}(t) = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0") + f", where t = year − {base_year}"

def predict_year(year, base_year, coef):
    return float(np.polyval(coef, float(year) - base_year))

def safe_symbolic(expr_text):
    x = sp.symbols("x", real=True)
    allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
               "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E,
               "a": sp.symbols("a"), "b": sp.symbols("b"), "c": sp.symbols("c"), "k": sp.symbols("k")}
    try:
        expr = sp.sympify(expr_text, locals=allowed, evaluate=True)
        if expr.has(sp.Function) and any(f.func not in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt) for f in expr.atoms(sp.Function)):
            raise ValueError("Function not supported.")
        if any(not isinstance(sym, sp.Symbol) or sym.name != "x" for sym in expr.free_symbols):
            raise ValueError("Use x and numeric constants only in a custom expression.")
        return x, expr
    except Exception as exc:
        raise ValueError(f"Could not parse function. Use forms such as x**2, exp(-0.2*x), log(x), or sin(x). Details: {exc}")

def numeric_real(expr, x, values):
    try:
        val = complex(sp.N(expr.subs(x, values)))
        return float(val.real) if abs(val.imag) < 1e-7 and math.isfinite(val.real) else np.nan
    except Exception:
        return np.nan

def make_chart(frame, base_year=None, coef=None, degree=None, single=None, batch=None, sample_increment=1, horizon=0, title="Historical data and model", x_label="Year", y_label="Value"):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not frame.empty:
        ax.scatter(frame["year"], frame["value"], label="Observed / working data", color="#1f77b4", s=30, zorder=3)
    if coef is not None and base_year is not None and not frame.empty:
        lo, hi = int(frame["year"].min()), int(frame["year"].max())
        display_horizon = max(0, int(horizon))
        if single is not None: display_horizon = max(display_horizon, int(single) - hi)
        if batch is not None and len(batch): display_horizon = max(display_horizon, int(max(batch)) - hi)
        xs = np.linspace(lo, max(hi, hi + display_horizon), 500)
        ys = np.polyval(coef, xs - base_year)
        hist = xs <= hi
        if hist.any(): ax.plot(xs[hist], ys[hist], label=f"Degree {degree} fitted curve (historical)", linewidth=2)
        if (~hist).any(): ax.plot(xs[~hist], ys[~hist], "--", label="Extrapolated segment", linewidth=2, color="#d62728")
        ax.axvline(hi, linestyle=":", color="gray", label="Historical boundary")
        if sample_increment > 1:
            sample_years = np.arange(lo, hi + 1, sample_increment)
            sample_vals = np.polyval(coef, sample_years - base_year)
            ax.scatter(sample_years, sample_vals, marker="x", label=f"Model markers every {sample_increment} year(s)", color="#2ca02c")
    if single is not None and coef is not None:
        ax.scatter([single], [predict_year(single, base_year, coef)], marker="*", s=140, label=f"Estimate for {single}", color="#ff7f0e")
    if batch is not None and coef is not None and len(batch):
        by = np.array(batch, dtype=int)
        ax.scatter(by, [predict_year(v, base_year, coef) for v in by], marker="D", s=38, label="Batch estimates", color="#9467bd")
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def create_pdf(title, metadata, equation, metrics, findings, preview, results, fig):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch, topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Paragraph(metadata, styles["Normal"]), Spacer(1, 8),
             Paragraph("Model", styles["Heading2"]), Paragraph(equation.replace("&", "&amp;"), styles["BodyText"]),
             Paragraph(metrics.replace("&", "&amp;"), styles["BodyText"]), Spacer(1, 8)]
    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=140, bbox_inches="tight")
    img.seek(0)
    story += [RLImage(img, width=9.3*inch, height=4.1*inch), Paragraph("Findings and limitations", styles["Heading2"]),
              Paragraph(findings.replace("&", "&amp;"), styles["BodyText"]), Spacer(1, 8)]
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    def safe_table(df, max_rows=12):
        data = [list(df.columns)] + df.head(max_rows).astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9e6f2")),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
        return tbl
    story.append(safe_table(preview))
    if not results.empty:
        story += [Spacer(1, 8), Paragraph("Calculated-year results", styles["Heading2"]), safe_table(results)]
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

st.title("📊 Hispanic Heritage Historical Data Explorer")
st.caption("Hispanic Heritage Data Lab • Official World Bank indicators • Interactive mathematics")
st.info("Brazil is included as a Latin American comparison country. It is not a Spanish-speaking country, but it is an important regional comparison.")

with st.sidebar:
    st.header("Historical Data Lab")
    country = st.selectbox("Featured country", list(COUNTRIES))
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph model-marker interval (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_country = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country])
    printer_mode = st.toggle("Printer-friendly view", False)

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
code, units, definition = INDICATORS[indicator_name]
st.markdown(f"**Research question:** How has *{indicator_name.lower()}* changed in {country}, and how does its historical pattern compare with {compare_country if compare_country != 'None' else 'a single-country trend'}?")
st.caption(f"Data window requested: {start_year}–{end_year} inclusive • World Bank indicator `{code}` • Retrieved {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}")
st.write(f"**Definition:** {definition}  \n**Units:** {units}  \n**Source:** [World Bank indicator page](https://api.worldbank.org/v2/country/{COUNTRIES[country]}/indicator/{code}?format=json)")

cache_key = f"wb_{COUNTRIES[country]}_{code}_{start_year}_{end_year}"
if st.button("Refresh source data"):
    st.cache_data.clear()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(country_code, indicator_code, start_year, end_year):
    return wb_fetch(country_code, indicator_code, start_year, end_year)
source_df, fetch_error = cached_fetch(COUNTRIES[country], code, start_year, end_year)
if fetch_error:
    st.error(fetch_error)
    st.warning("No replacement or random values are used. Check your connection and try Refresh source data.")
else:
    st.success(f"Retrieved {len(source_df)} non-null annual observations; actual coverage: {int(source_df.year.min())}–{int(source_df.year.max())}. This may be fewer than 70 observations.")

if "working_data" not in st.session_state or st.session_state.get("working_key") != cache_key:
    st.session_state["working_data"] = source_df.copy()
    st.session_state["working_key"] = cache_key
    st.session_state["data_edited"] = False
working_original = st.session_state["working_data"].copy()
if not fetch_error:
    edited = st.data_editor(working_original, num_rows="dynamic", use_container_width=True, key=f"editor_{cache_key}", column_config={
        "year": st.column_config.NumberColumn("Year", step=1, format="%d"),
        "value": st.column_config.NumberColumn(f"Value ({units})", format="%.6f"),
    })
    valid_df, validation_errors = validate_data(edited, indicator_name)
    if validation_errors:
        for error in validation_errors: st.error(error)
    else:
        st.session_state["working_data"] = valid_df
        if not valid_df.equals(source_df.reset_index(drop=True)):
            st.session_state["data_edited"] = True
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        if st.button("Reset to source"):
            st.session_state["working_data"] = source_df.copy()
            st.session_state["data_edited"] = False
            st.rerun()
    with col2:
        st.write("**Status:** " + ("User-modified scenario" if st.session_state.get("data_edited") else "Source data"))
    with col3:
        st.caption("Edit year/value cells to explore a scenario. Reset restores the downloaded source table.")
else:
    valid_df = pd.DataFrame(columns=["year", "value"])
    validation_errors = ["No source data available."]

model_ok = False
if not validation_errors and len(valid_df) >= degree + 1 and valid_df["year"].nunique() >= degree + 1:
    try:
        base_year, coef, fitted, residuals, r2, mae = fit_polynomial(valid_df, degree)
        model_ok = True
    except Exception as exc:
        st.error(str(exc))
elif not validation_errors:
    st.warning(f"Degree {degree} requires at least {degree + 1} distinct valid years; current valid rows: {len(valid_df)}.")

st.subheader("Historical model")
if model_ok:
    equation = poly_equation(coef, base_year)
    metrics_text = f"In-sample R²: {r2:.6f} • In-sample MAE: {mae:.6g} {units} • Observations: {len(valid_df)} • Historical range: {valid_df.year.min()}–{valid_df.year.max()} • Mean: {valid_df.value.mean():.6g} {units}"
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("In-sample R²", f"{r2:.4f}" if np.isfinite(r2) else "undefined")
    c2.metric("In-sample MAE", f"{mae:.4g} {units}")
    c3.metric("Valid observations", str(len(valid_df)))
    c4.metric("Historical coverage", f"{int(valid_df.year.min())}–{int(valid_df.year.max())}")
    st.code(equation)
    st.caption("The model uses centered time t = year − base year. Metrics describe fit to the same observations used to estimate the coefficients; they are not proof of predictive accuracy.")
    if printer_mode:
        st.markdown("### Printer-friendly report")
        st.write(f"**Country / indicator:** {country} — {indicator_name}")
        st.write(f"**Source status:** {'User-modified scenario' if st.session_state.get('data_edited') else 'World Bank source data'}")
        st.write(f"**Equation:** {equation}")
        st.write(f"**Metrics:** {metrics_text}")
        st.write(f"**Limitation:** Polynomial extrapolation is uncertain; no validated 95% prediction interval is provided.")
    residual_table = valid_df.copy()
    residual_table["fitted_value"] = fitted
    residual_table["residual"] = residuals
    with st.expander("View fitted values and residuals"):
        st.dataframe(residual_table, use_container_width=True)
    fig = make_chart(valid_df, base_year, coef, degree, sample_increment=sample_increment, horizon=horizon, title=f"{country}: {indicator_name}", y_label=f"{indicator_name} ({units})")
    st.pyplot(fig, use_container_width=True)
    st.caption(f"Forward display horizon control: {horizon} year(s). Graphs distinguish observed points, fitted history, the historical boundary, and extrapolation.")
    # Chronological holdout: reserve final 20% if enough points remain for the chosen degree.
    n = len(valid_df)
    split = int(np.floor(n * 0.8))
    if split >= degree + 1 and n - split >= 2:
        train = valid_df.iloc[:split]
        test = valid_df.iloc[split:]
        try:
            hb, hc, *_ = fit_polynomial(train, degree)
            hp = np.array([predict_year(y, hb, hc) for y in test.year])
            holdout_mae = float(np.mean(np.abs(test.value.to_numpy() - hp)))
            st.write(f"**Chronological holdout MAE:** {holdout_mae:.6g} {units} (last {len(test)} observations held out; separate from in-sample metrics).")
        except Exception:
            st.caption("Chronological holdout score could not be calculated for this selection.")
    else:
        st.caption("Not enough observations for a chronological holdout comparison with this polynomial degree.")
    # Single and batch estimates
    st.subheader("Calculate values")
    a,b = st.columns(2)
    with a:
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(int(valid_df.year.max())+1,1950),2100), step=1)
        calculate_single = st.button("Calculate Value")
    with b:
        mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
        if mode == "Year range":
            range_start = st.number_input("Start year", min_value=1950, max_value=2100, value=2000, step=1)
            range_end = st.number_input("End year", min_value=1950, max_value=2100, value=2030, step=1)
            range_step = st.slider("Batch range step (years)", 1, 10, 5)
            custom_text = ""
        else:
            custom_text = st.text_input("Comma-separated years", "2000, 2010, 2020, 2030")
            range_start, range_end, range_step = 0, -1, 1
        calculate_batch = st.button("Calculate batch")
    if calculate_single:
        st.session_state["single_result"] = {"year": int(single_year), "value": predict_year(single_year, base_year, coef), "key": (country, indicator_name, degree, tuple(valid_df.year), tuple(np.round(valid_df.value, 8)))}
    if calculate_batch:
        try:
            if mode == "Year range":
                if range_end < range_start: raise ValueError("End year must be at least the start year.")
                years = list(range(int(range_start), int(range_end)+1, int(range_step)))
            else:
                years = [int(v.strip()) for v in custom_text.split(",") if v.strip()]
                if not years: raise ValueError("Enter at least one comma-separated year.")
            invalid = [y for y in years if y < 1950 or y > 2100]
            if invalid: raise ValueError(f"Years must be between 1950 and 2100. Invalid: {invalid}")
            if len(years) > 1000: raise ValueError("Limit batch requests to 1,000 years.")
            st.session_state["batch_result"] = {"years": years, "key": (country, indicator_name, degree, tuple(valid_df.year), tuple(np.round(valid_df.value, 8)))}
            st.session_state["batch_error"] = None
        except Exception as exc:
            st.session_state["batch_error"] = str(exc)
    model_key = (country, indicator_name, degree, tuple(valid_df.year), tuple(np.round(valid_df.value, 8)))
    single = st.session_state.get("single_result")
    if single and single.get("key") == model_key:
        sy, sv = single["year"], single["value"]
        classification = "within fitted historical range" if valid_df.year.min() <= sy <= valid_df.year.max() else "outside fitted historical range (extrapolation)"
        observed = valid_df.loc[valid_df.year == sy, "value"]
        st.write(f"**Estimate for {sy}:** {sv:.8g} {units} — {classification}.")
        if not observed.empty:
            st.caption(f"At this year, actual observed value is {observed.iloc[0]:.8g} {units}; fitted estimate is not the same as the observation.")
        if not np.isfinite(sv) or abs(sv) > max(1e9, 1000 * max(1, valid_df.value.abs().max())):
            st.warning("This estimate is extreme relative to the observed scale. Polynomial extrapolation can become implausible; it has not been clipped.")
    elif single:
        st.info("Single-year result invalidated because the country, indicator, degree, or working data changed. Calculate again.")
    if st.session_state.get("batch_error"):
        st.error(st.session_state["batch_error"])
    batch_state = st.session_state.get("batch_result")
    batch_df = pd.DataFrame()
    if batch_state and batch_state.get("key") == model_key:
        years = batch_state["years"]
        batch_df = pd.DataFrame({"year": years, "estimate": [predict_year(y, base_year, coef) for y in years], "units": units})
        batch_df["classification"] = np.where((batch_df.year >= valid_df.year.min()) & (batch_df.year <= valid_df.year.max()), "within fitted range", "outside fitted range")
        st.dataframe(batch_df, use_container_width=True)
        if single and single.get("key") == model_key and single["year"] in years:
            matching = batch_df.loc[batch_df.year == single["year"], "estimate"]
            if not matching.empty:
                st.caption(f"Consistency check: single-year and batch estimate differ by {abs(float(matching.iloc[0])-single['value']):.3g} {units} (should be 0 apart from floating-point rounding).")
    elif batch_state:
        st.info("Batch results invalidated because the country, indicator, degree, or working data changed. Calculate again.")
    # Comparison country
    comparison_df = pd.DataFrame()
    if compare_country != "None":
        compare_frame, compare_error = cached_fetch(COUNTRIES[compare_country], code, start_year, end_year)
        if compare_error:
            st.warning(f"Comparison data unavailable: {compare_error}")
        else:
            comparison_df = compare_frame
            comp_lo, comp_hi = comparison_df.year.min(), comparison_df.year.max()
            main_lo, main_hi = valid_df.year.min(), valid_df.year.max()
            common = max(0, min(main_hi, comp_hi) - max(main_lo, comp_lo) + 1)
            st.write(f"**Country comparison:** {country} covers {main_lo}–{main_hi}; {compare_country} covers {comp_lo}–{comp_hi}; approximate overlapping year span: {common} year(s). Missing years and different coverage can affect comparison.")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(valid_df.year, valid_df.value, marker="o", markersize=3, label=country)
            ax2.plot(comparison_df.year, comparison_df.value, marker="o", markersize=3, label=compare_country)
            ax2.set(title=f"{indicator_name}: country comparison", xlabel="Year", ylabel=f"{indicator_name} ({units})")
            ax2.legend(); ax2.grid(alpha=.25); fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
    # Exports
    st.subheader("Export current selection")
    status = "User-modified scenario" if st.session_state.get("data_edited") else "Source data"
    summary = pd.DataFrame([{"country":country,"country_code":COUNTRIES[country],"indicator":indicator_name,"indicator_code":code,"units":units,"definition":definition,"retrieved_at":datetime.now().astimezone().isoformat(),"source_status":status,"degree":degree,"base_year":base_year,"equation":equation,"in_sample_r_squared":r2,"in_sample_mae":mae,"observations":len(valid_df),"historical_start":int(valid_df.year.min()),"historical_end":int(valid_df.year.max()),"mean":float(valid_df.value.mean())}])
    csv_data = valid_df.to_csv(index=False).encode("utf-8")
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        valid_df.to_excel(writer, sheet_name="Working data", index=False)
        residual_table.to_excel(writer, sheet_name="Fitted values", index=False)
        if not batch_df.empty: batch_df.to_excel(writer, sheet_name="Calculated years", index=False)
        if not comparison_df.empty: comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
    pdf = create_pdf("Hispanic Heritage Historical Data Explorer",
        f"{country} • {indicator_name} • World Bank {code} • Retrieved {datetime.now().date()} • Status: {status}",
        equation, metrics_text,
        f"MAE summarizes average absolute in-sample residual. Polynomial fits can overfit and extrapolate poorly. No causal claims or validated prediction intervals are provided. {definition}",
        valid_df, batch_df, fig)
    d1,d2,d3 = st.columns(3)
    d1.download_button("Download working data (CSV)", csv_data, file_name="working_data.csv", mime="text/csv")
    d2.download_button("Download report workbook (Excel)", xlsx_buf.getvalue(), file_name="historical_data_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d3.download_button("Download report (PDF)", pdf, file_name="historical_data_report.pdf", mime="application/pdf")
    st.caption("CSV contains the current validated working table. Excel contains summary, working data, fitted values, and calculated-year results when available. PDF includes model metadata, metrics, chart, limitations, preview, and batch results.")
else:
    fig = None
    st.info("A model and exports become available after valid data with enough distinct years are available.")

st.divider()
st.header("Interactive Mathematics Lab")
st.write("Choose a course level and context, then compare a fitted historical model, a suggested contextual function, or a custom function. Context functions are learning scenarios, not claims about real outcomes.")
level = st.radio("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], horizontal=True)
context = st.selectbox("Context", list(CONTEXTS))
default_x, default_xu, default_y, default_yu, default_desc = CONTEXTS[context]
with st.expander("Customize context and variables", expanded=True):
    context_description = st.text_input("Context description", default_desc)
    x_name = st.text_input("Independent variable", default_x)
    x_units = st.text_input("Independent-variable units", default_xu)
    y_name = st.text_input("Dependent variable", default_y)
    y_units = st.text_input("Dependent-variable units", default_yu)
    xmin, xmax = st.slider("Analysis interval", -20.0, 100.0, (0.0, 10.0), 0.5)
    if xmin >= xmax: st.error("Analysis interval must have a smaller left endpoint than right endpoint.")
function_choice = st.radio("Function source", ["Suggested function", "Build from parameters", "Use historical polynomial fit", "Enter my own function"], horizontal=True)
a0,b0,c0,k0 = PARAMS["a"], PARAMS["b"], PARAMS["c"], PARAMS["k"]
if function_choice == "Suggested function":
    expr_text = SUGGESTED[context]
    for symbol_name, numeric_value in [("a", a0), ("b", b0), ("c", c0), ("k", k0)]:
        expr_text = re.sub(rf"\\b{symbol_name}\\b", f"({numeric_value})", expr_text)
    st.code(f"f(x) = {expr_text}")
    st.caption("Suggested functions use default parameter values a=1, b=0, c=0, k=0.2. Choose Build from parameters to customize them.")
elif function_choice == "Build from parameters":
    expr_text = SUGGESTED[context]
    pa,pb,pc,pk = st.columns(4)
    a0 = pa.number_input("a", value=1.0, step=0.5)
    b0 = pb.number_input("b", value=0.0, step=0.5)
    c0 = pc.number_input("c", value=0.0, step=0.5)
    k0 = pk.number_input("k", value=0.2, step=0.1)
    expr_text = expr_text.replace("a", f"({a0})").replace("b", f"({b0})").replace("c", f"({c0})").replace("k", f"({k0})")
    st.code(f"f(x) = {expr_text}")
elif function_choice == "Use historical polynomial fit":
    if model_ok:
        # Convert centered-year polynomial to a polynomial in x=year for analysis.
        xx = sp.symbols("x", real=True)
        expr_text = str(sum(float(cval) * (xx - base_year) ** (len(coef)-i-1) for i,cval in enumerate(coef)))
        st.caption(f"Using current historical model; x represents calendar year and t = x − {base_year}.")
    else:
        expr_text = "x**2"
        st.warning("Historical fit is unavailable; using x² as a fallback classroom example.")
else:
    expr_text = st.text_input("Function in x", "x**2 - 4*x + 3")
    st.caption("Supported notation: +, -, *, **, /, sin(x), cos(x), tan(x), exp(x), log(x), sqrt(x), pi. Use ** for powers.")

x_symbol, expression = None, None
try:
    x_symbol, expression = safe_symbolic(expr_text)
    f_expr = sp.simplify(expression)
    f1_expr = sp.diff(f_expr, x_symbol)
    f2_expr = sp.diff(f1_expr, x_symbol)
    st.latex(r"f(x) = " + sp.latex(f_expr))
    if level == "Calculus":
        st.latex(r"f'(x) = " + sp.latex(f1_expr))
        st.latex(r"f''(x) = " + sp.latex(f2_expr))
    elif level == "Precalculus":
        st.write("Precalculus focus: how degree and coefficients shape the graph, turning points, and concavity through changing average rates.")
    else:
        st.write("Algebra 2 focus: domain and range, end behavior, graph trends, and average rates of change.")
    # Sample avoiding invalid values and discontinuities; masked invalid values break plotted lines.
    grid = np.linspace(xmin, xmax, 800)
    f_fun = sp.lambdify(x_symbol, f_expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        yy = np.asarray(f_fun(grid))
        if np.isscalar(yy): yy = np.full_like(grid, float(yy))
        yy = np.real_if_close(yy)
        yy = np.asarray(yy, dtype=float)
        yy[~np.isfinite(yy)] = np.nan
        yy[np.abs(yy) > 1e8] = np.nan
    current_x = st.slider("Current input x", float(xmin), float(xmax), float((xmin+xmax)/2), 0.1)
    left, right = st.slider("Average-rate interval [a, b]", float(xmin), float(xmax), (float(xmin), float(xmax)), 0.1)
    show_f = st.checkbox("Show f(x)", True)
    show_f1 = st.checkbox("Show f′(x)", level == "Calculus")
    show_f2 = st.checkbox("Show f″(x)", False)
    fval = numeric_real(f_expr, x_symbol, current_x)
    f1val = numeric_real(f1_expr, x_symbol, current_x)
    f2val = numeric_real(f2_expr, x_symbol, current_x)
    left_val = numeric_real(f_expr, x_symbol, left)
    right_val = numeric_real(f_expr, x_symbol, right)
    avg_rate = (right_val-left_val)/(right-left) if right != left and np.isfinite(left_val) and np.isfinite(right_val) else np.nan
    figm, axm = plt.subplots(figsize=(10, 4.8))
    if show_f: axm.plot(grid, yy, label="f(x)", linewidth=2)
    for enabled, deriv_expr, label in [(show_f1, f1_expr, "f′(x)"), (show_f2, f2_expr, "f″(x)")]:
        if enabled:
            d_fun = sp.lambdify(x_symbol, deriv_expr, modules=["numpy"])
            with np.errstate(all="ignore"):
                vals = np.asarray(d_fun(grid))
                if np.isscalar(vals): vals = np.full_like(grid, float(vals))
                vals = np.asarray(np.real_if_close(vals), dtype=float)
                vals[~np.isfinite(vals)] = np.nan
                vals[np.abs(vals) > 1e8] = np.nan
            axm.plot(grid, vals, label=label)
    if show_f and np.isfinite(fval):
        axm.scatter([current_x],[fval], s=65, label="Current point")
        if np.isfinite(f1val):
            tangent_x = np.linspace(max(xmin,current_x-(xmax-xmin)*.15), min(xmax,current_x+(xmax-xmin)*.15), 60)
            axm.plot(tangent_x, fval+f1val*(tangent_x-current_x), "--", label="Tangent at current x")
        if np.isfinite(left_val) and np.isfinite(right_val) and left != right:
            sec_x = np.linspace(left,right,60)
            axm.plot(sec_x, left_val+avg_rate*(sec_x-left), ":", linewidth=2, label="Secant over [a,b]")
            axm.scatter([left,right],[left_val,right_val], s=30)
    axm.axhline(0,color="gray",linewidth=.6); axm.axvline(0,color="gray",linewidth=.6)
    axm.set(xlabel=f"{x_name} ({x_units})", ylabel=f"{y_name} ({y_units})", title=f"{context}: function analysis")
    axm.grid(alpha=.25); axm.legend(); figm.tight_layout()
    st.pyplot(figm, use_container_width=True)
    st.write(f"**Context:** {context_description}")
    st.write(f"**At x = {current_x:g} {x_units}:** f(x) = {fval:.6g} {y_units}" if np.isfinite(fval) else f"f(x) is undefined or non-real at x = {current_x:g}.")
    if level == "Calculus":
        st.write(f"**Instantaneous rate:** f′(x) = {f1val:.6g} {y_units}/{x_units}" if np.isfinite(f1val) else "The first derivative is undefined at this point.")
        st.write(f"**Second derivative:** f″(x) = {f2val:.6g} {y_units}/{x_units}²" if np.isfinite(f2val) else "The second derivative is undefined at this point.")
    if np.isfinite(avg_rate):
        st.write(f"**Average rate on [{left:g}, {right:g}]:** {avg_rate:.6g} {y_units}/{x_units}")
    else:
        st.write("Average rate is undefined for this interval or its endpoint values.")
    # Numerical derivative-based analysis in requested interval.
    if level == "Calculus" and xmax > xmin:
        st.subheader("Critical points, extrema, concavity, and domain")
        st.caption("The following is numerical exploration over the selected interval. Roots are approximate; singularities or closely spaced roots may require manual verification.")
        try:
            critical = sp.calculus.util.function_range(f1_expr, x_symbol, sp.Interval(float(xmin),float(xmax)))
            roots = sp.solveset(sp.Eq(f1_expr,0), x_symbol, domain=sp.Interval(float(xmin),float(xmax)))
            root_values = []
            if isinstance(roots, sp.FiniteSet):
                root_values = sorted([float(sp.N(r)) for r in roots if abs(float(sp.im(sp.N(r)))) < 1e-8])
            elif roots is sp.EmptySet:
                root_values = []
            else:
                # numerical sign changes to capture common non-polynomial roots
                d_fun = sp.lambdify(x_symbol, f1_expr, "numpy")
                dg = np.linspace(xmin,xmax,1200)
                with np.errstate(all="ignore"): dv = np.asarray(d_fun(dg), dtype=float)
                for j in range(len(dg)-1):
                    if np.isfinite(dv[j]) and np.isfinite(dv[j+1]) and dv[j]*dv[j+1] < 0:
                        root_values.append(float((dg[j]+dg[j+1])/2))
            root_values = sorted(set(round(r,6) for r in root_values if xmin <= r <= xmax))
            extrema_rows = []
            for r in root_values:
                delta = min(1e-3, (xmax-xmin)/10000)
                lv,rv = numeric_real(f1_expr,x_symbol,r-delta),numeric_real(f1_expr,x_symbol,r+delta)
                kind = "local maximum" if lv > 0 and rv < 0 else "local minimum" if lv < 0 and rv > 0 else "stationary point; no sign-change extremum confirmed"
                extrema_rows.append({"x":r,"f(x)":numeric_real(f_expr,x_symbol,r),"classification":kind})
            if extrema_rows: st.dataframe(pd.DataFrame(extrema_rows), use_container_width=True)
            else: st.write("No critical points with f′(x)=0 were found numerically in the interval.")
            # Inflection candidates from f'' roots; verify sign changes around candidate.
            roots2 = sp.solveset(sp.Eq(f2_expr,0), x_symbol, domain=sp.Interval(float(xmin),float(xmax)))
            inflect_candidates = []
            if isinstance(roots2, sp.FiniteSet):
                inflect_candidates = [float(sp.N(r)) for r in roots2 if abs(float(sp.im(sp.N(r)))) < 1e-8]
            else:
                f2_fun = sp.lambdify(x_symbol,f2_expr,"numpy")
                dg = np.linspace(xmin,xmax,1200)
                with np.errstate(all="ignore"): d2v = np.asarray(f2_fun(dg),dtype=float)
                for j in range(len(dg)-1):
                    if np.isfinite(d2v[j]) and np.isfinite(d2v[j+1]) and d2v[j]*d2v[j+1] < 0:
                        inflect_candidates.append(float((dg[j]+dg[j+1])/2))
            inflect_rows=[]
            for r in sorted(set(round(v,6) for v in inflect_candidates if xmin < v < xmax)):
                delta=min(1e-3,(xmax-xmin)/10000)
                lv,rv=numeric_real(f2_expr,x_symbol,r-delta),numeric_real(f2_expr,x_symbol,r+delta)
                if np.isfinite(lv) and np.isfinite(rv) and lv*rv < 0:
                    inflect_rows.append({"x":r,"f(x)":numeric_real(f_expr,x_symbol,r),"concavity changes":True})
            st.write("**Verified inflection candidates** (second derivative changes sign):")
            st.dataframe(pd.DataFrame(inflect_rows) if inflect_rows else pd.DataFrame(columns=["x","f(x)","concavity changes"]), use_container_width=True)
            if sp.denom(sp.together(f_expr)) != 1:
                st.write(f"**Potential denominator restrictions:** denominator = `{sp.denom(sp.together(f_expr))}`; values that make it zero are excluded from the domain.")
            if f_expr.has(sp.log):
                st.write("**Domain note:** logarithms require positive arguments.")
            if f_expr.has(sp.tan):
                st.write("**Domain note:** tangent is undefined where its cosine denominator is zero.")
            if f_expr.has(sp.sqrt):
                st.write("**Domain note:** square roots of real-valued expressions require nonnegative radicands.")
            # Absolute extrema over closed interval: endpoints and verified critical points.
            candidates=[float(xmin),float(xmax)] + root_values
            vals=[numeric_real(f_expr,x_symbol,v) for v in candidates]
            finite=[(v,y) for v,y in zip(candidates,vals) if np.isfinite(y)]
            if finite:
                mn=min(finite,key=lambda p:p[1]); mx=max(finite,key=lambda p:p[1])
                st.write(f"**Absolute minimum on sampled closed interval:** f({mn[0]:.6g}) ≈ {mn[1]:.6g}")
                st.write(f"**Absolute maximum on sampled closed interval:** f({mx[0]:.6g}) ≈ {mx[1]:.6g}")
        except Exception as exc:
            st.warning(f"Some numerical calculus analysis could not be completed: {exc}")
    if context == "Engineering":
        st.info("If f represents position, f′ can represent velocity and f″ acceleration, provided the variables and units are position and time.")
    elif context in ("Business","Economic forecasting"):
        st.info("The function value represents the selected business/economic measure; the first derivative describes its rate of change, and the second derivative describes whether that rate is increasing or decreasing.")
    elif context == "Medical":
        st.info("The function is an educational concentration scenario, not a dosing recommendation or clinical prediction.")
    # Basic visual simulation: normalized bar responds to the current function value.
    st.subheader("Contextual visual simulation")
    simulation = st.selectbox("Visual type", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank or drug concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
    finite_y = yy[np.isfinite(yy)]
    if np.isfinite(fval):
        if len(finite_y) and np.max(finite_y) != np.min(finite_y):
            normalized = float(np.clip((fval-np.min(finite_y))/(np.max(finite_y)-np.min(finite_y)),0,1))
        else: normalized = 0.5
        st.progress(normalized, text=f"Illustrative normalized level: {normalized:.0%} of the visible graph's y-range (not a physical scale).")
        if simulation == "Walking person with prosthetic leg":
            st.write("🚶 🦿  Movement indicator")
        elif simulation == "Moving object / projectile":
            st.write("🏀  Moving-object indicator")
        elif simulation == "Tank or drug concentration":
            st.write("🧪  Concentration / tank-level indicator")
        elif simulation == "Population / cell colony":
            st.write("🦠  Population indicator")
        elif simulation == "Thermometer / weather":
            st.write("🌡️  Temperature indicator")
        else:
            st.write("📈  Business/economic indicator")
        st.caption("The moving graph point and displayed rates update with the current input. The icon is a contextual illustration; the progress level is normalized to the displayed graph range, not a real-world measurement.")
    else:
        st.warning("The simulation is unavailable at an undefined or non-real function value.")
except Exception as exc:
    st.error(str(exc))

st.divider()
st.subheader("How the app works")
with st.expander("Data flow, mathematics, assumptions, and limitations"):
    st.markdown("""
    1. **Retrieve:** the app calls the official World Bank Indicators API for the selected country, indicator, and 70-year request window. API errors and null values are reported rather than replaced with invented data.
    2. **Validate and edit:** the original response is kept separately from a year/value working table. The table checks whole-number unique years, numeric values, and basic indicator-specific constraints. Edited data are labeled as a user-modified scenario.
    3. **Fit:** a degree-3-to-8 polynomial is fit with centered time, `t = year − base year`, using NumPy. At least degree + 1 distinct observations are required.
    4. **Assess:** R² and mean absolute error (MAE) are in-sample metrics. A chronological holdout MAE is shown only when enough observations remain for a training fit and a test set. Neither guarantees future accuracy.
    5. **Analyze:** the Mathematics Lab uses SymPy for symbolic derivatives and numerical evaluations. Numerical root finding can miss roots or struggle near singularities; critical points and inflection candidates must be checked, not blindly classified.
    6. **Export:** CSV saves the working data; Excel includes summary and data/model result sheets; PDF includes a report, chart, findings, and preview.

    **Limitations:** World Bank coverage varies by indicator and country. Missing years are not imputed. A polynomial can overfit and often extrapolates poorly; extreme estimates are flagged rather than clipped. The app does not claim causation, invent historical causes, or provide validated 95% prediction intervals. The context functions and visual indicators are classroom illustrations, not professional financial, medical, engineering, or weather advice.
    """)
st.caption("Source: World Bank Indicators API. App-generated report date is based on the system clock.")
