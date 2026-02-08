# Brent Oil Change Point Analysis - Task 1

## Overview

This is **Task 1** of the Brent Oil Change Point Analysis project.  
The goal of this task is to **lay the foundation for analyzing Brent oil prices** by defining the workflow, preparing key event data, and documenting assumptions and limitations.

---

## Contents

The Task 1 folder contains:

| File Name                  | Description |
|-----------------------------|-------------|
| `data/events.csv`           | Structured dataset of 13 major geopolitical, economic, and OPEC events affecting Brent oil prices |
| `Task1_Brent_Oil_Analysis_Foundation.pdf` | 1–2 page document outlining the data analysis workflow, assumptions, limitations, and model understanding |

---

## Analysis Workflow

The planned workflow for Brent oil price analysis is:

1. **Load Data**: Import Brent oil price data and check for missing values.  
2. **Exploratory Data Analysis (EDA)**: Visualize trends, volatility, and price shocks.  
3. **Transform Data**: Calculate log returns for stability and stationarity.  
4. **Add Event Data**: Integrate `events.csv` to link major events with price changes.  
5. **Change Point Modeling**: Apply Bayesian change point detection to identify structural breaks in the time series.  
6. **Insight Generation**: Associate detected change points with key events and quantify their impact.

---

## Assumptions and Limitations

**Assumptions:**
- Major geopolitical, economic, and OPEC events can influence oil prices.  
- Change point models can detect significant shifts in the data.  

**Limitations:**
- Correlation does not imply causation.  
- Events happening close together may be difficult to separate.  
- Other factors like GDP, inflation, or exchange rates are not included in this analysis.  

---

## Communication Channels

Results and insights from Task 1 are intended to be shared via:

- PDF or blog post for stakeholders  
- Interactive dashboards in later tasks  
- Visual charts for quick trend analysis  

---

## References

- [Data Science Workflow](https://www.datascience-pm.com/data-science-workflow/)  
- [Change Point Analysis](https://forecastegy.com/posts/change-point-detection-time-series-python/)  
- [Bayesian Change Point Detection with PyMC3](https://www.pymc.io/blog/chris_F_pydata2022.html)  

---

## Next Steps

- Proceed to **Task 2**: Load Brent oil price data, perform exploratory analysis, and prepare it for Bayesian change point modeling.

