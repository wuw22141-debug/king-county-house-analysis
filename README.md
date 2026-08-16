# King County House Sales Analysis

This project explores residential house sales in King County, Washington, using Python. The work is organized as a progression from pandas practice, to visualization, to a more complete exploratory analysis. An interactive HTML dashboard is also included.

## Dataset

The project uses the **King County House Sales** dataset from OpenML (`data_id=42092`). It contains **21,613 observations and 21 original columns**, including sale price, bedrooms, bathrooms, living area, lot size, grade, condition, waterfront status, ZIP code, latitude, longitude, and construction information.

The raw dataset is not stored in this repository. The notebooks and dashboard build script load it from OpenML.

## Questions Explored

The analysis looks at questions such as:

- How is sale price related to living area and building grade?
- How do waterfront status and view relate to price?
- How do prices differ across ZIP codes and geographic locations?
- How do house prices and sales change over time?
- Which housing features are useful for predicting sale price?

## Repository Structure

```text
king-county-house-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_pandas_basics.ipynb
│   ├── 02_data_visualization.ipynb
│   └── 03_exploratory_analysis.ipynb
└── dashboard/
    ├── README.md
    ├── build_king_county_dashboard.py
    ├── king_county_dashboard_template.html
    └── king_county_house_sales_dashboard.html
```

## Notebooks

### `01_pandas_basics.ipynb`

Practice with pandas operations used later in the project, including:

- inspecting DataFrames
- selecting and filtering rows and columns
- creating and renaming columns
- sorting and summary statistics
- date conversion
- `groupby` operations
- duplicate checks and basic data cleaning

### `02_data_visualization.ipynb`

Focuses on preparing variables for plotting and exploring the data visually with pandas and Matplotlib. It includes several common plot types such as histograms, box plots, pie charts, and line plots.

### `03_exploratory_analysis.ipynb`

The main analysis notebook. It includes data preparation, feature engineering, descriptive statistics, and visual exploration of relationships between house prices and characteristics such as living area, grade, view, waterfront status, time, ZIP code, and geographic location.

## Interactive Dashboard

The `dashboard/` folder contains a self-contained interactive HTML dashboard.

To view the finished dashboard, open:

```text
dashboard/king_county_house_sales_dashboard.html
```

The dashboard source also includes a Python build script and an HTML template so the dashboard can be rebuilt from the data.

## Running the Project

Install the Python packages:

```bash
pip install -r requirements.txt
```

Then open the notebooks with Jupyter or upload them to Google Colab.

To rebuild the dashboard:

```bash
cd dashboard
python build_king_county_dashboard.py
```

## Libraries Used

- pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn

## Notes

The notebooks show the learning and analysis process, so some sections are intentionally step-by-step. The main project analysis is in `03_exploratory_analysis.ipynb`, while the dashboard provides a more compact interactive view of the results.
