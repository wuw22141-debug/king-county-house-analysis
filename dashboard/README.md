# Dashboard

This folder contains the interactive dashboard for the King County house-sales project.

## Files

- `king_county_house_sales_dashboard.html` — finished dashboard; open this file directly in a browser.
- `build_king_county_dashboard.py` — loads the OpenML data, prepares features, compares regression models, and creates the dashboard.
- `king_county_dashboard_template.html` — HTML/CSS/JavaScript template used by the build script.

## Rebuild

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run:

```bash
cd dashboard
python build_king_county_dashboard.py
```

By default, the script loads OpenML dataset `42092`. It can also accept a local CSV through the `--csv` option.
