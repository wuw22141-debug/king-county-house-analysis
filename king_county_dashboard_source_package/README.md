# King County House Sales Dashboard

This folder contains the source files for the interactive King County house-sales dashboard.

## Files

- `king_county_house_sales_dashboard.html` — finished interactive dashboard; open it directly in a browser.
- `build_king_county_dashboard.py` — Python script that downloads/loads the data, prepares features, trains regression models, and builds the dashboard.
- `king_county_dashboard_template.html` — HTML/CSS/JavaScript template used by the build script.
- `requirements.txt` — Python dependencies required to rebuild the dashboard.

## Rebuild

```bash
pip install -r requirements.txt
python build_king_county_dashboard.py
```

The script uses the King County House Sales dataset from OpenML (`data_id=42092`) unless a local CSV is supplied.
