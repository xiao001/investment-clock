# Investment Clock

Growth direction x inflation direction gives four phases (Recovery, Overheat, Stagflation, Reflation),
following Greetham (Merrill Lynch, 2004). Signals use a 2-month publication lag and real monthly returns
(1973-04 to 2026-08). Growth is a 2-of-3 vote of industrial production, the Chicago Fed National Activity
Index and unemployment; inflation is the direction of smoothed CPI YoY.

## Run the app
```
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Update the data
```
cd code
python investment_clock.py
```
Then rerun the app. Data comes from FRED, Yahoo Finance and Ken French's library, revised values only
(no real-time vintages). Educational research, not investment advice.

## Layout
- `app/` Streamlit app
- `code/` model script and notebook
- `results/model_outputs/` CSVs the app reads
