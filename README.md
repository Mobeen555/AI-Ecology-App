# EcoScope Research Agent (MVP)
Reservoir water-quality analytics: field data + satellite-derived data -> cleaning, date/site matching,
validation statistics, spatial/temporal patterns, N vs P drivers, bloom risk, auditable trail,
reproducible export (data, tables, figures, R code, methods draft). Groq is used only for planning and
interpretation, using compact statistics (low token).

## Run locally
    pip install -r requirements.txt
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your Groq key
    streamlit run app.py

## Deploy (GitHub -> Streamlit Community Cloud)
1. Create a free key at https://console.groq.com/keys
2. Create a new GitHub repo (e.g. `ecoscope-research-agent`), upload ALL files in this folder
   (app.py, requirements.txt, README.md, .gitignore, .streamlit/config.toml, .streamlit/secrets.toml.example).
   Do NOT upload secrets.toml.
3. Go to https://share.streamlit.io -> sign in with GitHub -> "Create app" -> "Deploy a public app from GitHub".
4. Pick the repo, branch `main`, main file path `app.py`.
5. Open "Advanced settings" -> Secrets, paste:  GROQ_API_KEY = "gsk_xxx"
6. Click Deploy. First build takes 2-4 minutes. Every `git push` redeploys automatically.

## Input format
Field file (CSV/XLSX): `date, site, lat, lon, chl_a, turbidity, temp, TP, TN` (extra columns allowed, e.g. phytoplankton counts).
Satellite file (CSV): `date, site, sat_chl_a, sat_turbidity, sat_temp, cloud_pct` (+ optional product metadata set in sidebar).
Use the built-in demo data to test first.

## Roadmap (Phase 2)
Sentinel-2 retrieval via Microsoft Planetary Computer STAC, map drawing of reservoir/points (folium/leafmap),
GAM (pygam), RDA (scikit-bio), phytoplankton community module, WASP/AQUATOX/CE-QUAL-W2 input exporters.
