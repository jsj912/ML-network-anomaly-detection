# dashboard.py
import os
import time
import json
import base64
from pathlib import Path

import pandas as pd
import numpy as np

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio

# ---------------- USER FILE PATHS ----------------
PACKETS_CSV = "all_packets.csv"
SCORES_CSV = "two_stage_scores.csv"
GBM_DIR = Path("shap_gbm")
IF_DIR = Path("if_explain")
TOP_N_CACHE = "top_scores_cache.parquet"   # cached top-n parquet
EDA_OUT = Path("eda_outputs")
EXPLANATION_OUTDIR = Path("./explanations")
EXPLANATION_OUTDIR.mkdir(exist_ok=True)

# ---------------- THEME & STYLES ----------------
COLORS = {
    'background': '#0a1929',
    'card': '#132f4c',
    'accent': '#1e88e5',
    'text': '#e0e0e0',
    'muted': '#9e9e9e',
    'border': '#1e4976',
}

pio.templates["custom_dark"] = pio.templates["plotly_dark"]
pio.templates["custom_dark"].update({
    'layout': {
        'font': {'family': 'Inter, sans-serif'},
        'plot_bgcolor': COLORS['card'],
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'colorway': [COLORS['accent'], '#f50057', '#00b0ff', '#00bfa5', '#ff9100'],
    }
})

HEADER_STYLE = {
    'textAlign': 'left',
    'color': COLORS['text'],
    'fontFamily': '"Inter", sans-serif',
    'fontWeight': '600',
    'marginTop': '20px',
    'marginBottom': '10px',
}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'borderRadius': '8px',
    'padding': '16px',
    'marginBottom': '20px',
    'border': f'1px solid {COLORS["border"]}',
}

# ---------------- SAFE SMALL PACKETS LOAD (lightweight) ----------------
if not os.path.exists(PACKETS_CSV):
    packets_df = pd.DataFrame(columns=['Protocol', 'Source_IP', 'Destination_IP', 'Length'])
else:
    # load a lightweight slice for overview responsiveness
    try:
        packets_df = pd.read_csv(PACKETS_CSV, nrows=150000, low_memory=False)
    except Exception:
        packets_df = pd.read_csv(PACKETS_CSV, nrows=20000, low_memory=False)

if 'Length' in packets_df.columns:
    packets_df['Length'] = pd.to_numeric(packets_df['Length'], errors='coerce').fillna(0)
packets_df.fillna({'Protocol': 'Unknown', 'Source_IP': 'Unknown', 'Destination_IP': 'Unknown', 'Info': 'No info'}, inplace=True)
str_cols = packets_df.select_dtypes(include=['object']).columns
packets_df[str_cols] = packets_df[str_cols].astype(str)

# ---------------- EDA summary precompute (lightweight) ----------------
summary_stats = {}
for candidate in ("combined_features.csv", "combined_cic_ctu.csv"):
    if Path(candidate).exists():
        try:
            tmp = pd.read_csv(candidate, nrows=200000, low_memory=False)
            if 'label' in tmp.columns:
                tmp['label_bin'] = tmp['label'].astype(str).str.upper().apply(lambda x: 0 if 'BENIGN' in x or 'NORMAL' in x else 1)
            elif 'label_bin' not in tmp.columns:
                tmp['label_bin'] = 0

            def safe_stats(col):
                if col in tmp.columns:
                    s = pd.to_numeric(tmp[col], errors='coerce').dropna()
                    if len(s) == 0:
                        return {"mean": None, "median": None}
                    return {"mean": float(s.mean()), "median": float(s.median())}
                return {"mean": None, "median": None}

            summary_stats = {
                "duration": safe_stats('duration'),
                "tot_pkts": safe_stats('tot_pkts') if 'tot_pkts' in tmp.columns else safe_stats('totpkts'),
                "tot_bytes": safe_stats('tot_bytes') if 'tot_bytes' in tmp.columns else safe_stats('totbytes'),
                "fwd_bytes": safe_stats('fwd_bytes') if 'fwd_bytes' in tmp.columns else safe_stats('srcbytes'),
                "flow_iat_mean": safe_stats('flow_iat_mean'),
            }
            break
        except Exception:
            summary_stats = {}
            continue

# ---------------- Helper functions ----------------
def embed_image_b64(path):
    """Return base64 data URL for a PNG or empty string if not found."""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return "data:image/png;base64," + encoded
    except Exception:
        return ""

def compute_top_n_from_scores(n=200, chunksize=250_000):
    """
    Scan SCORES_CSV in chunks robustly, keep top-n rows by 'rerank_score'.
    Cache to TOP_N_CACHE parquet to speed re-use.
    """
    if not os.path.exists(SCORES_CSV):
        return pd.DataFrame()

    try:
        if os.path.exists(TOP_N_CACHE) and os.path.getmtime(TOP_N_CACHE) >= os.path.getmtime(SCORES_CSV):
            cached = pd.read_parquet(TOP_N_CACHE)
            return cached.head(n).reset_index(drop=True)
    except Exception:
        pass

    top_df = None
    try:
        reader = pd.read_csv(SCORES_CSV, chunksize=chunksize, engine='python', on_bad_lines='skip')
    except Exception:
        reader = pd.read_csv(SCORES_CSV, chunksize=chunksize, low_memory=False)

    for chunk in reader:
        if 'rerank_score' not in chunk.columns:
            return chunk.head(n).reset_index(drop=True)
        chunk['rerank_score'] = pd.to_numeric(chunk['rerank_score'], errors='coerce').fillna(0)
        chunk = chunk.dropna(subset=['rerank_score'])
        if top_df is None:
            top_df = chunk.nlargest(n, 'rerank_score')
        else:
            top_df = pd.concat([top_df, chunk], ignore_index=True).nlargest(n, 'rerank_score')

    if top_df is None:
        return pd.DataFrame()

    top_df = top_df.sort_values('rerank_score', ascending=False).reset_index(drop=True)
    if 'top_index' not in top_df.columns:
        top_df['top_index'] = top_df.index
    try:
        top_df.to_parquet(TOP_N_CACHE, index=False)
    except Exception:
        pass
    return top_df.head(n).reset_index(drop=True)

def load_gbm_explanations():
    json_path = GBM_DIR / "gbm_shap_topN_explanations.json"
    if not json_path.exists():
        return pd.DataFrame()
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame()
    rows = []
    for item in data:
        rows.append({
            "alert_index": int(item.get("top_index", item.get("alert_index", -1))),
            "rerank_score": float(item.get("rerank_score", 0)),
            "label_bin": int(item.get("label_bin", 0)),
            "top_contributors": item.get("top_contributors", [])
        })
    return pd.DataFrame(rows)

def load_if_aggregate():
    agg = IF_DIR / "if_aggregate_topN.csv"
    if not agg.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(agg)
    except Exception:
        return pd.DataFrame()

def load_if_ablation():
    path = IF_DIR / "if_ablation_topN.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    mapping = {}
    if isinstance(data, dict):
        for k, v in data.items():
            mapping[str(k)] = v
        return mapping
    for item in data:
        idx = int(item.get("top_index", -1))
        mapping[str(idx)] = item
    return mapping

def load_if_zscores():
    path = IF_DIR / "if_zscores_topN.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    mapping = {}
    if isinstance(data, dict):
        for k, v in data.items():
            mapping[str(k)] = v
        return mapping
    for item in data:
        idx = int(item.get("top_index", -1))
        mapping[str(idx)] = item
    return mapping

def load_if_lof_df():
    path = IF_DIR / "if_lof_topN.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'top_index' not in df.columns and 'index' in df.columns:
            df = df.rename(columns={'index': 'top_index'})
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- App init ----------------
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = "Network Traffic Analysis"

# ---------------- Layout ----------------
app.layout = dbc.Container(
    style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '18px', 'color': COLORS['text']},
    children=[
        dbc.Row([
            dbc.Col([
                html.H1("Network Traffic Analysis", style={**HEADER_STYLE, 'fontSize': '26px'}),
                html.P("Packet-level overview + Explainability (GBM SHAP + IF explain).", style={'color': COLORS['muted']})
            ])
        ]),
        dcc.Tabs(id='tabs', value='tab-overview', children=[
            dcc.Tab(label='Overview', value='tab-overview'),
            dcc.Tab(label='Explainability', value='tab-explain'),
            dcc.Tab(label='EDA', value='tab-eda'),
        ]),
        html.Div(id='tab-content', style={'marginTop': '12px'})
    ]
)

# ---------------- Tab render callback ----------------
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-overview':
        return overview_layout()
    elif tab == 'tab-explain':
        return explainability_layout_placeholder()
    elif tab == 'tab-eda':
        return eda_layout()
    else:
        return html.Div("Unknown tab")

# ---------------- Overview layout ----------------
def overview_layout():
    total_packets = len(packets_df) if not packets_df.empty else 0
    unique_protocols = packets_df['Protocol'].nunique() if 'Protocol' in packets_df.columns else 0

    left_card = dbc.Card([
        dbc.CardBody([
            html.H5("Visualization Settings"),
            html.Label("Chart Type"),
            dcc.Dropdown(
                id='chart-type',
                options=[
                    {'label': 'Protocol Distribution', 'value': 'protocol'},
                    {'label': 'Top Source IPs', 'value': 'src'},
                    {'label': 'Top Destination IPs', 'value': 'dst'},
                    {'label': 'Packet Size Analysis', 'value': 'length'}
                ],
                value='protocol',
                clearable=False
            ),
            html.Hr(),
            html.Div([
                html.Div(["Total Packets"], style={'color': COLORS['muted']}),
                html.Div(f"{total_packets:,}", style={'color': COLORS['accent'], 'fontSize': '20px', 'fontWeight': '600'}),
                html.Br(),
                html.Div(["Unique Protocols"], style={'color': COLORS['muted']}),
                html.Div(f"{unique_protocols}", style={'color': '#00bfa5', 'fontSize': '18px', 'fontWeight': '600'})
            ])
        ])
    ], style=CARD_STYLE)

    right_card = dbc.Card([
        dbc.CardBody([
            dcc.Loading(dcc.Graph(id='main-graph', style={'height': '650px'}), type='circle')
        ])
    ], style={**CARD_STYLE})

    return dbc.Row([dbc.Col(left_card, md=3), dbc.Col(right_card, md=9)])

# ---------------- Explainability layout placeholder ----------------
def explainability_layout_placeholder():
    return dbc.Container([
        html.H3("Explainability"),
        html.P("Click 'Load Explainability Artifacts' to compute or load cached artifacts (fast)."),
        dbc.Button("Load Explainability Artifacts", id='load-explain-btn', color='primary'),
        html.Div(id='explain-area', style={'marginTop': '16px'})
    ])

# ---------------- EDA Layout ----------------
def eda_layout():
    OUT = EDA_OUT
    duration_png = OUT / "duration_benign_vs_malicious.png"
    pkt_png = OUT / "pktcount_benign_vs_malicious.png"
    bytes_png = OUT / "bytes_benign_vs_malicious.png"
    iat_png = OUT / "iat_benign_vs_malicious.png"
    proto_png = OUT / "protocol_counts.png"
    corr_png = OUT / "correlation_heatmap.png"
    anomaly_png = OUT / "anomaly_score_distribution.png"
    timeline_html = OUT / "timeline_traffic.html"

    def stat_line(label, stat):
        if not stat or stat.get("mean") is None:
            return html.Div([html.Span(label + ": ", style={'color': COLORS['muted']}), html.Span("N/A", style={'color': COLORS['text']})])
        mean = f"{stat['mean']:.3g}"
        median = f"{stat['median']:.3g}"
        return html.Div([html.Span(label + ": ", style={'color': COLORS['muted'], 'width':'160px', 'display':'inline-block'}),
                         html.Span(f"mean={mean}  median={median}", style={'color': COLORS['text']})])

    summary_card = dbc.Card([
        dbc.CardBody([
            html.H5("EDA Summary (sample)"),
            stat_line("Duration (s)", summary_stats.get("duration") if summary_stats else None),
            stat_line("Total pkts", summary_stats.get("tot_pkts") if summary_stats else None),
            stat_line("Total bytes", summary_stats.get("tot_bytes") if summary_stats else None),
            stat_line("Fwd bytes", summary_stats.get("fwd_bytes") if summary_stats else None),
            stat_line("Flow IAT mean", summary_stats.get("flow_iat_mean") if summary_stats else None),
            html.Div("Note: summary computed from a small sample file for responsiveness.", style={'color':COLORS['muted'], 'marginTop':'8px', 'fontSize':'12px'})
        ])
    ], style={**CARD_STYLE, 'marginBottom':'18px'})

    left = dbc.Col([
        html.H4("Feature distributions"),
        html.Div([
            html.Img(src=embed_image_b64(duration_png) if duration_png.exists() else "", style={'width':'100%','marginBottom':'12px'}) if duration_png.exists() else html.Div("Duration plot missing", style={'color':'#f1c40f'}),
            html.Img(src=embed_image_b64(pkt_png) if pkt_png.exists() else "", style={'width':'100%','marginBottom':'12px'}) if pkt_png.exists() else html.Div("Packet count plot missing", style={'color':'#f1c40f'}),
            html.Img(src=embed_image_b64(bytes_png) if bytes_png.exists() else "", style={'width':'100%','marginBottom':'12px'}) if bytes_png.exists() else html.Div("Bytes plot missing", style={'color':'#f1c40f'}),
            html.Img(src=embed_image_b64(iat_png) if iat_png.exists() else "", style={'width':'100%','marginBottom':'12px'}) if iat_png.exists() else html.Div("IAT plot missing", style={'color':'#f1c40f'}),
        ], style={'maxHeight':'62vh','overflowY':'auto'})
    ], md=6)

    right = dbc.Col([
        html.H4("Correlations & Scores"),
        html.Img(src=embed_image_b64(corr_png) if corr_png.exists() else "", style={'width':'100%','marginBottom':'12px'}) if corr_png.exists() else html.Div("Correlation heatmap missing", style={'color':'#f1c40f'}),
        html.Img(src=embed_image_b64(anomaly_png) if anomaly_png.exists() else "", style={'width':'100%','marginBottom':'12px'}) if anomaly_png.exists() else html.Div("Anomaly score plot missing", style={'color':'#f1c40f'}),
        html.Div([
            html.H6("Interactive timeline"),
            html.Div([html.A("Open timeline (interactive)", href=str(timeline_html) if timeline_html.exists() else "#", target="_blank")])
        ], style={'marginTop':'12px'})
    ], md=6)

    return dbc.Container([
        summary_card,
        dbc.Row([left, right])
    ], fluid=True)

# ---------------- Explainability loader ----------------
@app.callback(Output('explain-area', 'children'),
              Input('load-explain-btn', 'n_clicks'),
              prevent_initial_call=True)
def load_explain(n_clicks):
    start = time.time()
    top_df = compute_top_n_from_scores(n=200)
    if top_df.empty:
        return html.Div("No two_stage_scores.csv found or it could not be read.", style={'color': 'red'})

    gbm_df = load_gbm_explanations()
    if_df = load_if_aggregate()
    if_ablation = load_if_ablation()
    if_zscores = load_if_zscores()
    if_lof = load_if_lof_df()

    if 'top_index' in top_df.columns:
        top_df = top_df.rename(columns={'top_index': 'alert_index'})

    merged = top_df[['alert_index', 'rerank_score']].copy()
    if not gbm_df.empty:
        merged = merged.merge(gbm_df[['alert_index','label_bin','top_contributors']], on='alert_index', how='left')
    if not if_df.empty:
        merged = merged.merge(if_df[['top_index','if_score','lof_score']], left_on='alert_index', right_on='top_index', how='left')
        merged.drop(columns=['top_index'], inplace=True, errors='ignore')

    display_cols = [c for c in ['alert_index','rerank_score','if_score','lof_score','label_bin'] if c in merged.columns]
    table = dash_table.DataTable(
        id='explain-table',
        columns=[{"name": c, "id": c} for c in display_cols],
        data=merged[display_cols].to_dict('records'),
        page_size=12,
        row_selectable='single',
        style_table={'height': '380px', 'overflowY': 'auto'},
        style_header={'backgroundColor': '#222'},
        style_cell={'color': COLORS['text']}
    )

    right_col = dbc.Col([
        html.H5("GBM Global SHAP Summary"),
        html.Img(src=embed_image_b64(GBM_DIR / "global_shap_summary.png"), style={'width':'100%','border':'1px solid #333'}),
        html.Br(), html.Br(),
        html.H5("Per-alert explanation"),
        html.Img(id='per-alert-gbm-shap', src='', style={'width':'100%','border':'1px solid #333'}),
        html.Br(),
        html.H6("IF Ablation (top contributors)"),
        dash_table.DataTable(id='if-ablation', columns=[{"name":"feature","id":"feature"},{"name":"impact","id":"impact"}],
                             data=[], page_size=6, style_cell={'color':COLORS['text']}),
        html.Br(),
        html.H6("IF LOF Summary"),
        dash_table.DataTable(id='if-lof-table', columns=[{"name":"metric","id":"metric"},{"name":"value","id":"value"}],
                             data=[{}], page_size=6, style_cell={'color':COLORS['text']}),
        html.Br(),
        html.H6("IF Z-scores (top features)"),
        dash_table.DataTable(id='if-zscore-table', columns=[{"name":"feature","id":"feature"},{"name":"z","id":"z"}],
                             data=[], page_size=6, style_cell={'color':COLORS['text']}),
        html.Br(),
        dbc.Button("Download Explanation", id="download-explain-btn", color="secondary", className="mt-2"),
        dcc.Download(id="download-explain")
    ], md=6)

    left_col = dbc.Col([table], md=6)
    footer = html.Div(f"Loaded explainability artifacts in {time.time()-start:.1f}s", style={'color':COLORS['muted'],'marginTop':'8px'})

    return dbc.Row([left_col, right_col, html.Div(footer, style={'marginTop':'8px'})])

# ---------------- main graph callback ----------------
@app.callback(Output('main-graph', 'figure'), Input('chart-type', 'value'))
def update_main_graph(chart_type):
    if packets_df is None or packets_df.empty:
        return px.histogram(pd.DataFrame({'x':[]}), x=[])
    if chart_type == 'protocol':
        data = packets_df['Protocol'].value_counts().reset_index()
        data.columns = ['Protocol','Count']
        fig = px.bar(data, x='Protocol', y='Count', color='Protocol')
    elif chart_type == 'src':
        data = packets_df['Source_IP'].value_counts().head(10).reset_index()
        data.columns = ['Source_IP','Count']
        fig = px.bar(data, x='Count', y='Source_IP', orientation='h')
    elif chart_type == 'dst':
        data = packets_df['Destination_IP'].value_counts().head(10).reset_index()
        data.columns = ['Destination_IP','Count']
        fig = px.pie(data, values='Count', names='Destination_IP', hole=0.4)
    else:
        if 'Length' in packets_df.columns:
            fig = px.histogram(packets_df, x='Length', nbins=40)
        else:
            fig = px.histogram(pd.DataFrame({'Length':[]}), x='Length')

    fig.update_layout(template='custom_dark', plot_bgcolor=COLORS['card'], paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ---------------- selection callback (produces structured outputs) ----------------
@app.callback(
    Output('per-alert-gbm-shap', 'src'),
    Output('if-ablation', 'data'),
    Output('if-lof-table', 'data'),
    Output('if-zscore-table', 'data'),
    Input('explain-table', 'selected_rows'),
    State('explain-table', 'data'),
    prevent_initial_call=True
)
def on_alert_selected(selected_rows, table_data):
    if not selected_rows or not table_data:
        return "", [], [], []

    try:
        sel = selected_rows[0]
        row = table_data[sel]
    except Exception:
        return "", [], [], []

    alert_index = row.get('alert_index', None)
    if alert_index is None:
        return "", [], [], []

    gbm_png = GBM_DIR / f"gbm_shap_alert_{alert_index}.png"
    gbm_src = embed_image_b64(gbm_png) or ""

    ablation_map = load_if_ablation()
    ablation_entry = ablation_map.get(str(alert_index), None)
    ablation_data = []
    if ablation_entry:
        deltas = []
        if isinstance(ablation_entry, dict):
            deltas = ablation_entry.get('deltas') or ablation_entry.get('delta') or []
            if not isinstance(deltas, (list,tuple)):
                for k,v in ablation_entry.items():
                    if k in ('top_index','score','meta'):
                        continue
                    if isinstance(v, (int,float,str)):
                        deltas.append({"feature":k,"delta":v})
        elif isinstance(ablation_entry, list):
            deltas = ablation_entry

        for d in deltas[:10]:
            if isinstance(d, dict):
                feat = d.get('feature') or d.get('name') or next(iter(d.keys()), None)
                val = d.get('delta') if 'delta' in d else (d.get('impact') if 'impact' in d else None)
                ablation_data.append({"feature": feat, "impact": val})
            elif isinstance(d, (list,tuple)) and len(d) >= 2:
                ablation_data.append({"feature": d[0], "impact": d[1]})
            else:
                ablation_data.append({"feature": str(d), "impact": ""})

    lof_df = load_if_lof_df()
    lof_row = {"metric": None, "value": None}
    if not lof_df.empty:
        if 'top_index' in lof_df.columns:
            match = lof_df[lof_df['top_index'] == int(alert_index)]
            if not match.empty:
                if 'lof_score' in match.columns:
                    lof_row = {"metric": "lof_score", "value": float(match.iloc[0]['lof_score'])}
                elif 'lof' in match.columns:
                    lof_row = {"metric": "lof", "value": float(match.iloc[0]['lof'])}
                else:
                    for c in match.columns:
                        if c == 'top_index':
                            continue
                        try:
                            val = float(match.iloc[0][c])
                            lof_row = {"metric": c, "value": val}
                            break
                        except Exception:
                            continue

    zmap = load_if_zscores()
    zentry = zmap.get(str(alert_index), {})
    z_rows = []
    if isinstance(zentry, dict):
        possible = zentry.get('top_z') or zentry.get('z_top') or zentry.get('top_zscores') or []
        if isinstance(possible, list):
            for t in possible:
                if isinstance(t, (list,tuple)) and len(t) >= 2:
                    try:
                        z_rows.append({"feature": str(t[0]), "z": float(t[1])})
                    except Exception:
                        z_rows.append({"feature": str(t[0]), "z": str(t[1])})
                elif isinstance(t, dict) and 'feature' in t and 'z' in t:
                    z_rows.append({"feature": str(t['feature']), "z": float(t['z'])})
    if not z_rows and isinstance(zentry, dict):
        for k,v in zentry.items():
            if k in ('top_index','meta'):
                continue
            try:
                z_rows.append({"feature": str(k), "z": float(v)})
            except Exception:
                pass

    z_rows = z_rows[:12]
    lof_table_rows = [lof_row] if lof_row["metric"] is not None else []

    return gbm_src, ablation_data, lof_table_rows, z_rows

# ---------------- Download explanation callback ----------------
@app.callback(
    Output("download-explain", "data"),
    Input("download-explain-btn", "n_clicks"),
    State("explain-table", "selected_rows"),
    State("explain-table", "data"),
    prevent_initial_call=True
)
def download_explanation(n_clicks, selected_rows, table_data):
    if not selected_rows or not table_data:
        return dash.no_update

    sel = selected_rows[0]
    row = table_data[sel]
    alert_index = row.get('alert_index', None)
    if alert_index is None:
        return dash.no_update

    gbm_png_path = str((GBM_DIR / f"gbm_shap_alert_{alert_index}.png").resolve()) if (GBM_DIR / f"gbm_shap_alert_{alert_index}.png").exists() else None
    gbm_json_path = str((GBM_DIR / f"gbm_shap_alert_{alert_index}.json").resolve()) if (GBM_DIR / f"gbm_shap_alert_{alert_index}.json").exists() else None

    ablation_map = load_if_ablation()
    ablation_entry = ablation_map.get(str(alert_index), None)

    zmap = load_if_zscores()
    zentry = zmap.get(str(alert_index), None)

    lof_df = load_if_lof_df()
    lof_entry = None
    if not lof_df.empty and 'top_index' in lof_df.columns:
        match = lof_df[lof_df['top_index'] == int(alert_index)]
        if not match.empty:
            lof_entry = match.iloc[0].to_dict()

    payload = {
        "alert_index": int(alert_index),
        "base_row": row,
        "gbm_png": gbm_png_path,
        "gbm_json": gbm_json_path,
        "if_ablation": ablation_entry,
        "if_zscores": zentry,
        "if_lof": lof_entry,
        "generated_at": time.time()
    }

    out_path = EXPLANATION_OUTDIR / f"alert_{alert_index}_explanation.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print("Failed to write explanation JSON:", e)
        return dash.no_update

    return dcc.send_file(str(out_path))

# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
