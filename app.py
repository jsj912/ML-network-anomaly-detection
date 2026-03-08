import dash
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import numpy as np
import io
import base64

# -------------------
# Load Model + Metrics
# -------------------
model = joblib.load("baseline_lightgbm.pkl")

# Read metrics text
with open("metrics_lightgbm.txt", "r") as f:
    metrics_text = f.read()

# Load dataset sample for visualization
combined = pd.read_csv("combined_cic_ctu.csv", nrows=10000, low_memory=False)
combined['label'] = pd.to_numeric(combined['label'], errors='coerce').fillna(0).astype(int)
combined['protocol'] = pd.to_numeric(combined['protocol'], errors='coerce').fillna(0)

# -------------------
# Initialize Dash App
# -------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
app.title = "Malware Traffic Analysis Dashboard"

# -------------------
# Tab 1 — Dataset Overview
# -------------------
def dataset_overview():
    label_counts = combined['label'].value_counts().rename({0: 'Benign', 1: 'Malicious'})
    fig_label = px.pie(values=label_counts.values, names=label_counts.index, title="Traffic Class Distribution")

    protocol_counts = combined['protocol'].value_counts().nlargest(10)
    fig_proto = px.bar(x=protocol_counts.index, y=protocol_counts.values, title="Top 10 Protocols")

    return dbc.Container([
        html.H4("Dataset Overview", className="text-center mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_label), md=6),
            dbc.Col(dcc.Graph(figure=fig_proto), md=6)
        ]),
        html.P(f"Total samples: {len(combined):,}", className="text-center mt-4")
    ])

# -------------------
# Tab 2 — Model Performance
# -------------------
def model_performance():
    return dbc.Container([
        html.H4("Model Performance", className="text-center mb-4"),
        html.Pre(metrics_text, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'}),
        html.P("ROC Curve & Confusion Matrix are generated during model training.", className="text-center text-muted")
    ])

# -------------------
# Tab 3 — Feature Importance
# -------------------
def feature_importance():
    importances = model.feature_importances_
    features = list(combined.drop(columns=['label']).columns)
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15)
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title="Top 15 Feature Importances", color='Importance')
    return dbc.Container([
        html.H4("Feature Importance", className="text-center mb-4"),
        dcc.Graph(figure=fig)
    ])

# -------------------
# Tab 4 — Live Detection
# -------------------
def live_detection():
    return dbc.Container([
        html.H4("🚨 Live Detection — Upload Flow CSV", className="text-center mb-4"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag & Drop or ', html.A('Select a CSV File')]),
            style={
                'width': '100%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '2px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-prediction')
    ])

@app.callback(
    Output('output-prediction', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return html.Div()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df['protocol'] = pd.to_numeric(df.get('protocol', 0), errors='coerce').fillna(0)
    X_new = df.select_dtypes(include=[np.number]).fillna(0)
    preds = model.predict(X_new)
    df['Prediction'] = np.where(preds == 1, 'Malicious', 'Benign')

    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.head(20).to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'fontSize': 14},
        page_size=10
    )
    return html.Div([
        html.H5(f"File uploaded: {filename}"),
        html.P(f"Total rows: {len(df):,}"),
        html.P(f"Malicious predictions: {(df['Prediction']=='Malicious').sum()}"),
        html.Hr(),
        table
    ])

# -------------------
# Layout with Tabs
# -------------------
app.layout = dbc.Container([
    html.H2("Malware Traffic Detection Dashboard", className="text-center my-4"),
    dcc.Tabs([
        dcc.Tab(label='Dataset Overview', children=[dataset_overview()]),
        dcc.Tab(label='Model Performance', children=[model_performance()]),
        dcc.Tab(label='Feature Importance', children=[feature_importance()]),
        dcc.Tab(label='Live Detection', children=[live_detection()])
    ])
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
