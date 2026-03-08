# app_soc_dark.py
# Dark-styled SOC dashboard (copy of your app_soc.py with a dark aesthetic)
# Run: python app_soc_dark.py

import os
import io
import time
import base64
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

# ---------------- Settings ----------------
MAX_ROWS_PROCESS = 300_000
MODEL_FILES = {
    "scaler": Path("scaler_robust.pkl"),
    "if_model": Path("if_stage.pkl"),
    "reranker": Path("reranker_lgbm.pkl"),
    "shap_explainer": Path("shap_gbm/shap_gbm_explainer.joblib"),
}

# ---------- Thresholding config (tweak these) ----------
# IF_PERCENTILE: percentile of IF scores used as anomaly cutoff (e.g. 99 -> top 1%)
# Lowering this value will flag more rows as IF anomalies in small uploads.
IF_PERCENTILE =  85    # try 90 for top 10%, 95 for top 5%, 99 keeps top 1%

# RERANKER_THRESHOLD: probability threshold used for the supervised reranker (0..1)
# Lower -> more rows labelled suspicious by reranker.
RERANKER_THRESHOLD = 0.3   # try 0.3 (sensitive), 0.5 (default/strict), 0.2 (very sensitive)

# ---------------- App (dark theme) ----------------
external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Malware Traffic Detection (SOC demo) - Dark"

# ---------------- Global models ----------------
MODELS = {}
MODELS["_loaded_ok"] = False
MODELS["_load_error"] = None

# ---------- Shared dark styling helpers ----------
CARD_STYLE = {"backgroundColor": "#0b0f14", "border": "1px solid #1f2a33", "boxShadow": "0 4px 12px rgba(0,0,0,0.6)"}
SECTION_TITLE_STYLE = {"color": "#e6eef3", "marginBottom": "6px"}
TABLE_HEADER_STYLE = {"backgroundColor": "#162026", "color": "#e6eef3", "fontWeight": "600"}
TABLE_CELL_STYLE = {"backgroundColor": "#0b1115", "color": "#dfe9ef", "border": "none"}
PRE_STYLE = {"whiteSpace": "pre-wrap", "backgroundColor": "#050607", "color": "#cfe8ff", "padding": "12px", "borderRadius": "6px", "border": "1px solid #23303a"}

BTN_STYLE = {"marginRight": "8px", "backgroundColor": "#1b6ca8", "borderColor": "#174f79", "color": "white"}
BTN_SECONDARY_STYLE = {"marginRight": "8px", "backgroundColor": "#2a2f33", "borderColor": "#17191b", "color": "#cfe8ff"}

# ---------------- Utilities ----------------
def read_uploaded_csv(contents, filename):
    if contents is None:
        raise ValueError("No contents provided")

    if isinstance(contents, (list, tuple)):
        if len(contents) == 0:
            raise ValueError("Empty contents list")
        for c in contents:
            if isinstance(c, (str, bytes, bytearray)):
                contents = c
                break
        else:
            contents = contents[0]

    if isinstance(contents, (bytes, bytearray)):
        b = bytes(contents)
        ext = Path(filename).suffix.lower() if filename else ".csv"
        if ext == ".parquet":
            return pd.read_parquet(io.BytesIO(b))
        return pd.read_csv(io.BytesIO(b), low_memory=False)

    if isinstance(contents, str) and os.path.exists(contents):
        ext = Path(contents).suffix.lower()
        if ext == ".parquet":
            return pd.read_parquet(contents)
        return pd.read_csv(contents, low_memory=False)

    if not isinstance(contents, str):
        raise ValueError(f"Unsupported content type: {type(contents)}")

    if contents.startswith("data:") and "," in contents:
        try:
            _hdr, b64 = contents.split(",", 1)
            decoded = base64.b64decode(b64)
        except Exception as e:
            raise ValueError(f"Could not decode data URL: {e}")
    else:
        try:
            decoded = base64.b64decode(contents)
        except Exception:
            decoded = contents.encode("utf-8")

    try:
        return pd.read_csv(io.BytesIO(decoded), low_memory=False)
    except Exception:
        try:
            return pd.read_parquet(io.BytesIO(decoded))
        except Exception as e:
            try:
                text = decoded.decode("utf-8", errors="ignore")
                return pd.read_csv(io.StringIO(text), low_memory=False)
            except Exception as ee:
                raise ValueError(f"Failed to parse uploaded file: {e} / {ee}")

# ---------- Feature engineering (same as yours) ----------
def prepare_feature_matrix(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns.astype(str)]

    for c in [
        "duration",
        "fwd_pkts",
        "bwd_pkts",
        "fwd_bytes",
        "bwd_bytes",
        "flow_iat_mean",
        "flow_iat_std",
        "protocol",
    ]:
        if c not in df.columns:
            df[c] = 0

    numcols = [
        "duration",
        "fwd_pkts",
        "bwd_pkts",
        "fwd_bytes",
        "bwd_bytes",
        "flow_iat_mean",
        "flow_iat_std",
    ]
    for c in numcols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["protocol"] = df["protocol"].fillna("0").astype(str)
    uniques = sorted(list(df["protocol"].unique()))
    map_proto = {v: i for i, v in enumerate(uniques)}
    df["protocol"] = df["protocol"].map(map_proto).fillna(0).astype(int)

    eps = 1e-9
    df["tot_pkts"] = (df["fwd_pkts"] + df["bwd_pkts"]).replace(0, eps)
    df["tot_bytes"] = (df["fwd_bytes"] + df["bwd_bytes"]).replace(0, eps)

    df["log_duration"] = np.log1p(df["duration"].clip(lower=0))
    df["avg_pkt_size"] = df["tot_bytes"] / df["tot_pkts"]
    df["pkt_rate"] = df["tot_pkts"] / df["duration"].replace(0, eps)
    df["byte_rate"] = df["tot_bytes"] / df["duration"].replace(0, eps)
    df["fwd_bwd_pkt_ratio"] = (df["fwd_pkts"] + 1.0) / (df["bwd_pkts"] + 1.0)
    df["fwd_bwd_byte_ratio"] = (df["fwd_bytes"] + 1.0) / (df["bwd_bytes"] + 1.0)
    df["size_iat_ratio"] = df["avg_pkt_size"] / df["flow_iat_mean"].replace(0, 1.0)
    df["iat_cv"] = df["flow_iat_std"] / df["flow_iat_mean"].replace(0, 1.0)
    df["bytes_per_pkt_rate"] = df["byte_rate"] / df["pkt_rate"].replace(0, 1.0)
    df["log_pkt_rate"] = np.log1p(df["pkt_rate"].clip(lower=0))
    df["log_byte_rate"] = np.log1p(df["byte_rate"].clip(lower=0))
    df["log_avg_pkt_size"] = np.log1p(df["avg_pkt_size"].clip(lower=0))
    df["log_size_iat_ratio"] = np.log1p(df["size_iat_ratio"].clip(lower=0))
    df["log_bytes_per_pkt_rate"] = np.log1p(df["bytes_per_pkt_rate"].clip(lower=0))

    features_21 = [
        "duration",
        "log_duration",
        "fwd_pkts",
        "bwd_pkts",
        "tot_pkts",
        "fwd_bytes",
        "bwd_bytes",
        "avg_pkt_size",
        "pkt_rate",
        "byte_rate",
        "fwd_bwd_pkt_ratio",
        "fwd_bwd_byte_ratio",
        "size_iat_ratio",
        "iat_cv",
        "bytes_per_pkt_rate",
        "log_pkt_rate",
        "log_byte_rate",
        "log_avg_pkt_size",
        "log_size_iat_ratio",
        "log_bytes_per_pkt_rate",
        "protocol",
    ]

    for col in features_21:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features_21].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X, features_21

# ---------- Model loader ----------
def load_models(force_reload: bool = False):
    global MODELS
    if MODELS.get("_loaded_ok") and not force_reload:
        return MODELS

    import joblib

    MODELS.clear()
    MODELS["_loaded_ok"] = False
    MODELS["_load_error"] = None

    for k, p in MODEL_FILES.items():
        MODELS[k] = None
        try:
            print(f"[load_models] checking {k}: {p} exists? {p.exists()}")
            if p.exists():
                MODELS[k] = joblib.load(p)
                print(f"[load_models] loaded {k}: {type(MODELS[k])}")
            else:
                print(f"[load_models] {p} not found (optional={k=='shap_explainer'})")
        except Exception as e:
            MODELS[k] = None
            print(f"[load_models] failed to load {k} from {p}: {e}")
            traceback.print_exc()

    MODELS["_loaded_ok"] = any(MODELS.get(k) is not None for k in ["if_model", "reranker"])
    if not MODELS["_loaded_ok"]:
        MODELS["_load_error"] = "No main models loaded (both IF and reranker are None)"
    return MODELS

# ---------------- Layout (dark UI) ----------------
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2("Malware Traffic Detection — SOC Demo"),
                style={"paddingTop": "8px", "paddingBottom": "6px", "color": "#cfe8ff"},
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Upload a flow CSV (min columns: duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, flow_iat_mean, flow_iat_std, protocol).",
                                        style={"color": "#bcd8f2"},
                                    ),
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div(["Drag & Drop or Click to Select CSV/Parquet"]),
                                        style={
                                            "width": "100%",
                                            "height": "56px",
                                            "lineHeight": "56px",
                                            "borderWidth": "1px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "6px",
                                            "textAlign": "center",
                                            "marginBottom": "8px",
                                            "backgroundColor": "#071018",
                                            "color": "#cfe8ff",
                                            "borderColor": "#1f2a33",
                                        },
                                        multiple=False,
                                    ),
                                    html.Div(id="upload-status", style={"whiteSpace": "pre-wrap", "color": "#ffb3b3"}),
                                ]
                            )
                        ],
                        style=CARD_STYLE,
                    ),
                    md=12,
                )
            ],
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.Div("Uploaded rows (first 50)", style=SECTION_TITLE_STYLE)),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="uploaded-table",
                                    page_size=10,
                                    style_header=TABLE_HEADER_STYLE,
                                    style_cell=TABLE_CELL_STYLE,
                                    style_table={"overflowX": "auto"},
                                    tooltip_delay=300,
                                )
                            ),
                        ],
                        style=CARD_STYLE,
                    ),
                    md=7,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.Div("Top per-row outputs", style=SECTION_TITLE_STYLE)),
                            dbc.CardBody(
                                [
                                    dash_table.DataTable(
                                        id="top-outputs",
                                        page_size=10,
                                        style_header=TABLE_HEADER_STYLE,
                                        style_cell=TABLE_CELL_STYLE,
                                        style_table={"overflowX": "auto"},
                                    ),
                                    html.Div(style={"height": "8px"}),
                                    html.Div(
                                        [
                                            html.Button("Load Models (lazy)", id="btn-load-models", style=BTN_STYLE),
                                            html.Button("Download suspicious rows", id="btn-download-suspicious", style=BTN_SECONDARY_STYLE),
                                            html.Div(id="model-load-status", style={"whiteSpace": "pre-wrap", "color": "#9ef0b9", "display": "inline-block", "marginLeft": "8px"}),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        style=CARD_STYLE,
                    ),
                    md=5,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.Div("Investigation — suspicious rows (original columns)", style=SECTION_TITLE_STYLE)),
                        dbc.CardBody(
                            [
                                html.P(
                                    "This table shows full original columns for rows classified as suspicious by either the reranker or Isolation Forest.",
                                    style={"color": "#bcd8f2"},
                                ),
                                dash_table.DataTable(
                                    id="suspicious-table",
                                    page_size=10,
                                    style_header=TABLE_HEADER_STYLE,
                                    style_cell=TABLE_CELL_STYLE,
                                    style_table={"overflowX": "auto"},
                                ),
                            ]
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                md=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.Div("Analyst report (why these flows look malicious / anomalous)", style=SECTION_TITLE_STYLE)),
                        dbc.CardBody(
                            html.Pre(id="report-text", style=PRE_STYLE)
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                md=12,
            ),
            className="mb-4",
        ),
        # hidden store
        dcc.Store(id="uploaded-store", storage_type="memory"),
    ],
    fluid=True,
    style={"backgroundColor": "#050607", "minHeight": "100vh", "padding": "18px"},
)

# ---------- Main upload callback (unchanged logic, minor refactor uses IF_PERCENTILE & RERANKER_THRESHOLD) ----------
@app.callback(
    Output("upload-status", "children"),
    Output("uploaded-table", "data"),
    Output("uploaded-table", "columns"),
    Output("top-outputs", "data"),
    Output("top-outputs", "columns"),
    Output("uploaded-store", "data"),
    Output("report-text", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    start = time.time()
    if contents is None:
        return (
            "No file uploaded",
            [],
            [],
            [],
            [],
            None,
            "",
        )

    try:
        df = read_uploaded_csv(contents, filename)
    except Exception as e:
        tb = traceback.format_exc()
        return (
            f"ERROR processing upload: {e}\n\n{tb}",
            [],
            [],
            [],
            [],
            None,
            "Error while reading file; no report generated.",
        )

    nrows = len(df)
    if nrows == 0:
        return (
            "Uploaded file parsed but contains 0 rows",
            [],
            [],
            [],
            [],
            None,
            "No data rows available to analyse.",
        )

    if nrows > MAX_ROWS_PROCESS:
        df = df.sample(MAX_ROWS_PROCESS, random_state=42).reset_index(drop=True)
        nrows = len(df)

    try:
        X, feats = prepare_feature_matrix(df)
    except Exception as e:
        tb = traceback.format_exc()
        return (
            f"ERROR while preparing features: {e}\n\n{tb}",
            [],
            [],
            [],
            [],
            None,
            "Feature engineering failed; no report generated.",
        )

    stats = {}
    for col, qs in [
        ("pkt_rate", [0.5, 0.95]),
        ("byte_rate", [0.5, 0.95]),
        ("tot_bytes", [0.5, 0.95]),
        ("duration", [0.5, 0.95]),
        ("iat_cv", [0.1, 0.5]),
        ("fwd_bwd_byte_ratio", [0.05, 0.95]),
    ]:
        if col in X.columns:
            for q in qs:
                try:
                    stats[f"{col}_{int(q*100)}"] = float(X[col].quantile(q))
                except Exception:
                    pass

    models = load_models()
    model_status = "models_loaded" if models.get("_loaded_ok") else "models_not_loaded"

    uploaded_cols = [{"name": c, "id": c} for c in df.columns[:50]]
    uploaded_data = df.head(50).to_dict("records")

    outputs = []
    out_cols = []
    verdict = None
    status_debug = ""
    seen_debug = []
    report_text = ""

    try:
        scaler = models.get("scaler")
        if_model = models.get("if_model")
        reranker = models.get("reranker")

        def prepare_input_for_model(model, X_df, X_scaled_arr=None):
            fnames = getattr(model, "feature_names_in_", None)
            if fnames is not None:
                fnames = list(fnames)
                for f in fnames:
                    if f not in X_df.columns:
                        X_df[f] = 0.0
                return X_df[fnames].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0).values

            n_exp = getattr(model, "n_features_in_", None)
            if X_scaled_arr is not None:
                base_arr = np.asarray(X_scaled_arr)
            else:
                base_arr = X_df.astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0).values

            if n_exp is not None:
                cur_n = base_arr.shape[1]
                if cur_n == n_exp:
                    return base_arr
                elif cur_n > n_exp:
                    return base_arr[:, :n_exp]
                else:
                    pad = np.zeros((base_arr.shape[0], n_exp - cur_n))
                    return np.hstack([base_arr, pad])
            return base_arr

        try:
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                feat_in = list(scaler.feature_names_in_)
                for f in feat_in:
                    if f not in X.columns:
                        X[f] = 0.0
                X_prep = X[feat_in].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                seen_debug.append(f"Using scaler.feature_names_in_ ({len(feat_in)} cols)")
            else:
                X_prep = X.copy().astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                seen_debug.append("Scaler has no feature_names_in_; using local feature order")
        except Exception as e:
            X_prep = X.copy().astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            seen_debug.append(f"Exception reconciling features: {e}")

        X_scaled = None
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_prep)
                seen_debug.append("Scaler transform succeeded")
            except Exception as e:
                try:
                    n_exp = getattr(scaler, "n_features_in_", None)
                    base = X_prep.values
                    if n_exp is not None:
                        cur_n = base.shape[1]
                        if cur_n < n_exp:
                            extra = np.zeros((base.shape[0], n_exp - cur_n))
                            base2 = np.hstack([base, extra])
                        else:
                            base2 = base[:, :n_exp]
                        X_scaled = scaler.transform(base2)
                        seen_debug.append(f"Scaler transform by padding/truncating to {n_exp}")
                    else:
                        X_scaled = scaler.transform(base)
                        seen_debug.append("Scaler transform fallback used raw base")
                except Exception as e2:
                    X_scaled = None
                    seen_debug.append(f"Scaler transform ultimately failed: {e2}")

        # ----- IF scoring -----
        scores = None
        if_threshold = float("inf")
        if if_model is not None:
            try:
                X_for_if = prepare_input_for_model(if_model, X_prep, X_scaled_arr=X_scaled)
                try:
                    raw_scores = if_model.decision_function(X_for_if)
                    scores = -raw_scores
                    seen_debug.append(f"IF decision_function succeeded with input shape {X_for_if.shape}")
                except Exception as e_if_df:
                    preds = if_model.predict(X_for_if)
                    scores = np.where(preds == -1, 1.0, 0.0)
                    seen_debug.append(f"IF decision_function failed; used predict fallback (shape {X_for_if.shape}): {e_if_df}")
            except Exception as e:
                scores = None
                seen_debug.append(f"IF scoring failed: {e}")

        # compute IF threshold using IF_PERCENTILE (configurable)
        if scores is not None:
            try:
                if_scores_arr = np.asarray(scores)
                if if_scores_arr.size > 0:
                    if_threshold = float(np.percentile(if_scores_arr, IF_PERCENTILE))
                else:
                    if_threshold = float("inf")
            except Exception:
                if_threshold = float("inf")

        # ----- Reranker scoring -----
        rerank_scores = None
        if reranker is not None:
            try:
                X_for_reranker = prepare_input_for_model(reranker, X_prep, X_scaled_arr=X_scaled)
                try:
                    proba = reranker.predict_proba(X_for_reranker)
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        rerank_scores = proba[:, 1]
                    else:
                        rerank_scores = proba[:, 0]
                    seen_debug.append(f"Reranker.predict_proba succeeded with input shape {X_for_reranker.shape}")
                except Exception as e_proba:
                    try:
                        pred_scores = reranker.predict(X_for_reranker)
                        rerank_scores = np.asarray(pred_scores, dtype=float)
                        seen_debug.append(f"Reranker.predict succeeded with input shape {X_for_reranker.shape}")
                    except Exception as e_pred:
                        rerank_scores = None
                        seen_debug.append(f"Reranker scoring failed both predict_proba and predict: {e_proba} / {e_pred}")
            except Exception as e:
                rerank_scores = None
                seen_debug.append(f"Reranker prepare/score failed: {e}")

        # ----- Build outputs & verdicts -----
        top_df_for_report = None
        if scores is not None:
            N = min(200, X.shape[0])
            out_df = pd.DataFrame({"index": np.arange(N), "anomaly_score": np.round(scores[:N], 6)})
            if rerank_scores is not None:
                out_df["rerank_score"] = np.round(rerank_scores[:N], 6)
            if "tot_pkts" in X.columns:
                out_df["tot_pkts"] = X["tot_pkts"].astype(int).values[:N]
            if "tot_bytes" in X.columns:
                out_df["tot_bytes"] = X["tot_bytes"].astype(int).values[:N]

            def row_verdict(row):
                try:
                    rscore = float(row.get("rerank_score", 0.0))
                except Exception:
                    rscore = 0.0
                try:
                    iscore = float(row.get("anomaly_score", 0.0))
                except Exception:
                    iscore = 0.0

                # supervised reranker check (configurable)
                if rerank_scores is not None and rscore >= RERANKER_THRESHOLD:
                    return "Suspicious (reranker)"

                # IF anomaly check using IF_PERCENTILE-derived threshold
                if scores is not None and iscore >= if_threshold:
                    return "Suspicious (IF anomaly)"

                return "Benign"

            out_df["verdict"] = out_df.apply(row_verdict, axis=1)
            outputs = out_df.to_dict("records")
            out_cols = [{"name": c, "id": c} for c in out_df.columns]
            top_df_for_report = out_df.copy()
        else:
            outputs = []
            out_cols = []

        # ----- App-level verdict -----
        verdict = "Unknown"
        details = []
        suspicious_by_rerank = False
        suspicious_by_if = False

        if scores is not None:
            if rerank_scores is not None:
                try:
                    rer = np.asarray(rerank_scores)
                    cnt_rer = int(np.sum(rer >= RERANKER_THRESHOLD))
                    suspicious_by_rerank = cnt_rer > 0
                    details.append(f"Reranker threshold={RERANKER_THRESHOLD}, count={cnt_rer}")
                except Exception:
                    suspicious_by_rerank = False

            if not suspicious_by_rerank:
                try:
                    if_scores_arr = np.asarray(scores)
                    if if_scores_arr.size > 0:
                        top_count = int(np.sum(if_scores_arr >= if_threshold))
                        suspicious_by_if = top_count > 0
                        details.append(f"IF top{100-IF_PERCENTILE}% threshold (percentile={IF_PERCENTILE})={if_threshold:.4f}, count={top_count}")
                except Exception:
                    suspicious_by_if = False

            if suspicious_by_rerank:
                verdict = "MALWARE SUSPECTED (by reranker)"
            elif suspicious_by_if:
                verdict = "MALWARE SUSPECTED (by IF top anomalies)"
            else:
                verdict = "No strong malware signals detected"
        else:
            details.append("No scores computed (models missing or scoring failed)")

        status_debug = " | ".join(seen_debug + details)

        # ----- Analyst report text ----- #
        report_lines = []
        report_lines.append(f"Total flows analysed: {nrows} (showing top {min(200, nrows)} for scoring).")
        report_lines.append(f"Models: {model_status}")
        report_lines.append(f"App-level verdict: {verdict}")
        report_lines.append("")

        if scores is None:
            report_lines.append("No anomaly scores were computed. This usually means the models failed to load.")
        else:
            if suspicious_by_rerank:
                report_lines.append("- At least one flow crossed the malware classifier threshold (rerank_score ≥ {:.2f}).".format(RERANKER_THRESHOLD))
            elif suspicious_by_if:
                report_lines.append(f"- No flows crossed the malware classifier threshold, but Isolation Forest flagged the top {100-IF_PERCENTILE}% most anomalous flows for investigation (IF percentile={IF_PERCENTILE}).")
            else:
                report_lines.append("- Neither the reranker nor Isolation Forest found flows that stand out strongly from the benign baseline.")
            report_lines.append("")

            if top_df_for_report is not None:
                sus_top = top_df_for_report[top_df_for_report["verdict"].str.contains("Suspicious", na=False)]
                if sus_top.empty:
                    report_lines.append("No individual flows in the analysed window were labelled as Suspicious; the dataset currently looks benign.")
                else:
                    report_lines.append(f"Suspicious flows within analysed window: {len(sus_top)}")
                    for _, row in sus_top.iterrows():
                        idx = int(row["index"])
                        a_score = float(row["anomaly_score"])
                        r_score = float(row.get("rerank_score", 0.0))
                        verdict_row = row["verdict"]

                        report_lines.append("")
                        report_lines.append(f"Flow index {idx} — {verdict_row}")
                        report_lines.append(f"  Anomaly score (IF): {a_score:.6f}")
                        report_lines.append(f"  Malware probability (reranker): {r_score:.6f}")

                        reasons = []
                        if 0 <= idx < len(X):
                            feat_row = X.iloc[idx]

                            if ("pkt_rate_95" in stats) and (feat_row.get("pkt_rate", 0.0) >= stats["pkt_rate_95"]):
                                reasons.append(f"Unusually high packet rate ({feat_row.get('pkt_rate', 0.0):.2f} pkts/s vs ~{stats['pkt_rate_95']:.2f} 95th percentile).")

                            if ("byte_rate_95" in stats) and (feat_row.get("byte_rate", 0.0) >= stats["byte_rate_95"]):
                                reasons.append("Very high byte throughput compared to normal flows (possible large transfer / exfiltration).")

                            if ("tot_bytes_95" in stats) and (feat_row.get("tot_bytes", 0.0) >= stats["tot_bytes_95"]):
                                reasons.append(f"Total bytes in flow are unusually large ({feat_row.get('tot_bytes', 0.0):.0f} bytes).")

                            if ("duration_95" in stats) and (feat_row.get("duration", 0.0) >= stats["duration_95"]):
                                reasons.append("Flow duration is very long compared to typical traffic, consistent with persistent C2 channels.")

                            fbr = float(feat_row.get("fwd_bwd_byte_ratio", 1.0))
                            low_key = "fwd_bwd_byte_ratio_5"
                            high_key = "fwd_bwd_byte_ratio_95"
                            if high_key in stats and fbr >= stats[high_key]:
                                reasons.append("Traffic is strongly one-directional in bytes (forward >> backward) — possible exfiltration or bulk upload.")
                            elif low_key in stats and fbr <= stats[low_key]:
                                reasons.append("Traffic is strongly one-directional in bytes (backward >> forward) — heavy server->client transfer detected.")

                            if ("iat_cv_10" in stats) and (feat_row.get("iat_cv", 0.0) <= stats["iat_cv_10"]):
                                reasons.append("Inter-arrival times have very low variability (beacon-like communication).")

                        if not reasons:
                            reasons.append("The model flagged this flow as anomalous compared to training data; combination of bytes/timing/direction differs from typical benign traffic.")

                        for r in reasons:
                            report_lines.append(f"  • {r}")

        report_text = "\n".join(report_lines)

    except Exception as e:
        outputs = []
        out_cols = []
        verdict = "Error during scoring"
        status_debug = f"Scoring exception: {e}\n{traceback.format_exc()}"
        report_text = (
            "An error occurred while scoring the flows. Check console traceback.\n\n" + status_debug
        )

    status_msg = f"Parsed {nrows} rows, prepared {len(feats)} features in {(time.time() - start):.1f}s. Models: {model_status}."
    if verdict:
        status_msg += f" Verdict: {verdict}."
    if status_debug:
        status_msg += f" Debug: {status_debug}"

    print("[handle_upload] " + status_msg)

    try:
        uploaded_store_json = df.to_json(date_format="iso", orient="split")
    except Exception:
        uploaded_store_json = None

    return status_msg, uploaded_data, uploaded_cols, outputs, out_cols, uploaded_store_json, report_text

# ---------- model load button ----------
@app.callback(
    Output("model-load-status", "children"),
    Input("btn-load-models", "n_clicks"),
    prevent_initial_call=True,
)
def on_click_load(n):
    t0 = time.time()
    load_models(force_reload=True)
    ok = MODELS.get("_loaded_ok", False)
    if ok:
        return f"Models loaded successfully in {(time.time() - t0):.1f}s"
    else:
        return f"Model load failed:\n{MODELS.get('_load_error')}"

# ---------- suspicious investigation table ----------
@app.callback(
    Output("suspicious-table", "data"),
    Output("suspicious-table", "columns"),
    Output("suspicious-table", "style_data_conditional"),
    Input("top-outputs", "data"),
    State("uploaded-store", "data"),
    prevent_initial_call=False,
)
def build_investigation_table(top_outputs, uploaded_store_json):
    if not uploaded_store_json or not top_outputs:
        return [], [], []

    try:
        full_df = pd.read_json(uploaded_store_json, orient="split")
    except Exception:
        try:
            full_df = pd.DataFrame(uploaded_store_json)
        except Exception:
            return [], [], []

    top_df = pd.DataFrame(top_outputs)
    if top_df.empty:
        return [], [], []

    suspicious_idx = top_df[top_df["verdict"].str.contains("Suspicious", na=False)]["index"].tolist()
    if not suspicious_idx:
        return [], [], []

    try:
        sus_df = full_df.loc[suspicious_idx].reset_index(drop=True)
    except Exception:
        sus_df = full_df.iloc[suspicious_idx].reset_index(drop=True)

    cols = [{"name": str(c), "id": str(c)} for c in sus_df.columns]
    data = sus_df.to_dict("records")

    style = [
        {"if": {"row_index": i}, "backgroundColor": "#2a1111", "color": "#ffdede"}
        for i in range(len(data))
    ]

    return data, cols, style

# ---------- download suspicious rows ----------
@app.callback(
    Output("download-suspicious", "data"),
    Input("btn-download-suspicious", "n_clicks"),
    State("top-outputs", "data"),
    State("uploaded-store", "data"),
    prevent_initial_call=True,
)
def download_suspicious(n_clicks, top_outputs_data, uploaded_store_json):
    if not top_outputs_data or not uploaded_store_json:
        return dash.no_update

    top_df = pd.DataFrame(top_outputs_data)
    try:
        full_df = pd.read_json(uploaded_store_json, orient="split")
    except Exception:
        full_df = pd.DataFrame(uploaded_store_json)

    suspicious_idx = top_df[top_df["verdict"].str.contains("Suspicious", na=False)]["index"].tolist()
    if not suspicious_idx:
        return dcc.send_data_frame(pd.DataFrame({"message": ["No suspicious rows detected in this upload."]}).to_csv, "no_suspicious.csv", index=False)

    try:
        sus_df = full_df.loc[suspicious_idx].reset_index(drop=True)
    except Exception:
        sus_df = full_df.iloc[suspicious_idx].reset_index(drop=True)

    return dcc.send_data_frame(sus_df.to_csv, "suspicious_rows.csv", index=False)

# ---------------- Run server ----------------
if __name__ == "__main__":
    print("Starting dark SOC app (fast start; models load lazily).")
    app.run(debug=True, host="0.0.0.0", port=8050)
