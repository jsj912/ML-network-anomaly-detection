import joblib
print("if_stage.pkl ->", type(joblib.load("if_stage.pkl")))
print("if_scaler.pkl ->", type(joblib.load("if_scaler.pkl")))
# also print the other candidates so you can compare:
print("if_model.pkl ->", type(joblib.load("if_model.pkl")))
print("scaler_robust.pkl ->", type(joblib.load("scaler_robust.pkl")))
