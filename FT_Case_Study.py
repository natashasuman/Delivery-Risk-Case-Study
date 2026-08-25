import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 25)

RANDOM_STATE = 42

DATA_PATH = "C:/Users/natas/OneDrive/Desktop/ML-case_study.csv"

#EDA - most of the studying data is done on excel(mostly why and which colums should be removed based on judgement)

df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = [c.strip() for c in df.columns]
print("EDA")
print(f"Raw shape: {df.shape}")

target = "Late_delivery_risk"
before = len(df)
df = df[df["Delivery Status"] != "Shipping canceled"].copy()
#orders cancelled can not be classified as completed shipping mislabled hence removed
print(f"Dropped {before - len(df)} rows where Delivery Status == 'Shipping canceled'") 
print(f"Target balance:\n{df[target].value_counts(normalize=True).round(3)}")

print("MISSING VALUES") # to verify and remove these values if nmost of the data is missing
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print((miss / len(df) * 100).round(1).astype(str) + "%" if len(miss) else "None found")

print("ZERO-VARIANCE COLUMNS")
nunique = df.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
print(constant_cols if constant_cols else "None found")

#spliting the order date time into different columns to easily indetify any patterns in weekend/season/holidays
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="mixed")
df["order_dow"] = df["order_date"].dt.dayofweek
df["order_month"] = df["order_date"].dt.month
df["order_hour"] = df["order_date"].dt.hour
df = df.sort_values("order_date").reset_index(drop=True)


# Excluded these columns to prevent information lekage while training, also these won't be present at order time
leakage_cols = [
    "Days for shipping (real)",  
    "Delivery Status",             
    "shipping date (DateOrders)",  
]

# these are given name not useful and can't be predictive
pii_cols = [
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Street", "Customer Zipcode", "Order Zipcode",
    "Product Description", "Product Image", "Product Name", "Product Card Id",
]

# ambiguous order-time availability (judgment call)
ambiguous_cols = [
    "Order Status",   # evolves after placement of orders
    "Benefit per order", "Order Profit Per Order", "Order Item Profit Ratio",
]

zero_variance_cols = constant_cols  # Product Status always 0 which won't help our model(no variance to see patterns)

#1:1 mapping with scheduled days, no new information
redundant_cols = ["Shipping Mode"]

# tested directly, no meaningful relationship to target does not improves model with this information/no dependencies
tested_and_dropped = [
    "Order Item Quantity",      
    "Sales per customer",     
    "Customer Segment",        
    "Order Item Product Price",  
    "Order Item Discount Rate",  
]

#order time availability of these infomation(scheduled days)
categorical_features = ["Category Name", "Market", "Order Region", "Department Name", "Type"]
numeric_features = ["Days for shipment (scheduled)", "order_dow", "order_month", "order_hour"]
final_features = categorical_features + numeric_features

print("FEATURE LIST")
print(f"Categorical ({len(categorical_features)}):", categorical_features)
print(f"Numeric ({len(numeric_features)}):", numeric_features)
total_excluded = (len(leakage_cols) + len(pii_cols) + len(ambiguous_cols)
                   + len(zero_variance_cols) + len(redundant_cols) + len(tested_and_dropped))
print(f"Total kept: {len(final_features)} | Total excluded: {total_excluded}")

null_check = df[final_features + [target]].isnull().sum()
print("Null check:", "clean" if null_check.sum() == 0 else null_check[null_check > 0])

# verifying if anything in existing columns ahve any redundant columns
print("Training model for feature importance")

# splitting data for feature importance
split_idx = int(len(df) * 0.8)
split_date = df.loc[split_idx, "order_date"]
train_df = df[df["order_date"] < split_date]
test_df = df[df["order_date"] >= split_date]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])
rf_check = Pipeline([("prep", prep), ("clf", RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE))])
rf_check.fit(train_df[final_features], train_df[target])

# function for feature importance, one hor encoding is used for catergorical feature
def grouped_importance(fitted_pipeline, cat_feats, num_feats):
    ohe = fitted_pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_feats)
    all_names = list(cat_names) + num_feats
    raw = pd.Series(fitted_pipeline.named_steps["clf"].feature_importances_, index=all_names)
    grouped = {}
    for feat in cat_feats:
        grouped[feat] = raw[[c for c in all_names if c.startswith(feat + "_")]].sum()
    for feat in num_feats:
        grouped[feat] = raw[feat]
    return pd.Series(grouped).sort_values(ascending=False)

importance_check = grouped_importance(rf_check, categorical_features, numeric_features)
print("Feature importance %:")
print((importance_check / importance_check.sum() * 100).round(1))

LOW_IMPORTANCE_THRESHOLD = 0.005
redundant_features = ["Market", "Department Name"]    # similar columns already exist with 95% similar content       
drop_after_validation = sorted(set(redundant_features))#similar feature exist and low importance

categorical_features = [f for f in categorical_features if f not in drop_after_validation]
final_features = categorical_features + numeric_features

print("FINAL FEATURE LIST") #judgment filtering + redundant and feature importance
print(f"Categorical ({len(categorical_features)}):", categorical_features)
print(f"Numeric ({len(numeric_features)}):", numeric_features)
print(f"Total: {len(final_features)} features")

#Modelling - 

print("\n" + "="*70)
print(f"Split at {split_date.date()}  ->  train={len(train_df):,} rows, test={len(test_df):,} rows")
X_train, y_train = train_df[final_features], train_df[target]
X_test, y_test = test_df[final_features], test_df[target]

# using dummy classifier by sklearn to compare with baseline and other advance models
# this classifier just takes the most frequent in the data
# this gives 57% accuracy because the data is imbalaced around 58:42 (not significant for this amount of data)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

# logistic regression(baseline) with just 1 column as that contributes(84.7%) the most in model prediction(according to feature importance)
baseline = LogisticRegression(max_iter=1000)
baseline.fit(train_df[["Days for shipment (scheduled)"]], y_train)
pred_base = baseline.predict(test_df[["Days for shipment (scheduled)"]])
proba_base = baseline.predict_proba(test_df[["Days for shipment (scheduled)"]])[:, 1]


# Logistic Regression (full filtered feature set)
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])
logreg = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)
proba_lr = logreg.predict_proba(X_test)[:, 1]

# Random Forest (full filtered feature set)
rf = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE))])
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

# combined comparison table (test set)
print("\n" + "="*70)
print("RESULTS")
rows = []
for name, pred, proba in [
    ("Dummy (majority class)", pred_dummy, None),
    ("Baseline: Scheduled Days alone", pred_base, proba_base),
    ("Logistic Regression (full features)", pred_lr, proba_lr),
    ("Random Forest (full features)", pred_rf, proba_rf),
]:
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba) if proba is not None else None,
        "PR-AUC": average_precision_score(y_test, proba) if proba is not None else None,
    })
results = pd.DataFrame(rows)
print(results.round(3).to_string(index=False))


# OVERFITTING CHECK
# train vs test (large gap would mean memorized noise not a real pattern)
print("OVERFITTING CHECK (train vs test)")
def evaluate(model, X, y, label):
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {"Set": label, "Accuracy": accuracy_score(y, pred), "F1": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba)}

overfit_rows = []
for name, model in [("Logistic Regression", logreg), ("Random Forest", rf)]:
    overfit_rows.append({"Model": name, **evaluate(model, X_train, y_train, "TRAIN")})
    overfit_rows.append({"Model": name, **evaluate(model, X_test, y_test, "TEST")})
overfit_df = pd.DataFrame(overfit_rows)
print(overfit_df.round(3).to_string(index=False))


model_df = df[final_features + [target, "order_date"]].copy()
model_df.to_csv("model_ready_data.csv", index=False)


#new feature which will help ops control which orders to focus on
'''plt.figure(figsize=(8, 4.5))
plt.hist(proba_rf, bins=50, color="#A33638")
plt.xlabel("Predicted probability of being late")
plt.ylabel("Number of orders (test set)")
plt.title("Distribution of predicted risk scores")
plt.tight_layout()
plt.savefig("C:/Users/natas/OneDrive/Desktop/proba_distribution.png", dpi=120)
plt.close()
print("C:/Users/natas/OneDrive/Desktop/proba_distribution.png")'''


#finding bins for the threshold using probility of certain orders getting delayed
counts, bin_edges = pd.cut(proba_rf, bins=50, retbins=True)
bin_counts = pd.Series(proba_rf).groupby(pd.cut(proba_rf, bins=50)).size()
low_density_bins = bin_counts[bin_counts < bin_counts.max() * 0.02]
if len(low_density_bins) > 0:
    print("\nLow density in probability distribution:")
    print(low_density_bins)



print("\n" + "="*70)
print("Precision / Recall across threshold")
rows = []
for t in [round(x, 2) for x in [i / 100 for i in range(30, 90, 5)]]:
    pred = (proba_rf >= t).astype(int)
    rows.append({
        "Threshold": t,
        "% flagged": pred.mean(),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
    })
grid = pd.DataFrame(rows)
print(grid.round(3).to_string(index=False))


# Boundary 1 (Low and Medium) placed in the gap found in the distribution above
LOW_MEDIUM_CUT = 0.60
# for few false alarms here
TARGET_PRECISION = 0.90
candidates = grid[grid["Precision"] >= TARGET_PRECISION].sort_values("Threshold")
MEDIUM_HIGH_CUT = candidates["Threshold"].min() if len(candidates) else 0.80
 
print("FINAL RISK BOUNDARIES")
print(f"Low risk:    probability < {LOW_MEDIUM_CUT}")
print(f"Medium risk: {LOW_MEDIUM_CUT} <= probability < {MEDIUM_HIGH_CUT}")
print(f"High risk:   probability >= {MEDIUM_HIGH_CUT}  (precision >= {TARGET_PRECISION:.0%}")
 
risk_level = pd.cut(proba_rf, bins=[0, LOW_MEDIUM_CUT, MEDIUM_HIGH_CUT, 1.0],
               labels=["Low", "Medium", "High"])
risk_summary = pd.DataFrame({"risk_level": risk_level, "actual_late": y_test.values}).groupby("risk_level", observed=True).agg(
    n_orders=("actual_late", "size"),
    pct_of_total=("actual_late", lambda x: len(x) / len(y_test)),
    actual_late_rate=("actual_late", "mean"),
)
print("\nRisk on test set:")
print(risk_summary.round(3))