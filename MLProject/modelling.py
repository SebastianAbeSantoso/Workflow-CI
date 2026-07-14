import argparse
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

mlflow.set_experiment("msml-ci-retraining")

model_df = pd.read_csv(args.data_path)
model_df['year_month'] = pd.to_datetime(model_df['year_month'])

cutoff = model_df['year_month'].quantile(0.8, interpolation='nearest')
train = model_df[model_df['year_month'] <= cutoff].reset_index(drop=True)
test = model_df[model_df['year_month'] > cutoff].reset_index(drop=True)

features = ['lag_1', 'lag_2', 'lag_3', 'rolling_avg_3', 'month', 'year',
            'province_enc', 'market_enc', 'commodity_enc']
target = 'target'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="gbr_ci_retrain") as run:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("Training complete.")