import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import optuna, mlflow
from feature import clean_train_data, clean_test_data

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask])**2))

features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'dow_sin', 'dow_cos', 'lag_7',
            'lag_14', 'rolling_7d_mean', 'Month', 'Week']

X_train, y_train = clean_train_data()
X_val, y_val = clean_test_data()

train_data = lgb.Dataset(X_train[features], label=y_train)
val_data = lgb.Dataset(X_val[features], label=y_val, reference=train_data)

params={
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.05, 'num_leaves': 127,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'verbose': -1
}

model=lgb.train(params,train_data, 2000, valid_sets=[val_data], callbacks=[lgb.early_stopping(100)])

with open("/home/aiman-nasir/AI-ML-projects/demand-forecasting/models/lgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
# model.save_model('/models/model.txt')

preds = model.predict(X_val[features])
print(f'Validation RMSPE: {rmspe(y_val.values, preds):.4f}')

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_val[features].sample(500))
plt.figure(figsize=(12, 8)) 
plt.tight_layout()
shap.summary_plot(shap_vals,X_val[features].sample(500), show=False)
plt.savefig("lgb_shap_summary.png", bbox_inches="tight", dpi=300)
plt.close()

#________________________________________________

# Optimising Model Hyperparameters (optional)
#________________________________________________

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', .01, .3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', .5, 1),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = lgb.train({**params, 'objective': 'regression', 'metric': 'rmse', 'verbose':-1},
                          train_data, 500, valid_sets=[val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
        score = rmspe(y_val.values, model.predict(X_val[features]))
        mlflow.log_metric('val_rmspe', score)
        return score

opt_status = input('Conduct hyperparameter optimisation? (Y/N): ')

if opt_status.upper() == 'Y':
    with mlflow.start_run(run_name='optuna-lgb-search'):
        study=optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        print(f'Best RMSPE: {study.best_value}')
        print(f'Best params: {study.best_params}')