import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap # package used to calculate Shap values
import pickle


# append the path of the parent directory
import sys
sys.path.append("..")
from lib.utils import *


if __name__ == '__main__':
    # Clean sales data
    df_raw = pd.read_csv("../data/raw/ventaNavidad.csv")

    df = normalise_sales(df_raw)
    df = get_sales_stats(df)

    df.to_csv("../data/lottery_nr_sales.csv", index=False)

    # Get beauty metrics
    beauty_metrics = Nr_properties()
    # save to csv    
    beauty_metrics.to_csv('../data/nr_beauty_metrics.csv', index=False)

    #--------------------------

    features = pd.read_csv('../data/nr_beauty_metrics.csv', dtype={'str_n':str})
    target = pd.read_csv('../data/lottery_nr_sales.csv')

    # Tune the parameters of the model to improve the R² (feature engineering)
    features = clean_features(features)

    target = target['mean']

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.30, random_state=42)

    # train an XGBoost model
    xgb_model = xgb.XGBRegressor().fit(X_train, y_train)

    # train shap explainer
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(features)

    # Save models to pickel files
    pickle.dump(shap_values, open('../models/shapvalues.sav', 'wb'))
    pickle.dump(xgb_model, open('../models/xgb_model.pkl', 'wb'))
    pickle.dump(explainer, open('../models/shap_model.pkl', 'wb'))