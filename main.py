import pandas as pd
import numpy as np
import xgboost as xgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = list(train[train.columns[0:len(train.columns) - 1]])

for i in features:
    if train[i].dtypes == 'O':
        train[i] = pd.factorize(train[i])[0]
        test[i] = pd.factorize(test[i])[0]

xgbmodel = xgb.XGBRegressor(data        = train[features],
                            label       = train['Response'],
                            eta         = 0.025,
                            depth       = 10,
                            nrounds     = 2500,
                            objective   = "reg:linear",
                            eval_metric = "rmse")


xgbmodel.fit(train[features], train['Response'])

predictions = xgbmodel.predict(test)
predictions

submission = pd.DataFrame()

submission['Id'] = test['Id']
submission['Response'] = np.round(predictions,0).astype('int')

submission.Response.value_counts()

submission['Response'] = np.where(submission['Response'] < 1, 1, submission['Response'])

submission.to_csv("submission.csv", index = False)
