import pandas as pd
from data import y
from BTCPred.encoders import X_scaled, scaler_y
from BTCPred.trainer import model


original = pd.DataFrame(y)
predictions = pd.DataFrame(scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1,1)))
