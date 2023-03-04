import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PyEMD import CEEMDAN

class DataPrepper():
  def __init__(self, tickers_list, features_list, start_date, end_date, normalize_result=False, scaler_type="std") -> None:
     self.tickers_list = tickers_list
     self.features_list = features_list
     self.start_date = start_date
     self.end_date = end_date
     self.normalize_result = normalize_result

     available_scaler_types = ["std", "minmax"]
     if scaler_type not in available_scaler_types:
        raise Exception(f"Scaler type {scaler_type} not available. Available scaler types are: {available_scaler_types}")
     self.scaler_type = scaler_type

     self.processed_data = {}
     self.original_data = {}
     self.scalers = {}

  def prepare(self,):
    for ticker in self.tickers_list:

        data = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=True, group_by="ticker")
        original_data_df = data.copy()
        self.original_data[ticker] = original_data_df

        data_scaled = pd.DataFrame()

        if self.normalize_result == True:
          if self.scaler_type == "std":
            current_scaler = StandardScaler()
          elif self.scaler_type == "minmax":
            current_scaler = MinMaxScaler()
          else:
             raise Exception(f"Unsupported scaler type = {self.scaler_type}")
          data_scaled[data.columns] = current_scaler.fit_transform(data[data.columns])
          self.scalers[ticker] = current_scaler
        else:
           data_scaled = data
         
        # data_descaled = pd.DataFrame()
        # data_descaled[data.columns] = current_scaler.inverse_transform(data_scaled[data.columns])

        for feature in self.features_list:
        
            ceemdan = CEEMDAN()
            cIMFs = ceemdan(data_scaled[feature].values)

            for i in range(len(cIMFs)):
                if i != len(cIMFs) - 1:
                    data[f"IMF{i}_{feature}"] = cIMFs[i]
                else:
                    data[f"Residue_{feature}"] = cIMFs[i]


        data.drop(self.features_list, axis=1, inplace=True)
        self.processed_data[ticker] = data

  def get_processed_data(self):
     return self.processed_data
  
  def get_original_data(self):
     return self.original_data
  
  def get_scalers(self):
     return self.scalers

    