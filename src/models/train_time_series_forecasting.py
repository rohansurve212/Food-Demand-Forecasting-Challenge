import pandas as pd
from fbprophet import Prophet
from preprocess_ts_data import concatenate_features

# Make a dataframe of center id, meal id and the two concatenated into one string
center_meal_combo_id = concatenate_features(df_time_series, 'center_id', 'meal_id')

def make_time_series_predictions(full_raw_time_series, forecast_period):
    """
        Use Prophet: A Time Series Forecasting technique developed and open-sourced by Facebook
    """
    time_series_period = len(full_raw_time_series['week'].value_counts().index.to_list())
    total_predictions=pd.DataFrame()
    for combo in list(center_meal_combo_id['centre_id_meal_id']):
        combo_time_series = full_raw_time_series.loc[full_raw_time_series['center_meal_combo_id'] == combo,['ds','y']]
        # Instantiate a Prophet object and fit it to our time series
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(combo_time_series)
        # Make a dataframe of future dates
        future = m.make_future_dataframe(freq='W', periods=forecast_period)
        # Make predictions on future dates
        forecast = m.predict(future)
        mini_predictions = forecast.loc[:,['ds','yhat']]
        combo_series = pd.DataFrame([combo]*(time_series_period + forecast_period), columns=['centre_meal_combo_id'])
        predictions = pd.concat([combo_series, mini_predictions], axis=1)
        total_predictions = total_predictions.append(predictions, ignore_index=True)

    return total_predictions
