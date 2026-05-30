from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from src.data_loader_eda import get_data


def load_forecast(store_id, horizon=30):

    df, df_test = get_data()
    scaler =Scaler()

    store_df = df[df["Store"] == store_id].sort_values("Date")
    ts = TimeSeries.from_dataframe(
        store_df.set_index("Date")[["Sales"]],
        freq="D",
        fill_missing_dates=True,
        fillna_value=0,
    )

    ts_scaled = scaler.fit_transform(ts)
    model = TFTModel.load('/home/aiman-nasir/AI-ML-projects/demand-forecasting/models/tft_model.pt')
    forecast_scaled = model.predict(n=horizon, series=ts_scaled)
    forecast = scaler.inverse_transform(forecast_scaled)

    forecast_df = forecast.to_dataframe().reset_index()
    forecast_df.columns = ["date", "mean"]

    # if the model does not provide uncertainty directly, approximate it
    forecast_df["std"] = forecast_df["mean"] * 0.05
    forecast_df["lower"] = forecast_df["mean"] - 1.96 * forecast_df["std"]
    forecast_df["upper"] = forecast_df["mean"] + 1.96 * forecast_df["std"]

    return forecast_df