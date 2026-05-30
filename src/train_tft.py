# Temporal Fussion Transformer

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from src.data_loader_eda import get_data

df, df_test = get_data()

series_list=[]
for store_id, grp in df.groupby('Store'):
    ts = TimeSeries.from_dataframe(grp.set_index('Date')[['Sales']],
                                   freq = 'D', fill_missing_dates=True, fillna_value=0)
    series_list.append(ts)

#Scaling
scaler= Scaler()
series_scaled=scaler.fit_transform(series_list)

#Data split
train = [s[:-102] for s in series_scaled]
val = [s[-102:] for s in series_scaled]

model = TFTModel(
    input_chunk_length=60,
    output_chunk_length=42,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    add_relative_index=True,
    dropout=.1,
    batch_size=64,
    n_epochs=1,
    likelihood=None,
    random_state=42
)

model.fit(train, val_series=val, verbose=True)

model.save('/home/aiman-nasir/AI-ML-projects/demand-forecasting/models/tft_model.pt')