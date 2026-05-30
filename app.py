import streamlit as st 
import plotly.graph_objects as go 
import pandas as pd
from inventory_rec import inventory_recommendations
from src.data_loader_eda import get_data
from src.forecast import load_forecast

df, df_test = get_data()

st.set_page_config(page_title='Demand Forecaster', layout='wide')
st.title('Retail Demand Forecasting System')

store_id = st.sidebar.selectbox('Select store', sorted(df['Store'].unique()))
service_level = st.sidebar.slider('Service level', 0.8, .99, .95, step=.01)

hist= df[df['Store'] == store_id].set_index('Date')['Sales']
forecast = load_forecast(store_id)

fig = go.Figure()
fig.add_scatter(x=hist.index[-90:], y=hist[-90:], name='Historical', line=dict(color='#534AB7'))

fig.add_scatter(x=forecast['date'], y=forecast['mean'], name='Forecast', line=dict(color='#1D9E75', dash='dot'))

fig.add_scatter(x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                y=forecast['upper'].tolist() + forecast['lower'].tolist()[::-1],
                fill='toself', fillcolor='rgba(29, 158, 117, .15)',
                line=dict(color='rgba(0,0,0,0)'), name=f'80% interval')

st.plotly_chart(fig, use_container_width=True)

rec = inventory_recommendations(forecast_mean=forecast['mean'].sum(), forecast_std=forecast['std'].mean())
col1, col2, col3, col4 = st.columns(4)
col1.metric('30-day forecast', f'{int(rec['forecast_next_30d']):,} units')
col2.metric('Safety stock', f'{int(rec['safety_stock']):,} units')
col3.metric('Reorder point', f'{int(rec['reorder_point']):,} units')
col4.metric('Optimal order qty', f'{int(rec['economic_order_qty']):,} units')