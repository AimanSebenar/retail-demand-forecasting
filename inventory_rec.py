import numpy as np
from scipy.stats import norm

def inventory_recommendations(
        forecast_mean, forecast_std, lead_time_days=7, 
        service_level=.95, holding_cost_per_unit=.5, order_cost=20
):
    z = norm.ppf(service_level)

    # Safety stock: buffer for demand variability during lead time
    safety_stock = z*forecast_std* np.sqrt(lead_time_days)

    # reoder point: demand during lead time + safety stock
    daily_mean = forecast_mean/30
    reorder_point = (daily_mean*lead_time_days) + safety_stock

    #Economic ORder Quantity (EOQ)
    annual_demand = forecast_mean *12
    eoq = np.sqrt(((2*annual_demand*order_cost) / holding_cost_per_unit))

    return {
        'forecast_next_30d': round(forecast_mean, 0),
        'safety_stock': round(safety_stock, 0),
        'reorder_point': round(reorder_point, 0),
        'economic_order_qty': round(eoq, 0),
        'service_level': f'{service_level*100:.0f}%'
    }