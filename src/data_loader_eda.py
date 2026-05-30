# %% [markdown]
# ### One-time Load Data from Kaggle (must join competition first)

# %%
# import kagglehub

# kagglehub.login()

# %%
# Download latest version
# path = kagglehub.competition_download('rossmann-store-sales')

# print("Path to competition files:", path)

# %% [markdown]
# ### EDA

# %%
import pandas as pd
import matplotlib.pyplot as plt

train_path='/home/aiman-nasir/AI-ML-projects/demand-forecasting/data/train.csv'
test_path='/home/aiman-nasir/AI-ML-projects/demand-forecasting/data/test.csv'

df=pd.read_csv(train_path, parse_dates=['Date'])
df=df.sort_values(['Store', 'Date'])

df_test =pd.read_csv(train_path, parse_dates=['Date'])
df_test=df_test.sort_values(['Store', 'Date'])

df.head()

# %%
df.info()

# %%
df.StateHoliday.unique()

# %%
store1 = df[df['Store']==1]
store1.set_index('Date')['Sales'].plot(figsize=(14,4), title='Store 1 Daily Sales')
plt.tight_layout()
plt.show()

print(df.groupby('Promo')['Sales'].mean())

# %%
def get_data():
    """
    Return the loaded train and test dataframes for use when this notebook
    is imported as a module via import_ipynb.
    """
    return df, df_test

class RossmannDataLoader:
    @staticmethod
    def load_data():
        """Return copies of the already loaded dataframes."""
        return df.copy(), df_test.copy()

    @staticmethod
    def reload_data():
        """
        Reload the raw CSV files from disk and return fresh dataframes.
        Useful if you want to ensure the original data is read again.
        """
        train = pd.read_csv(train_path, parse_dates=['Date']).sort_values(['Store', 'Date'])
        test = pd.read_csv(test_path, parse_dates=['Date']).sort_values(['Store', 'Date'])
        return train, test

# %%



