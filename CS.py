import pandas  as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("train.csv")

# print(data.head())
# print(data.info())
# print(data.isnull().sum())


