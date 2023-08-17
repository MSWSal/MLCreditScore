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

# print(data["Credit_Score"].value_counts())


# figure = px.box(data,
# x="Occupation",
# color="Credit_Score",
# title="CS based on Occ",
# color_discrete_map={'Poor':'red',
#                     'Standard': 'yellow',
#                     'Good':'green'})

# figure.show()

figure2 = px.box(data,
                 x="Credit_Score",
                 y = "Annual_Income",
                 color="Credit_Score",
                 title="CS based on annual Income",
                 color_discrete_map={'Poor':'red',
                                     'Standard': 'yellow',
                                     'Good':'green'})
figure2.update_traces(quartilemethod="exclusive")
figure2.show()



