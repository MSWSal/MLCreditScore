import pandas  as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

pio.templates.default = "plotly_white"

data = pd.read_csv("train.csv")

print(data.head())
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

# figure2 = px.box(data,
#                  x="Credit_Score",
#                  y = "Annual_Income",
#                  color="Credit_Score",
#                  title="CS based on annual Income",
#                  color_discrete_map={'Poor':'red',
#                                      'Standard': 'yellow',
#                                      'Good':'green'})
# figure2.update_traces(quartilemethod="exclusive")
# figure2.show()

data["Credit_Mix"] = data["Credit_Mix"].map({"Standard":1,
                                             "Good":2,
                                             "Bad":0})

x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary","Num_Bank_Accounts",
                   "Num_Credit_Card","Interest_Rate","Num_of_Loan",
                   "Delay_from_due_date","Num_of_Loan",
                   "Delay_from_due_date","Num_of_Delayed_Payment",
                   "Credit_Mix","Outstanding_Debt",
                   "Credit_History_Age","Monthly_Balance"]])

y = np.array(data[["Credit_Score"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

model = RandomForestClassifier()
model.fit(xtrain,ytrain)





