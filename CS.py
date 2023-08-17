import pandas  as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

pio.templates.default = "plotly_white"

data = pd.read_csv("train.csv")



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

model = RandomForestRegressor()
model.fit(xtrain,ytrain)

print("Credit Score Prediction:")

a = float(input("Annual Income: "))
b = float(input("Monthly Inhand salary: "))
c = float(input("Num of Bank accounts: "))
d= float(input("Num of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Num of Loans: "))
g = float(input("Avg num of delays by person: "))
h = float(input("Num of delayed payments: "))
i = float(input("cedit Mix(0 for bad, 1 for standard, 3 for good ): "))
j = float(input("Outstanding debt: "))
k = float(input("Credit History age: "))
l = float(input("Monthly balance: "))

features = np.array([[a,b,c,d,e,f,g,h,i,j,k,l]])
print("predicted credit score = ", model.predict(features))