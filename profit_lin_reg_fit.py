import pickle
from sklearn.linear_model import LinearRegression


product = [[25], [30], [35], [40], [45],[50],[55],[60],[65]]
profit = [50000, 60000, 75000, 80000, 90000, 95000, 100000, 110000, 120000]

model = LinearRegression()
model.fit(product, profit)

# Save the trained model using pickle
with open('modelprofit.pkl', 'wb') as file:
    pickle.dump(model, file)
