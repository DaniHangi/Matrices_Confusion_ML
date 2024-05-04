import pandas as pd 
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# Importing the dataset
df = pd.read_csv("Height-Weight.csv")

# To see first 5 observations (rows) of your data
df.head() 
#df.head(n)


#df1 = df[["weight", "height"]] or 
df1 = df.drop(['sex', 'repwt', 'repht'], axis = 1)

# df.describe(include = 'all')
#df.describe()
#df1.describe()



# Scatter Plot
# plt.scatter(df1.height, df1.weight)
# plt.xlabel("Height")
# plt.ylabel("Weight")
# plt.title("Scatter Plot")
# plt.show()


# Correlation Matrix
# df1.corr()


X = df1.height
y = df1.weight
# X.shape


X = X.values.reshape(len(X),1)
y = y.values.reshape(len(y),1)
# X = X.values.reshape(-1,1)
# print(X.shape, y.shape)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

# Saving model to disk
joblib.dump(regressor, open('model.joblib','wb'))


# Loading model to compare the results
model = joblib.load(open('model.joblib','rb'))
# print(model.predict([[178.00]]))


