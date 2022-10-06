import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = pd.read_csv("Real_estate.csv")
    # cleaning the data of unnecessary columns, for simplicity we are also removing booleans
    data = data.drop(['No', 'X1 transaction date', 'X5 latitude', 'X6 longitude'], axis=1)

    # replacing blank characters with

    data.fillna(0, inplace=True)

    # makes new csv of new data this code only needs to be run once, as the csv file will be stored in the same
    # directory

    data.to_csv("Real_estate_new.csv")
    dataClean = pd.read_csv("Real_estate_new.csv")

    # define X and Y
    X = dataClean.drop(['Y house price of unit area'], axis=1).values
    Y = dataClean['Y house price of unit area'].values

    # split the data into training and test sets
    # test size ensures that 25% of the values are used for the testing and
    # the rest of the values are used for training the model
    # by default the random_state is 0 so that the generation is deterministic
    # however we can add a random state value in the train_test_split function to change this if we so choose

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    # now we need to train the model on the training set

    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    # tries user input and loops until user enters int values

    while True:
        try:
            age = int(input("Please enter the age of the house: "))
            distance = int(input("Please enter the distance to the nearest MRT station (meters): "))
            num_stores = int(input("Please enter the number of convenience stores nearby "))
            # if roomChoice is a positive int exit the loop
            if isinstance(age, int) and isinstance(distance, int) and isinstance(num_stores, int):
                break
        except ValueError as e:  # raise a value error if the inputted values are not of type int

            print(f'An {e} error occurred! Please enter integer values!')

    # predicts the price based on the user inputs taken above
    pred_price = mlr.predict([[0, age, distance, num_stores]])

    print("The predicted house price in USD is: $", int(pred_price[0] * 320))

    # now we can check to see if our predictions are accurate
    # now we in order to evaluate the accuracy of the model we can evaluate using the r2 score
    pred_y = mlr.predict(x_test)

    print("The R Squared (R2) Score is:", r2_score(y_test, pred_y))
    print("Thus, the goodness of fit is not very strong however,"
          "\nthis can be due to the limitations of linear regression as well as the data itself shown in the plot")

    # now we will visualize the data as a scatter plot

    plt.figure(figsize=(15, 10))  # sets the dimensions of the plot

    plt.scatter(y_test, pred_y)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()