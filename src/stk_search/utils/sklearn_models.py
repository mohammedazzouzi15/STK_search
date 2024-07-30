from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def plot_results(y_test,y_pred_test,y_train,y_pred_train,y_val,y_pred_val):
    """Plot results of the model
    Inputs: y_test, y_pred_test, y_train, y_pred_train, y_val, y_pred_val.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred_test,label="test")
    ax.scatter(y_train, y_pred_train,label="train")
    ax.scatter(y_val, y_pred_val,label="val")
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.legend()
    plt.show()
    # calculate MAE, MSE, RMSE


def train_test_model(model, X_train, y_train, X_test, y_test,X_val,y_val):
        """Function that trains a model, and tests it.
        Inputs: sklearn model, train_data, test_data.
        """
        # Train model
        model.fit(X_train, y_train)

        # Calculate RMSE on training
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        model_train_mse = mean_squared_error(y_train, y_pred_train)
        model_test_mse = mean_squared_error(y_test, y_pred_test)
        model_train_mse ** 0.5
        model_test_mse ** 0.5
        return y_pred_train, y_pred_test, y_pred_val,model

def train_model(X_rpr,y,min_test_set=-3):
    """Train a model using XGBoost
    Inputs: X_rpr, y, min_test_set
    Returns: y_train, y_test, y_val, y_pred_train, y_pred_test, y_pred_val, model.
    """
    X_test = X_rpr[y>min_test_set]#.detach().numpy()
    y_test = y[y>min_test_set]#.detach().numpy()

    X_train, X_val, y_train, y_val = train_test_split(X_rpr[y<min_test_set],y[y<min_test_set] , test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    # transform data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    xgb_reg = HistGradientBoostingRegressor(random_state=0)  # using 10 trees and seed=0
    #xgb_reg = GradientBoostingRegressor(n_estimators=50, random_state=0)  # using 10 trees and seed=0
    # Train and test XGBoost model
    y_pred_train, y_pred_test,y_pred_val,model = train_test_model(xgb_reg, X_train, y_train, X_test, y_test,X_val,y_val)
    plot_results(y_test,y_pred_test,y_train,y_pred_train,y_val,y_pred_val)
    return y_train,y_test,y_val,y_pred_train, y_pred_test,y_pred_val,model
