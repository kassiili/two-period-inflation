import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

def median_trend(x, y, points_per_bar=10):
    xy = np.vstack([x,y])
    xy = xy[:,xy[0,:].argsort()]
    datapoints = xy[0,:].size
    bars = datapoints//points_per_bar

    #Excĺude first elements (if necessary) to allow reshaping:
    if (datapoints%bars != 0):
        tmpX = xy[0,(datapoints%bars):]
        tmpY = xy[1,(datapoints%bars):]
        datapoints -= datapoints%bars
    else:
        tmpX = xy[0,:]
        tmpY = xy[1,:]

    #Reshape into a 2D numpy array with number of rows = bars:
    tmpX = tmpX.reshape((bars, int(datapoints/bars)))
    tmpY = tmpY.reshape((bars, int(datapoints/bars)))

    #Calculate the medians of the rows of the reshaped arrays:
    medianX = np.median(tmpX, axis=1)
    medianY = np.median(tmpY, axis=1)

    return [medianX, medianY]


def error_bars():
    return None

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),\
            LinearRegression(**kwargs))

def poly_fit(x,y,max_deg=4):

    # Set parameter space:
    param_grid = {'polynomialfeatures__degree': np.arange(20),
            'linearregression__fit_intercept': [True, False],
            'linearregression__normalize': [True, False]}

    # Split to test and training:
    test_size = 0.3
    x_train, x_test, y_train, y_test = train_test_split(x, y,\
            test_size=test_size)

    # Find best parameter values:
    clf = GridSearchCV(PolynomialRegression(), param_grid, cv=5)
    clf.fit(x_train.reshape(-1,1),y_train)
    train_err = clf.best_score_
    test_err = clf.score(x_test.reshape(-1,1), y_test)

    # Construct model curve:
    model = clf.best_estimator_
    print(model)
    xfit = np.linspace(min(x),max(x),10000)
    yfit = model.fit(x.reshape(-1,1),y).predict(xfit.reshape(-1,1))

    return xfit,yfit,train_err,test_err

