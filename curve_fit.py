import numpy as np

def calc_median(x, y, points_per_bar=10):
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

