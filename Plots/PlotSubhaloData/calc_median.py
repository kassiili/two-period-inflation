import numpy as np

def calc_median_trend(x, y, bars=10):
    xy = np.vstack([x,y])
    xy = xy[:,xy[0,:].argsort()]
    datapoints = xy[0,:].size

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

#def calc_median_trend(x, y):
#    minX = min(x)
#    maxX = max(x)
#    bars = 10
#    tmpX = np.linspace(minX, maxX, num=bars)
#    medianTrend = np.empty([bars-1,])
#    for i in range(bars-1):
#        print(y[np.logical_and(tmpX[i] < x, x < tmpX[i+1])], '\n')
#        medianTrend[i] = np.median(y[np.logical_and(tmpX[i] < x, x < tmpX[i+1])])
#
#    return [tmpX[:-1] + (maxX-minX)/(2*bars), medianTrend] 

