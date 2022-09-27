import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re
from sklearn.cluster import KMeans

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

# KMeans class not required until Problem 7
class KMeansClustering(KMeans):

    def __init__(self, data, k):
        super().__init__(n_clusters=k, random_state=0)
        self.fit(data)
        self.labels = self.predict(data)

    def get_centroids(self):
        'return np array of shape (n_clusters, n_features) representing the cluster centers'
        return self.cluster_centers_

    def get_labels(self):
        'Predict the closest cluster each sample in data belongs to. returns an np array of shape (samples,)'
        return self.labels

    def total_inertia(self):
        'returns the total inertia of all clusters, rounded to 4 decimal points'
        return round(self.inertia_, 4)



class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """
        # intialization section
        avg_temp = []
        
        # iterates through all the years
        for i in range(len(years)):
            year = years[i]
            #city_daily = np.array([])   # resets list of all temps from all cities for given year
            city_daily = []
                
            # iterates through all the cities
            for i in range(len(cities)):
                city = cities[i]
                # combines all the daily temps from each city for given year
                city_daily.append(self.get_daily_temps(city, year))   
                
                """
                Ask for help on combining numpy arrays of different dimensions
                """
            avg_temp.append(np.mean(city_daily))   # calculates the average temp for given year
                
        # returns the average annual temperature for each city for a given year 
        return np.array(avg_temp)

def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    # initializaton section
    m = 0
    b = 0
    num_sum = np.zeros(len(x))      # np array of each element of numerator of slope calculation
    denom_sum = np.zeros(len(x))    # np array of each element of denominator of slope calculation
    
    x_avg = np.mean(x)  # average x-values
    y_avg = np.mean(y)  # average y-values
    
    # iterates through each sample point in the 1D numpy array
    for i in range(len(x)):
        x_val = x[i]
        y_val = y[i]
        num_sum[i] = (x_val-x_avg)*(y_val - y_avg)  # calculates element i of numerator summation
        denom_sum[i] = (x_val-x_avg)**2             # calculates element i of denominator summation 
    
    tot_num_sum = np.sum(num_sum)       # size 0D array, returns a scalar value
    tot_denom_sum = np.sum(denom_sum)   # size 0D array, returns a scalar value   
    m = tot_num_sum/tot_denom_sum   # slope for the line of best fit
    b = y_avg - (m*x_avg)           # y-int for the line of best fit
    
    # returns a tuple of the slope, y-int of the line of best fit
    return (m, b)

def squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line

    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    # intialization section
    sum_error = 0
    sq_err = 0.0
    
    # iterates through each x value
    for i in range(len(x)):
        x_val = x[i]
        y_val = y[i]
        y_pred = m*x_val + b            # calculates the predicted y-value using regression line equation
        sum_error+= (y_val - y_pred)**2   # summation of sq of error of actual - predicted
            
    # returns squared error of linear regression model for given data points
    return sum_error   

def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # initialization section
    poly_array = [] 
    
    # iterates through each degree, 
    for index in range(len(degrees)):
        deg = degrees[index]
        poly_array.append(np.polyfit(x, y, deg))    # calculates coefficients of 
                                                    # polynomial regression model
                                                    # of specific degree, adds to list
        
    # returns a list of np array of polynomial regression models
    return poly_array    

def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    # intialization section
    r_sq = []
    all_y_pred = []
    
    # iterates through each regression model
    for model in models:
        y_pred = np.polyval(model,x)        # calculate y-predicted values for each model
        r2 = r2_score(y, y_pred)            # calculates r squared values for specific model
        round_r2 = round(r2,4)              # rounds R-squared value
        r_sq.append(round_r2)                
        
        # checks if the data should be plotted
        if display_graphs:
            title_label = f"The R^2 {round_r2} model with degree {len(model)-1}"
            
            # checks if linear regression model
            if len(model) == 2:
                SE_slope = standard_error_over_slope(x, y, y_pred, model)   # calculates STD over slope
                round_SE_slope = round(SE_slope, 4) 
                title_label += f"and ratio of SE/slope {round_SE_slope} value"
            
            # plotting the values
            plt.plot(x, y, "b.")        # plots blue dots of the actual x,y values
            plt.plot(x, y_pred, "r")    # plots red solid line of regression model points
            
            # labeling the graph
            plt.xlabel("Year")
            plt.ylabel("Temperature for given year (Celsius)")
            plt.titlelabel(title_label)
            
            plt.show()  # displays plot
            
    # returns the R-squared values for each model
    return r_sq

def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    # intialization section
    slope_dict = {}
    max_slope = 0
    min_slope = 0
   
    # iterates through every possible x,y value in interval
    for i in range(len(x)-length+1):
        slope = linear_regression(x[i:i+length], y[i:i+length])[0]    # calculates the slope on specificed interval
        slope_dict[(i, i+length)] = slope  # updates dictionary w/ slopes w/in specified interval
          
    # checks if none of slopes are w/in interval
    if slope_dict == {}:
        # returns None
        return None
    
    # checks if finding max positive slope
    if positive_slope:
        max_slope = max(slope_dict.values())
        
        # checks if there is no max slope or slope is negative
        if max_slope <= 0:
            # returns none, no interval with max pos slope
            return None
        
    # checks if finding max negative slope
    else:
        max_slope = min(slope_dict.values())
        
        # checks if there is no max slope or slope is positive
        if max_slope >= 0:
            # returns none, no interval with max neg slope
            return None
    
    # iterates through each slope to find most positive or negative
    for key, value in slope_dict.items():
        
        # checks if we need to find positive slope
        if value == max_slope:
            # returns the max slope for the positive interval
            return (key[0], key[1], max_slope)
    
def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    # intialization
    interval_list = []
    epsilon = 1e-8
    
    # checks if returns an empty list
    if len(x) < 2:
        return []
    
    for length in range(2, len(x)+1):
        pos_slope = get_max_trend(x, y, length, True)
        neg_slope = get_max_trend(x, y, length, False)
        #isNone_pos = False
        #isNone_neg = False
        
        # checks if there is no extreme positive slope
        if pos_slope is None and neg_slope is None:
            interval_list.append((0, length, None)) # no extreme slope for this interval  
        
        # checks if there is only a positive slope
        elif neg_slope is None:
            interval_list.append(pos_slope)
        
        # checks if there is only a negative slope
        elif pos_slope is None:
            interval_list.append(neg_slope)
            
        # checks if negative slope more extreme
        elif (pos_slope[2] - abs(neg_slope[2]) < epsilon):
            interval_list.append(neg_slope)
        
        # checks if positive slope more extreme
        else:
            interval_list.append(pos_slope)
       
    # returns the intervals of most extreme slope in increasing interval lenths
    return interval_list

def calculate_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    # intialization section
    rmse = 0.0
    num_sum = 0
    
    # iterates through each actual and predicted y value
    for i in range(len(y)):
        num_sum+= (y[i] - estimated[i])**2
        
    rsme = np.sqrt(num_sum/len(y))
    
    # returns the root mean square error term
    return rsme

def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    # intialization section
    all_rsme = []
    
    # iterates through each model
    for model in models:
        y_pred = np.polyval(model, x)
        rsme = round(calculate_rmse(y, y_pred), 4)    # calculates the rsme value
        all_rsme.append(rsme)
       
        # checks if data should be plotted
        if display_graphs:
            title_label = f"The RSME {rsme} model with degree {len(model)-1}"
            x_label = "Year"
            y_label = "Temperature for given year (Celsius)"
            
            # plotting the data
            plt.plot(x, y, "b.")
            plt.plot(x, y_pred, "r")
            
            # labeling the graph
            plt.title(title_label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            
            # show plot
            plt.show()
        
    # returns the rsme values for each model
    return all_rsme

def cluster_cities(cities, years, data, n_clusters):
    '''
    Clusters cities into n_clusters clusters using their average daily temperatures
    across all years in years. Generates a line plot with the average daily temperatures
    for each city. Each cluster of cities should have a different color for its
    respective plots.

    Args:
        cities: a list of the names of cities to include in the average
                daily temperature calculations
        years: a list of years to include in the average daily
                temperature calculations
        data: a Dataset instance
        n_clusters: an int representing the number of clusters to use for k-means

    Note that this part has no test cases, but you will be expected to show and explain
    your plots during your checkoff
    '''
    # intialization section
    all_city_avg_day = []       # list of np arrays    
    
    # iterates through each city
    for i in range(len(cities)):
        city = cities[i]
        all_daily_temp = []
        
        # iterates through each year
        for year in years:
            daily_temp = data.get_daily_temps(city, year)   # gets all daily temp for one year
            
            # checks if leap year
            if len(daily_temp) == 366:
                daily_temp = np.delete(daily_temp, 365) # gets rid of temperature from last day
                
            all_daily_temp.append(daily_temp)   # adds the daily temp for that year
        
        all_daily_temp_arr = np.array(all_daily_temp)           
        avg_daily_temp = np.mean(all_daily_temp_arr, axis=0)    # calculates average temp for each day from 1961 to 2016
        all_city_avg_day.append(avg_daily_temp)                 # adds each cities' average temperature for a day from 1961 to 2016
    
    kmeans = KMeans(n_clusters)                        # instantiates an instance of kmeans object       
    clusters = kmeans.fit(all_city_avg_day)            # gets clusters for average daily temp for each city
    labels = clusters.labels_                          # gets labels for all the clusters
    
    # iterates through all the cities
    for i in range(len(cities)):
        
        # checks if label for city is 0
        if labels[i] == 0:
            plt.plot(years, all_city_avg_day[i], "r")   # plots a red line for cluster one
        # checks if label for city is 1
        elif labels[i] == 1:
            plt.plot(years, all_city_avg_day[i], "b")   # plots a blue line for cluster two
        # checks if label for city is 2
        elif labels[i] == 2:
            plt.plot(years, all_city_avg_day[i], "g")   # plots a green line for cluster three
        # checks if label for city is 3
        else:
            plt.plot(years, all_city_avg_day[i], "m")   # plots a magenta line for cluster four
        
    # labeling the graph
    plt.title("The Clustering of the Daily Average Temperatures from 1961 to 2016 for each city")
    plt.xlabel("Years")
    plt.ylabel("Average Daily Temperature(Celsius)")
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    x = np.zeros(56)    # array that will have all years from 1961 to 2016
    y1 = np.zeros(56)    # array that will have all the temperatures for Dec 1st from 1961 to 2016
    
    # instantiation section
    data = Dataset("data.csv")
    
    # iterates through each year from 1961 to 2016
    for year in range(1961, 2017): 
        x[year-1961] = year
        y1[year-1961] = data.get_temp_on_date("BOSTON", 12, 1, year)
            
    models = generate_polynomial_models(x, y1, [1])    # generates degree one ploynomial regression model
    evaluate_models(x, y1, models, True)               # evaluates and plots the degree one polynomial regression model
    
    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    y2 = data.calculate_annual_temp_averages(["BOSTON"], range(1961, 2017))
    models = generate_polynomial_models(x, y2, [1])    # generates degree one ploynomial regression model
    evaluate_models(x, y2, models, True)               # evaluates and plots the degree one polynomial regression model
    
    ##################################################################################
    # Problem 5B: INCREASING TRENDS
    y3 = data.calculate_annual_temp_averages(["SEATTLE"], range(1961, 2017))
    interval = get_max_trend(x, y3, 30, True)
    low_index = interval[0]
    up_index = interval[1]
    models = generate_polynomial_models(x[low_index:up_index], y3[low_index:up_index], [1])
    evaluate_models(x, y3, models, True)
    
    ##################################################################################
    # Problem 5C: DECREASING TRENDS
    y4 = data.calculate_annual_temp_averages(["SEATTLE"], range(1961, 2017))
    interval = get_max_trend(x, y3, 15, False)
    low_index = interval[0]
    up_index = interval[1]
    models = generate_polynomial_models(x[low_index:up_index], y4[low_index:up_index], [1])
    evaluate_models(x, y3, models, True)

    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.

    ##################################################################################
    # Problem 6B: PREDICTING
    # Generating model based on the training data set
    years1 = [int (i) for i in range(1961, 2000)]
    x2 = np.array(years1)
    y5 = data.calculate_annual_temp_averages(CITIES, years1)
    models = generate_polynomial_models(x2, y5, [2, 10]) # creates a linear regression model of training data set
    evaluate_models(x2, y5, models, True)
    
    # Testing model based on the training data set
    years2 = [int (j) for j in range(2000, 2017)]
    x3 = np.array(years2)
    y6 = data.calculate_annual_temp_averages(CITIES, years2)

    evaluate_rmse(x3, y6, models, True) # plots the rsme of actual and predicted national average daily temperature
    
    ##################################################################################
    # Problem 7: KMEANS CLUSTERING (Checkoff Question Only)
    ####################################################################################
    cluster_cities(CITIES, range(1961, 2017), data, 4)
