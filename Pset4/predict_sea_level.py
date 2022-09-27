import PIL, PIL.Image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

import scipy.stats as st
#from scipy.interpolate import interp1d

class floodmapper:
    """
    Create a floodmapper instance, for a map in a specified bounding box and a particular mass gis dbf file.
    Parameters are:
        tl - (lat,lon) pair specifying the upper left corner of the map data to load

        br - (lat,lon) pair specifying the bottom right corner of the map data to load 

        z - Zoom factor of the map data to load
        
        dbf_path - Path to the dbf file listing properties to load
        
        load - Boolean specifying whether or not to download the map data from mapbox.  If false, mapbox files must 
        have been previously loaded.  If using this as a 6.0001 problem set, you should have received pre-rendered map tiles
        such that it is OK to pass false to this when using the specific parameters in the __main__ code block below.
    """
    def __init__(self, tl, br, z, dbf_path, load):
        self.mtl = maptileloader.maptileloader(tl, br, z)
        dbf = massdbfloader.massdbfloader(dbf_path)
        if (load):
            self.mtl.download_tiles()
        self.pts = dbf.get_points_in_box(tl,br)
        self.ul, self.br = self.mtl.get_tile_extents()
        self.elevs = self.mtl.get_elevation_array()

    """
    Return a rendering as a PIL image of the map where properties below elev are highlighted
    """
    def get_image_for_elev(self, elev):
        fnt = ImageFont.truetype("Arial.ttf", 80)
        im = self.mtl.get_satellite_image()
        draw = ImageDraw.Draw(im)
        draw.text((10,10), f"{elev} meters", font=fnt, fill=(255,255,255,128))
        for name, lat, lon, val in self.pts:
            # print(name)
            x = int((((lon - self.ul[0]) / (self.br[0] - self.ul[0]))) * self.elevs.shape[1])
            y = int((((lat - self.ul[1]) / (self.br[1] - self.ul[1]))) * self.elevs.shape[0])
            # print(x,y)
            # print(e[x,y])
            el = int(self.elevs[y,x]*15)
            #print(e[y,x])
            c = f"rgb(0,{el},200)"
            if (self.elevs[y,x] < elev):
                c = f"rgb(255,0,0)"
            draw.ellipse((x-3,y-3,x+3,y+3), PIL.ImageColor.getrgb(c))
        return im

    """
    Return an array of (property name, lat, lon, elevation (m), value (USD)) tuples where properties
    are below the specified elev.
    """
    def get_properties_below_elev(self, elev):
        out = []
        for name, lat, lon, val in self.pts:
            x = int((((lon - self.ul[0]) / (self.br[0] - self.ul[0]))) * self.elevs.shape[1])
            y = int((((lat - self.ul[1]) / (self.br[1] - self.ul[1]))) * self.elevs.shape[0])
            if (self.elevs[y,x] < elev):
                out.append((name,lat,lon, self.elevs[y,x], val))

        return out

#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper - mean) / st.norm.ppf(.975)


def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)


def load_slc_data():
    """
	Loads data from sea_level_change.csv and puts it into numpy arrays

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year', 'Lower', 'Upper']
    return (df.Year.to_numpy(), df.Lower.to_numpy(), df.Upper.to_numpy())

###################
# End helper code #
###################

##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""
    # intialization section
    interp_results = np.zeros((81,5))   # dimensions of the interpolated results
    year_index = 0      # initializes index of already interp years between 2020 and 2100

    sl_data = load_slc_data()   # loads in the sea level data
    years = sl_data[0]          # years are decades from 2020 to 2100
    sl_low = sl_data[1]         # lower end of sea levels rising
    sl_high = sl_data[2]        # higher end of sea levels rising
    
    # loops through each year from 2020 to 2100
    for year in range(2020, 2101):
        index = year-2020 # index can range from 0 to 80
        interp_results[index, 0] = year

        # checks if year in the array of years(decades from 2020 to 2100) with data
        if year in years:            
            low_result = sl_low[year_index]
            high_result = sl_high[year_index]
            year_index+=1   # increments the index to move to next already calculated year
            
        # checks if the year has no data
        else:
            low_result = interp(year, years, sl_low)
            high_result = interp(year, years, sl_high)
            
        mean = (low_result+high_result)/2
        std = calculate_std(high_result, mean)
        
        # updates the interpolated results numpy array
        interp_results[index, 1] = mean
        interp_results[index, 2] = low_result
        interp_results[index, 3] = high_result
        interp_results[index, 4] = std
     
    # checks if you need tho plot the results
    if show_plot:
        # plot values
        x_values = interp_results[:, 0]     # all years (2020 to 2100)
        y_up = interp_results[:, 3]         # all upper slr interp values
        y_low = interp_results[:, 2]        # all lower slr interp values
        y_mean = interp_results[:, 1]       # all mean(average of low_slr and high_slr) slr values
        
        # plot labels
        title_label = "Time-series of projected annual average water level and 95% confidence interval"
        x_label = "Year"
        y_label = "Projected annual mean water level (ft)"
        legend_label = ("Upper", "Lower", "Mean")
        
        # putting the values, title, and labels on the plot 
        plt.plot(x_values, y_up, "b--")
        plt.plot(x_values, y_low, "r--")
        plt.plot(x_values, y_mean, 'o-')
        plt.title(title_label)
        plt.legend(legend_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.show() # displays the plot
    
    # return a 2D numpy array with year, mean, 2.5 percentile, 97.5 percentile, and standard deviation
    return interp_results

def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    # loops through each row of our data
    for d in data:
        
        # checks if year we are trying to simulate
        if year == d[0]:
            mean = d[1] # gets the mean for that year
            std = d[4]  # gets the std for that year
    
    # returns a normal distribution (simulation) of slr for particular year
    return np.random.normal(mean, std, num)
    
def plot_mc_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    # intializes section
    title_label = "Simulated projected sea level change in feet from 2020 to 2100"
    y_label = "Relative Water Level Change (ft)"
    x_label = "Year"
    legend_label = ("Upper bound", "Lower bound", "Mean") 
    
    # iterates through each year
    for year in range(2020, 2101):
        # does a Monte Carlo simulation for each year, runs 500 times
        # plots a scatter plot of the Monte Carlo Simulation
        plt.scatter(year, simulate_year(data, year, 500), "r")
        
    x_values = data[:, 0]       # all years of data
    y_mean = data[:, 1]         # all the mean sea levels for each year
    y_slr_low = data[:, 2]
    y_slr_up = data[:, 3]
    
    plt.plot(x_values, y_slr_up, "m-")      # plots upper sea level rise for all the years
    plt.plot(x_values, y_slr_low, "g")      # plots lower sea level rise for all the year
    plt.plot(x_values, y_mean, "b--")       # plots mean sea level rise for all the years
    
    # putting legend, title, and labeling the axis    
    plt.title(title_label)    
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend_label)
    plt.show()

##########
# Part 2 #
##########

def water_level_est(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    # initialization section
    sim_water_lvl = []
    
    # iterates through each year to simulate water levels
    for year in range(2020, 2101):
        sim_val = simulate_year(data, year, 1)  # gets a numpy array of all simulated sea level
        sim_water_lvl.append(np.mean(sim_val))  # adds the mean of the simulated water level to list
    
    # returns a list of simulated water levels for each year
    return sim_water_lvl

def calculate_prop_damage(water_lvl, water_level_loss_array, house_value):
    """
    Helper function created to calculate the property damage for one year.
    
    Args:
        water_lvl: the water level for one year
        water_level_loss_array: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention or with flood prevention
            (as an integer percentage), depending on the 2-d numpy array passed in
        house_value: the value of the property we are estimating cost for
    Returns:
        a float of the property damage in 1000s for that year
    """
    # checks case 1, if water level is less than or equal to 5 feet
    if water_lvl <= 5:
        prop_percent = 0    # no property damage
    
    # checks case 2, if water level is between 5 and 10 feet
    elif water_lvl > 5 and water_lvl < 10:
        # if the water level is an int 
        if type(water_lvl) == int:
            index = water_lvl - 5
            prop_percent = (water_level_loss_array[index, 1])/100
    
        # if water level is not an integer
        else:
            # target_years - > water_lvl
            # input_years - > all the water levels you want to interpolate on (x-values)
            # input_data - > property damage loss % (y-values)
            # gets the property loss percentage
            prop_percent = (interp(water_lvl, water_level_loss_array[:, 0], water_level_loss_array[:, 1]))/100
    
    # checks case 3, if water level is greater than or equal to 10 feet
    else:
        prop_percent = 1
    
    # returns the property damaged incurred in one year, in 1000s
    return (house_value*prop_percent)/1000  # calculates the property damage in the 1000s

def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    # intialization section
    prop_percent = 0
    prop_cst = 0
    damage_cst = []
    
    # iterate through all simulated water levels for 2020-2100
    for water_lvl in water_level_list:
        prop_cst = calculate_prop_damage(water_lvl, water_level_loss_no_prevention, house_value)
        damage_cst.append(prop_cst) # adds property damage for that year to list of all property damage costs
        
    # returns a list of all the damage cost for each year
    return damage_cst 


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
               cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    # intialization section
    prevention_bool = False
    prop_cst = 0
    damage_cst = []
    
    # iterates through all the water levels 
    for water_lvl in water_level_list:
        # checks if should use with prevention
        if prevention_bool:
            prop_cst = calculate_prop_damage(water_lvl, water_level_loss_with_prevention, house_value)  # calculates the property damage cost with prevention
            damage_cst.append(prop_cst) # adds the property damage cost for one year
 
        # no prevention will be used
        else:
            prop_cst = calculate_prop_damage(water_lvl, water_level_loss_no_prevention, house_value)    # calculates the property damage cost with no prevention
            damage_cst.append(prop_cst) # adds the property damage cost for one year
            
            # checks if the damage cost accrued over the years has reached or exceeded the cost threshold
            if prop_cst*1000 >= cost_threshold:
                prevention_bool = True  # will start using water level prevention
    
    # returns the list of property damage cost for each year from 2020 to 2100
    return damage_cst

def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    # returns a list of the property damage cost for all years between 2020 and 2100
    # with water level loss with prevention
    return repair_only(water_level_list, water_level_loss_with_prevention, house_value)
    
def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
                    cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    # intialization section
    repair_avg = []
    wait_avg = []
    immediate_avg = []
    repair_list_all = []
    wait_list_all = []
    immediate_list_all = []
    title_label = "Property damage cost comparison"
    x_label = "Year"
    y_label = "Estimated Damage Cost ($K)"
    legend_label = ("Repair-only Scenario", "Wait-a-bit Scenario", "Prepare-immediately Scenario")
    sum_repair = np.zeros(81)
    sum_wait = np.zeros(81)
    sum_immediate = np.zeros(81)
    
    # does a Monte Carlo simulation of the different preparation strategies and then plots them
    for i in range(500):
        water_level_list = water_level_est(data)    # calculates water level for each year
        repair_list = repair_only(water_level_list, water_level_loss_no_prevention, house_value)
        wait_list = wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold)                        
        immediate_list = prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value)
        plt.scatter(data[:, 0], repair_list, "b", label = "Repair-only Scenario")
        plt.scatter(data[:, 0], wait_list, "r", label = "Wait-a-bit Scenario")
        plt.scatter(data[:, 0], immediate_list, "y", label = "Prepare-immediately Scenario")   
        sum_repair = np.sum(sum_repair, repair_list)            # summing up all the simulations for repair
        sum_wait = np.sum(sum_wait, wait_list)                  # summing up all the simualtions for wait a bit
        sum_immediate = np.sum(sum_immediate, immediate_list)   # summing up all the simulation for immediate 
     
    # plot the scatter plot of 500 simulations of the 
    repair_avg = sum_repair/500
    wait_avg = sum_wait/500
    immediate_avg = sum_immediate/500
    
    # plotting the x and y values onto the plot
    plt.plot(data[:, 0], repair_avg, "k--")
    plt.plot(data[:, 0], wait_avg,"m--")
    plt.plot(data[:, 0], immediate_avg, "g--")
    
    # putting the titel, legend, and axis title labels onto the plot
    plt.title(title_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend_label)
    plt.show()
    
if __name__ == '__main__':
    
    # Comment out the 'pass' statement below to run the lines below it
    #pass

    import maptileloader
    import massdbfloader
    
    # # Uncomment the following lines to plot generate plots
    data = predicted_sea_level_rise()
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    plot_mc_simulation(data)
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
    
    # # Uncomment the following lines to visualize sea level rise over a map of Boston
    tl = (42.3586798 +.04, - 71.1000466 - .065)
    br = (42.3586798 -.02, - 71.1000466 + .065)
    dbf = 'cambridge_2021.dbf'
    fm = floodmapper(tl,br,14,dbf,False)

    print("Getting properties below 5m")
    pts = fm.get_properties_below_elev(5.0)
    print(f"First one: {pts[0]}")
    
    print("The next few steps may take a few seconds each.")

    fig, ax = plt.subplots(figsize=(12,10), dpi=144)
    
    ims=[]
    print("Generating image frames for different elevations")
    for el_cutoff in np.arange(0,15,.5):
         print(el_cutoff)
         im = fm.get_image_for_elev(el_cutoff)
         im_plt = ax.imshow(im, animated=True)
         if el_cutoff == 0:
             ax.imshow(im)  # show an initial one first

         ims.append([im_plt])

    print("Building animation")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                     repeat_delay=1000)
    print("Saving animation to animation.gif")
    ani.save('animation.gif', fps=30)

    print("Displaying Image")
    plt.show()
