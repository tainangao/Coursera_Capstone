#!/usr/bin/env python
# coding: utf-8

# <h1> PART 1 - PLOTTING THE APARTMENT SALES PRICE ON CHOROPLETH

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# In[2]:


CLIENT_ID = 'HMJSQ3GVV5LKNSTVIMD1DZYOS1E2WPOYUF43FCPWX4QNKLFW' # your Foursquare ID
CLIENT_SECRET = 'PWDCF25ERDO4YXDNO4IINJGEKEOBCJO2P0WSGINATUEJWIGR' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[3]:


df=pd.read_excel('rollingsales_manhattan.xls',skiprows=4)
df.head()


# In[88]:


df.columns


# <h2> Data Cleaning </h2>
# 
# <h3> Let's first replace 0 with NaN </h3>

# In[4]:


df['RESIDENTIAL UNITS'].replace(0,np.nan,inplace=True)
df['SALE PRICE'].replace(0,np.nan,inplace=True)
df['GROSS SQUARE FEET'].replace(0,np.nan,inplace=True)


# <h3> Check the null values </h3>

# In[5]:


missing_data = df.isnull()
missing_data.head(10)


# In[6]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# <h3> Keep the residential apartment data and rows with 'SALE PRICE' value </h3>
# 
# <b> The relationship between ['RESIDENTIAL UNITS'] and ['COMMERCIAL UNITS'] </b> <br>
# Generally, if the value in ['COMMERCIAL UNITS'] is null, that in ['RESIDENTIAL UNITS'] is not null, and vice versa. In another words, these 2 columns tell us if an apartment is (mainly) residential or commercial. Since we only want to analyze residential apartments, we will drop all rows containing commercial apartment data. 
# 
# In the meanwhile, SALE PRICE data are absent in some rows. Since price is our target variable, let's drow those rows.
# 

# In[7]:


# simply drop whole row with NaN in "RESIDENTIAL UNITS" and "SALE PRICE" column
df.dropna(subset=["RESIDENTIAL UNITS"], axis=0, inplace=True)
df.dropna(subset=["SALE PRICE"], axis=0, inplace=True)

# reset index, because we droped wo rows
df.reset_index(drop=True, inplace=True)


# In[8]:


missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# <h3> Eliminate outliers </h3>
# 
# Now that we got rid of all the null values, let's create our target column: ['PRICE PER SQUARE FOOT'], and also eliminate the outliers so that our Choropleth can be more representative.

# In[9]:


df['PRICE PER SQUARE FOOT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']


# In[10]:


df['PRICE PER SQUARE FOOT'].plot(kind='box')


# In[11]:


df['PRICE PER SQUARE FOOT'].describe(include='all')


# In[12]:


iqr = 1506.823351 - 402.500977
iqr


# In[13]:


outlier1 = 402.500977 - 1.5*iqr
outlier1


# In[14]:


outlier2 = 1506.823351 + 1.5*iqr
outlier2


# In[15]:


# Get names of indexes for outliers in column PRICE PER SQUARE FEET 
indexNames = df[ df['PRICE PER SQUARE FOOT'] > outlier2 ].index
 
# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)


# In[16]:


df['PRICE PER SQUARE FOOT'].plot(kind='box')


# In[17]:


df.info()


# <h3> Matching df['NEIGHBORHOOD'] and the neighborhoods in the GeoJson file </h3>
# 
# df['NEIGHBORHOOD'] is in fact different from the neighborhoods in the GeoJson file. The majority of neighborhoods in these 2 files are different while only a small number of them are exact match. In order to map the Choropleth correctly, we have to match each and every apartment in df with the neighborhood in the GeoJson file.
# 
# <h3> 1. Create a new column ['FULL ADDRESS'] </h3>

# In[18]:


df['NEIGHBORHOOD'] = df['NEIGHBORHOOD'].apply(lambda x: x.title())
df['NEIGHBORHOOD'].sample(5)


# In[19]:


df['FULL ADDRESS']=df['ADDRESS']+', '+df['NEIGHBORHOOD']+', Manhattan, NY, USA'
df['FULL ADDRESS'].sample(2)


# <h3> 2. Get geographic coordinates for all the apartments </h3>

# In[20]:


get_ipython().system(' pip install geocoder')


# In[21]:


import requests
import logging
import time

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


# In[22]:


# Set your Google API key here. 

# Even if using the free 2500 queries a day, its worth getting an API key since the rate limit is 50 / second.
# With API_KEY = None, you will run into a 2 second delay every 10 requests or so.
# With a "Google Maps Geocoding API" key from https://console.developers.google.com/apis/, 
# the daily limit will be 2500, but at a much faster rate.
# Example: API_KEY = 'AIzaSyC9azed9tLdjpZNjg2_kVePWvMIBq154eA'
API_KEY = 'AIzaSyCvBck-H6M_xSPAXbw9tAP8-aF-G2uwZJ0'
# Backoff time sets how many minutes to wait between google pings when your API limit is hit
BACKOFF_TIME = 30
# Set your output file name here.
output_filename = 'geo.csv'
# Set your input file here
input_filename = df
# Specify the column name in your input data that contains addresses here
address_column_name = "FULL ADDRESS"
# Return Full Google Results? If True, full JSON results from Google are included in output
RETURN_FULL_RESULTS = False


# In[23]:


# Make a big list of all of the addresses to be processed.
addresses = df['FULL ADDRESS'].tolist()


# In[23]:


def get_google_results(address, api_key=None, return_full_response=False):
    """
    Get geocode results from Google Maps Geocoding API.
    
    Note, that in the case of multiple google geocode reuslts, this function returns details of the FIRST result.
    
    @param address: String address as accurate as possible. For Example "18 Grafton Street, Dublin, Ireland"
    @param api_key: String API key if present from google. 
                    If supplied, requests will use your allowance from the Google API. If not, you
                    will be limited to the free usage of 2500 requests per day.
    @param return_full_response: Boolean to indicate if you'd like to return the full response from google. This
                    is useful if you'd like additional location details for storage or parsing later.
    """
    # Set up your Geocoding url
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json?address={}".format(address)
    if api_key is not None:
        geocode_url = geocode_url + "&key={}".format(api_key)
        
    # Ping google for the reuslts:
    results = requests.get(geocode_url)
    # Results will be in JSON format - convert to dict using requests functionality
    results = results.json()
    
    # if there's no results or an error, return empty results.
    if len(results['results']) == 0:
        output = {
            "formatted_address" : None,
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "google_place_id": None,
            "type": None,
            "postcode": None
        }
    else:    
        answer = results['results'][0]
        output = {
            "formatted_address" : answer.get('formatted_address'),
            "latitude": answer.get('geometry').get('location').get('lat'),
            "longitude": answer.get('geometry').get('location').get('lng'),
            "accuracy": answer.get('geometry').get('location_type'),
            "google_place_id": answer.get("place_id"),
            "type": ",".join(answer.get('types')),
            "postcode": ",".join([x['long_name'] for x in answer.get('address_components') 
                                  if 'postal_code' in x.get('types')])
        }
        
    # Append some other details:    
    output['input_string'] = address
    output['number_of_results'] = len(results['results'])
    output['status'] = results.get('status')
    if return_full_response is True:
        output['response'] = results
    
    return output


# # Don't run the following code!!
# 
# This block of code will extract geographic coordinates for all the apartments, however, it will run for 20-30 minutes everytime we restart the kernel. <br>
# 
# After running this block of code once, I have all the coordinates saved to a CSV file. 

# In[24]:



# Create a list to hold results
results = []
# Go through each address in turn
for address in addresses:
    # While the address geocoding is not finished:
    geocoded = False
    while geocoded is not True:
        # Geocode the address with google
        try:
            geocode_result = get_google_results(address, API_KEY, return_full_response=RETURN_FULL_RESULTS)
        except Exception as e:
            logger.exception(e)
            logger.error("Major error with {}".format(address))
            logger.error("Skipping!")
            geocoded = True
            
        # If we're over the API limit, backoff for a while and try again later.
        if geocode_result['status'] == 'OVER_QUERY_LIMIT':
            logger.info("Hit Query Limit! Backing off for a bit.")
            time.sleep(BACKOFF_TIME * 60) # sleep for 30 minutes
            geocoded = False
        else:
            # If we're ok with API use, save the results
            # Note that the results might be empty / non-ok - log this
            if geocode_result['status'] != 'OK':
                logger.warning("Error geocoding {}: {}".format(address, geocode_result['status']))
            logger.debug("Geocoded: {}: {}".format(address, geocode_result['status']))
            results.append(geocode_result)           
            geocoded = True

    # Print status every 100 addresses
    if len(results) % 100 == 0:
    	logger.info("Completed {} of {} address".format(len(results), len(addresses)))
            
    # Every 500 addresses, save progress to file(in case of a failure so you have something!)
    if len(results) % 500 == 0:
        pd.DataFrame(results).to_csv("{}_bak".format(output_filename))

# All done
logger.info("Finished geocoding all addresses")
# Write the full results to csv using the pandas library.
pd.DataFrame(results).to_csv(output_filename, encoding='utf8')


# <h3> 3. Import the CSV file generated and merge with df </h3>

# In[24]:


mn=pd.read_csv('Manhattan geo.csv')
mn.head()


# In[25]:


mn = mn[['input_string','latitude','longitude']]
mn.head()


# In[26]:


mn.rename(columns={'input_string':'FULL ADDRESS'}, inplace=True)
mn.info()


# In[27]:


dfa = df.drop_duplicates(subset=['FULL ADDRESS'])
dfb = mn.drop_duplicates(subset=['FULL ADDRESS'])

df1 = pd.merge(dfa, dfb, how='inner', on='FULL ADDRESS')


# In[28]:


df1.shape


# <h3> Moving on, we will be using df1 instead of df </h3>

# <h3> 4. Match each row with the correct neighborhood in the GeoJson file </h3>
# 
# 
# <h4> First - Set the GeoJson file to Manhattan only </h4>

# In[29]:


import json

with open('Neighborhood Tabulation Areas.geojson') as f:
    jsdata = json.load(f)


# In[30]:


# keep Manhattan only data

new_features = []
for element in jsdata["features"]:
    if 'Manhattan' in element['properties']['boro_name']:    
        new_features.append(element)                # new_features has the one's you want
# and then re-assign features to the list with the elements you want
jsdata["features"] = new_features


# In[31]:


# check if only the geojson file is Manhattan only

# assign relevant part of JSON to test
test = jsdata['features']

# tranform venues into a dataframe
dataframe = json_normalize(test)
dataframe['properties.boro_name'].value_counts()


# <h4> Second - Find out the neighborhoods for all apartments </h4>

# In[32]:


# !pip install shapely[vectorized]

from shapely.geometry import Point, shape
from shapely.geometry.polygon import Polygon

# shapely documentation 
# https://shapely.readthedocs.io/en/latest/manual.html


# In[33]:


results=[]
i=0

length = len(df1['FULL ADDRESS'])

while i < length:
    # construct point based on lat/long returned by geocoder
    point = Point(df1['longitude'][i], df1['latitude'][i])


    # check each polygon to see if it contains the point
    for feature in jsdata['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            results.append(feature)
            
    i=i+1


# In[34]:


results = pd.DataFrame(results)


# In[35]:


results.info()


# In[36]:


results.head()


# In[38]:


results1 = results['properties'].to_dict()
results1


# In[39]:


results2 = pd.DataFrame(results1) 


# In[40]:


results3= results2.transpose()
results3.head()


# In[41]:


results4 = pd.DataFrame(results3['ntaname']) 
results4.head()


# <h4> Finally - Merge the result and df1 </h4>

# In[42]:


frames = [df1, results4]

df2 = pd.concat(frames, sort=False, axis=1)


# <b> Check the result </b>

# In[43]:


df2.shape


# In[44]:


results4.info()


# In[45]:


df2.head()


# In[46]:


df2.tail()


# <h2> Choropleth </h2>
# 
# We are finally done with data cleaning. Let's plot the Choropleth!

# In[47]:


df2['PRICE PER SQUARE FOOT'].describe()


# In[48]:


# create a numpy array of length 6 and has linear spacing from the min to max
threshold_scale = np.linspace(df2['PRICE PER SQUARE FOOT'].min(),
                              df2['PRICE PER SQUARE FOOT'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the max price

# let Folium determine the scale.
manhattan_map = folium.Map(location=[40.7831, -73.9712], zoom_start=12)
manhattan_map.choropleth(
    geo_data=jsdata,
    data=df2,
    columns=['ntaname', 'PRICE PER SQUARE FOOT'],
    key_on='feature.properties.ntaname',
    threshold_scale=threshold_scale,
    fill_color= 'YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Apartment sales price in NY',
    reset=True
)
manhattan_map


# # Part 2 - CLUSTERING THE NEIGHBORHOODS

# ## 1. Download and Explore Dataset
# 
# New York City has a total of 5 boroughs and 306 neighborhoods. In order to segement the neighborhoods and explore them, we will essentially need a dataset that contains the 5 boroughs and the neighborhoods that exist in each borough as well as the the latitude and logitude coordinates of each neighborhood. 
# 
# Luckily, this dataset exists for free on the web. https://geo.nyu.edu/catalog/nyu_2451_34572

# In[49]:


get_ipython().system("wget -q -O 'newyork_data.json' https://cocl.us/new_york_dataset")
print('Data downloaded!')


# #### Load and explore the data

# In[50]:


with open('newyork_data.json') as json_data:
    newyork_data = json.load(json_data)


# Notice how all the relevant data is in the *features* key, which is basically a list of the neighborhoods. So, let's define a new variable that includes this data.

# In[51]:


neighborhoods_data = newyork_data['features']


# Let's take a look at the first item in this list.

# In[52]:


neighborhoods_data[0]


# #### Tranform the data into a *pandas* dataframe
# 
# The next task is essentially transforming this data of nested Python dictionaries into a *pandas* dataframe. So let's start by creating an empty dataframe.

# In[53]:


# define the dataframe columns
column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 

# instantiate the dataframe
neighborhoods = pd.DataFrame(columns=column_names)


# Take a look at the empty dataframe to confirm that the columns are as intended.

# In[54]:


neighborhoods


# Then let's loop through the data and fill the dataframe one row at a time.

# In[55]:


for data in neighborhoods_data:
    borough = neighborhood_name = data['properties']['borough'] 
    neighborhood_name = data['properties']['name']
        
    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]
    
    neighborhoods = neighborhoods.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)


# Quickly examine the resulting dataframe.

# In[56]:


neighborhoods.head()


# And make sure that the dataset has all 5 boroughs and 306 neighborhoods.

# In[57]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(neighborhoods['Borough'].unique()),
        neighborhoods.shape[0]
    )
)


# Slice the original dataframe and create a new dataframe of the Manhattan data.

# In[58]:


manhattan_data = neighborhoods[neighborhoods['Borough'] == 'Manhattan'].reset_index(drop=True)
manhattan_data.head()


# #### Use geopy library to get the latitude and longitude values of Manhattan.
# 
# In order to define an instance of the geocoder, we need to define a user_agent. We will name our agent <em>ny_explorer</em>, as shown below.

# In[59]:


address = 'Manhattan, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))


# #### Create a map of Manhattan with neighborhoods superimposed on top.

# In[60]:


# create map of Manhattan using latitude and longitude values
map_manhattan = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(manhattan_data['Latitude'], manhattan_data['Longitude'], manhattan_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_manhattan)  
    
map_manhattan


# #### Define Foursquare Credentials and Version

# In[61]:


CLIENT_ID = 'HMJSQ3GVV5LKNSTVIMD1DZYOS1E2WPOYUF43FCPWX4QNKLFW' # your Foursquare ID
CLIENT_SECRET = 'PWDCF25ERDO4YXDNO4IINJGEKEOBCJO2P0WSGINATUEJWIGR' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ## 2. Explore Neighborhoods in NYC
# 
# #### Let's create a function to get nearby venues for all the neighborhoods in Manhattan

# In[62]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[63]:


LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius


# #### Now write the code to run the above function on each neighborhood and create a new dataframe called *manhattan_venues*.

# In[64]:


manhattan_venues = getNearbyVenues(names=manhattan_data['Neighborhood'],
                                   latitudes=manhattan_data['Latitude'],
                                   longitudes=manhattan_data['Longitude']
                                  )


# #### Let's check the size of the resulting dataframe

# In[65]:


print(manhattan_venues.shape)
manhattan_venues.head()


# Let's check how many venues were returned for each neighborhood

# In[66]:


manhattan_venues.groupby('Neighborhood').count()


# #### Let's find out how many unique categories can be curated from all the returned venues

# In[67]:


print('There are {} uniques categories.'.format(len(manhattan_venues['Venue Category'].unique())))


# ## 3. Analyze Each Neighborhood

# In[68]:


# one hot encoding
manhattan_onehot = pd.get_dummies(manhattan_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
manhattan_onehot['Neighborhood'] = manhattan_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [manhattan_onehot.columns[-1]] + list(manhattan_onehot.columns[:-1])
manhattan_onehot = manhattan_onehot[fixed_columns]

manhattan_onehot.head()


# In[69]:


# examine the new dataframe size.
manhattan_onehot.shape


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
# 
# 

# In[70]:


manhattan_grouped = manhattan_onehot.groupby('Neighborhood').mean().reset_index()
manhattan_grouped


# #### Let's confirm the new size

# In[71]:


manhattan_grouped.shape


# #### Find out the most common venues in each neighborhood

# In[72]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[73]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = manhattan_grouped['Neighborhood']

for ind in np.arange(manhattan_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(manhattan_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## 4. Cluster Neighborhoods
# 
# <h4> First, find the best K </h4>

# In[74]:


manhattan_k=manhattan_grouped.drop('Neighborhood', axis=1)


# In[75]:


manhattan_k.head()


# In[76]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(manhattan_k)
    Sum_of_squared_distances.append(km.inertia_)


# In[77]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal K')
plt.show()


# #### Let's use k=5

# In[78]:


# set number of clusters
kclusters = 5

manhattan_grouped_clustering = manhattan_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(manhattan_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[79]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

manhattan_merged = manhattan_data

# merge nyc_grouped with nyc_data to add latitude/longitude for each neighborhood
manhattan_merged = manhattan_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

manhattan_merged.head() # check the last columns!


# In[80]:


manhattan_merged.dropna( subset=['Cluster Labels'], axis=0, how='any',inplace=True)


# Finally, let's visualize the resulting clusters

# In[81]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(manhattan_merged['Latitude'], manhattan_merged['Longitude'], manhattan_merged['Neighborhood'], manhattan_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)-1],
        fill=True,
        fill_color=rainbow[int(cluster)-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# # PART 3 - ANALYSIS
# 
# Now that we generated the Choropleth and clustered the neighborhoods, let's combine the first 2 parts!

# In[82]:


# add markers to the Choropleth
markers_colors = []
for lat, lon, poi, cluster in zip(manhattan_merged['Latitude'], manhattan_merged['Longitude'], manhattan_merged['Neighborhood'], manhattan_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)-1],
        fill=True,
        fill_color=rainbow[int(cluster)-1],
        fill_opacity=0.7).add_to(manhattan_map)
       
manhattan_map


# <h2> Examine and analyze clusters </h2>
# 
# Let's examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, we can then assign a name to each cluster. 

# #### Cluster 1 - Restaurants
# 
# This cluster has 7 neighborhoods and they are scarcely located in midtown and uptown. Compared to other clusters, the residential apartment sales price in this cluster is the cheapest. Except for Tudor City, the rest of the neighborhoods have an average for under \\$627 per square foot. The price in Tudor City, however, ranges from \\$1,254 to \\$1,882.
# 
# The most common venue in this cluster is restaurants  of different types, such as cafes, pizza places, Mexican restaurants, and Italian restaurants, to name just a few.

# In[83]:


manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 0, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]


# #### Cluster 2 - career
# 
# This cluster has 16 neighborhoods and is mainly located on the east side of the Manhattan island, occupying all the way from downtown to uptown. The residential apartment sales price ranges from free to \\$1,882 per square foot. Some of the most expensive neighborhoods in this cluster are Financial District, Chinatown, and East Villege.
# 
# Some of the most common venues in this cluster are: <br>
#  - Bars, where people hangout after work
#  - Coffee shops, which is a necessity for office workers
#  - Restaurants, including Chinese, Japanese, Italian, American, etc., satisfying  people's various interests
#  - Gyms, where office workers love going at least several times a week
# 
# Let's say this cluster is best for people who have high-paying jobs (perhaps  in Wall Street), and want to be socially  and physically fit.

# In[84]:


manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 1, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]


# #### Cluster 3 - Art
# 
# This cluster has 15 neighborhoods and is located in the downtown and midtown. Some renowned neighborhoods in it are Upper East Side, Midtown, Greenwich Village, SoHo, West Village, and Little Italy. The residential apartment sales price is the most expensive among all the neighborhoods. The majority in this neighborhood has a price per square foot from \\$1,882 to \\$2,509. 
# 
# Having Lincoln Square, Clinton, Upper East Side, and SoHo in this cluster, this cluster is the most artistic one among all. This cluster is good for people who value art, want to have a good social life, and like to work out. Some of the most common venues are:
# - Italian restaurants 
# - Theaters
# - Gyms
# - Clothing/fashion shops
# 

# In[85]:


manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 2, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]


# #### Cluster 4 - Stuyvesant Town
# 
# This cluster only has one neighborhood, Stuyvesant Town, a historical town. It offers quite a few venues for people to spend their leisure time - bars, playgrounds, and parks. It's a good neighborhood to sip a drink or play sports with friends.

# In[86]:


manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 3, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]


# #### Cluster 5 - Roosevelt Island
# 
# This cluster also has only one neighborhood, Roosevelt Island. As a small island just next by Manhattan island, Roosevelt island is a lot of people's choice when considering buying homes. Compared to other boroughs, the commute from Roosevelt Island to Manhattan island is much more convenient. The most common venues in this cluster also indicate that it's a good neighborhood to live a convenient and normal life. 
# 
# The apartment sales price ranges from \\$1,254 to \\$1,882 per square foot. 
# 
# 

# In[87]:


manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 4, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]

