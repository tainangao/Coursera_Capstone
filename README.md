# Coursera_Capstone
IBM Data Science Professional Certificate capstone project

**Please click here to check out the [Jupyter notebook](https://nbviewer.jupyter.org/github/tainangao/Coursera_Capstone/blob/master/Neighborhoods-in-Manhattan.ipynb)**, as the .ipynb file uploaded is too large to load on GitHub.

I also wrote a [blog post](https://medium.com/@jacqueline3749/residential-apartment-sales-price-and-venues-analysis-of-manhattan-1a0ab261d5a9) about this project (photo taken by me =))


## Residential Apartment Sales Price and Venues Analysis of Manhattan

When people consider buying residential apartments, what they often take into consideration are the price and what the neighborhood has to offer. In this project, I will
1. plot a Choropleth using Folium library, JSON files, Google Geocoding API, and Shapely library
2. cluster all the neighborhoods in Manhattan into 5 clusters using Folium library, Foursquare API, GeoPy library, and K-means




### Analysis Process
#### Part 1: Choropleth
1. import `rollingsales_manhattan.xls` as `df`
2. drop commercial apartments and apartments without price
3. eliminate outliers
4. match the neighborhoods in `df` and **GeoJSON** file
   - create new column `df['FULL ADDRESS']`
   - get geo-coordinates for all apartments using **Google API**
   - name the new dataframe as `df1`
   - match each apartment with the neighborhood in GeoJSON file
      - import `Neighborhood Tabulation Areas.geojson` and only keep Manhattan data 
      - find out the GeoJSON neighborhood for each department using **Shapely library**
         *Shapely detects if a coordinate is in a polygon *
      - name the new dataframe as `df2`
5. plot the choropleth using **Folium library**

#### Part 2: Clustering
1. download and explore `newyork_data.json`
   - load and explore the data. Create a dataframe called `neighborhoods`
   - slice `neighborhoods` and create a new dataframe of the Manhattan data, called `manhattan_data`
   - get the latitude and longitude values of Manhattan using **GeoPy library**
   - create a map of Manhattan with neighborhoods superimposed on top using **Folium library**
2. explore neighborhoods in Manhattan
   - get nearby venues for all the neighborhoods in Manhattan using **Foursquare API**. 
     Name the dataframe as `manhattan_venues`. It has 3328 venues and 339 unique categories
3. analyze each neighborhood
   - perform one hot encoding and create a dataframe called `manhattan_onehot`
   - group rows by neighborhood and take the mean frequency of occurrence of each category. 
     Create a dataframe called `manhattan_grouped`
   - find out the most common venues in each neighborhood
     create a dataframe called `neighborhoods_venues_sorted`
4. cluster neighborhoods using **K-means**
   - find the best k
   - create a new dataframe that includes the cluster and the top 10 venues for each neighborhood
     name it `manhattan_merged`
   - visualize the resulting clusters using **Folium library**
  
#### Part 3: Analysis
1. combine the Choropleth and the cluster map
2. examine and analyze clusters
   - Cluster 1 - Restaurants
   - Cluster 2 - Career
   - Cluster 3 - Art
   - Cluster 4 - Stuyvesant Town
   - Cluster 5 - Roosevelt Island

     
