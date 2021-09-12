# Cyclistic_bike_EDA
The data was taken from Google Data Certificate capstone. Analysis done using python. Check the complete EDA on notebook file.

## Background
In 2016, Cyclistic launched a successful bike-share offering. Since then, the program has grown to a fleet of 5,824 bicycles that are geotracked and locked into a network of 692 stations across Chicago. The bikes can be unlocked from one station and returned to any other station in the system anytime.
Until now, Cyclistic’s marketing strategy relied on building general awareness and appealing to broad consumer segments. One approach that helped make these things possible was the flexibility of its pricing plans: single-ride passes, full-day passes, and annual memberships. Customers who purchase single-ride or full-day passes are referred to as casual riders. Customers who purchase annual memberships are Cyclistic members.
Cyclistic’s finance analysts have concluded that annual members are much more profitable than casual riders. Although the pricing flexibility helps Cyclistic attract more customers, the marketing director believes that maximizing the number of annual members will be key to future growth. Rather than creating a marketing campaign that targets all-new customers, she believes there is a very good chance to convert casual riders into members. She notes that casual riders are already aware of the Cyclistic program and have chosen Cyclistic for their mobility needs.
The clear goal has been set: Design marketing strategies aimed at converting casual riders into annual members. In order to do that, however, the marketing analyst team needs to better understand how annual members and casual riders differ, why casual riders would buy a membership, and how digital media could affect their marketing tactics. Marketing team are interested in analyzing the Cyclistic historical bike trip data to identify trends.

## Dataset
There are about ~5million data collected throughout 2020 to mid 2021. Initially data was splitted as separate csv file every month. The data has been combined to become a single csv file *'2020-2021_divvy_tripdata.csv'* and utilizing chunk operation to open the whole dataset for further analysis.

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
```

```python
# Start with read csv file using chunk

df = pd.read_csv('2020-2021_divvy_tripdata.csv', iterator=True, chunksize=1000)
df2 = []
for chunk in df:
    df2.append(chunk)

df2 = pd.concat(df2)
```

```python
# Further preliminary check 
df2.shape # There are 5,515,093 data row and 15 columns

df2.head(3)

# Check uniqueness of each data row
df2.nunique()

# Check any duplicate data. found none.
df2.duplicated().sum()

# Remove duplicate data, if any
#df2 = df2.drop_duplicates()
```

### Define context & dynamics variable:
From the preliminary check, all variables classified onto context and dynamic. This will help us to analyze the data further in a structured manner.
|Context variable:|Dynamic variable:                    |
|:----------------|:------------------------------------|
|ride_id          |started_at, ended_at                 |
|rideable_type    |start_station_name, end_station_name |
|member_casual    |start_station_id, end_station_id     |
|                 |start_lat, start_lng                 |
|                 |end_lat, end_lng                     |

Note:
- 'Unnamed: 0' & 'Unnamed: 0.1' doesnt provide any context<br>
- time constraint shall become another aspect to analyze out these classification.

```python
# Drop unused column 'Unnamed: 0', 'Unnamed: 0.1'
df2.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace=True)

# Check null data,it contains 435537 data with NaN (~7% of total data). Drop the null data.
df2.isnull().sum() 
df2.dropna(how='any',axis=0, inplace = True)
df2 = df2.reset_index(drop = True)
```
### Investigate context variable 'ride_id': Find out if there is any of rider book more than 1 time
```python
# Check if there is any duplicate of ride_id
df2[df2['ride_id'].duplicated()]

# Check how many non-duplicate of ride_id
df2[~df2['ride_id'].duplicated()]

# Count how many duplication of rider who book more than 1, then plot
dfa = df2['ride_id'].value_counts().reset_index(name='counts').query('counts > 1')

# Check if there is any duplicate of ride_id
df2[df2['ride_id'].duplicated()]['member_casual'].value_counts()

# Check how 'rideable_type' distributed among these 'loyal' rider
df2[df2['ride_id'].duplicated()]['rideable_type'].value_counts()
```
> #### *Key insight:*
> * From context variable, we found 208 'loyal' rider (0.004% of total rider throughout 2020-2021) book ride more than 1 (2 times)
> * 183 'loyal' rider are member (87% of 'loyal' rider) 
> * All rider select docked_bike for their ride.
> * 5,097,348 rider only 1 time off book.

```python

# Find out how many 'member' in one time off (1TO) rider and how 'rideable_type' distributed among them
df3 = df2[~df2['ride_id'].duplicated()]
df3.groupby(['member_casual']).size()

# There are 3,038,214 member among 1TO rider population
df3 = pd.pivot_table(df3, index=['rideable_type'], columns=['member_casual'], values = ['ride_id'], aggfunc='count')
df3['ride_id']

# Plot the ride type popularity among 1TO rider
fig = go.Figure()
fig = px.bar(df3['ride_id'], x = df3['ride_id'].index, y = df3['ride_id'].columns, barmode='group')
fig.update_layout(xaxis_title="Rideable_type",
                  yaxis_title="Ride count",
                  title={'text': "Ride type popularity among 1TO rider in 2020-2021",
                         'y':0.93,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'})
fig.show()
```
> #### *Key insight:*
> * From context variable, we found potential 2,041,134 1TO casual rider (40% of 1TO population) to become member
> * Docked_bike still the most popular ride selection among member and casual rider 
