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
![](https://github.com/yudhi-chen/cyclistic_bike_EDA/blob/main/Images/fig1-ride_type_popularity.png)

> #### *Key insight:*
> * *From context variable, we found potential 2,041,134 1TO casual rider (40% of 1TO population) to become member.*
> * *Docked_bike still the most popular ride selection among member and casual rider.* 

### Investigate dynamic variable
#### Check the start and end date if any of data 'ended_at' < 'started_at'

```python
# Assign datetime data type to 2 columns
df2['started_at'] = pd.to_datetime(df2['started_at'])
df2['ended_at'] = pd.to_datetime(df2['ended_at'])

# Check the datetime data if there's any data where 'ended_at' < 'started_at'
df2[df2['ended_at'] < df2['started_at']].value_counts().sum()

# Considering the total amount 10,579 data found (~0.2% of total data), safe to drop the data
df2.drop(df2[df2['ended_at'] < df2['started_at']].index, inplace=True, axis = 0)
df2 = df2.reset_index(drop = True)
df2.nunique()

# Notice there are huge difference 'start_station_name' and 'start_station_id' and this variable also correlate with GPS coordinate 'start_lat' & 'start_lng'. 
# Perform random check in one of station name.
df2['start_station_name'].value_counts()

# Select station name: Streeter Dr & Grand Ave and perform further check
df2[df2['start_station_name'] == 'Streeter Dr & Grand Ave']['start_station_id'].value_counts()
#df2[(df2['start_station_name'] == 'Streeter Dr & Grand Ave') & (df2['start_station_id'] == 35)].tail(3)
#df2[(df2['start_station_name'] == 'Streeter Dr & Grand Ave') & (df2['start_station_id'] == 35.0)].head(3)
#df2[(df2['start_station_name'] == 'Streeter Dr & Grand Ave') & (df2['start_station_id'] == '13022')].head(3)
```
> *We found that few discrepancy on GPS coordinate that causes variation in one station name. One case highlighted in station 'Streeter Dr & Grand Ave' can have various coordinate:*
> * *start_lat: 41.892278, 41.8923, 41.892278*
> * *start_lng: -87.612043, -87.612, -87.612043*

#### Find the most crowded station
```python
# Perform analysis using station name. Sort the 'start_station_name' based on frequency and map it
# Count the frequency of start station name
df4 = df2['start_station_name'].value_counts().to_frame()

# Filter the required variable station name and GPS location
df4a = df2[['start_station_name','start_lat','start_lng']]

# Keep non-duplicated first value appear on 'start_station_name' and reset the index
df4a = df4a[~df4a['start_station_name'].duplicated(keep = 'first')].reset_index(drop = True)

# Re-index the frequency based on 'start_station_name' and reset the index
df4 = df4.reindex(df4a['start_station_name']).rename(columns={'start_station_name': 'freq'}).reset_index(drop=True)

# Concatenate two dataframe
df4 = pd.concat([df4a,df4], axis=1)

# Sort station name value based on frequency
df4.sort_values('freq',ascending=False)

# Check the crowd based on binning
df4['marker_color'] = pd.cut(df4['freq'], bins=5, 
                              labels=['yellow', 'green', 'blue', 'magenta','red'])
                              
# Filter the station name with color magenta and above (red)
df4[df4['marker_color'] >= 'magenta'].sort_values('freq')                              
```
```python
# Perform similar analysis for end station name
# Count the frequency of end station name
df5 = df2['end_station_name'].value_counts().to_frame()

# Filter the required variable station name and GPS location
df5a = df2[['end_station_name','end_lat','end_lng']]

# Keep non-duplicated first value appear on 'end_station_name' and reset the index
df5a = df5a[~df5a['end_station_name'].duplicated(keep = 'first')].reset_index(drop = True)

# Re-index the frequency based on 'end_station_name' and reset the index
df5 = df5.reindex(df5a['end_station_name']).rename(columns={'end_station_name': 'freq'}).reset_index(drop=True)

# Concatenate two dataframe
df5 = pd.concat([df5a,df5], axis=1)

# Sort station name value based on frequency
df5.sort_values('freq',ascending=False)

# Check the crowd based on binning
df5['marker_color'] = pd.cut(df5['freq'], bins=5, 
                              labels=['yellow', 'green', 'blue', 'magenta','red'])

# Filter the station name with color magenta and above (red)
df5[df5['marker_color'] >= 'magenta'].sort_values('freq')
```
> #### *Key insight:*
> *These analysis shows there are 6 potential places to approach rider from both start and end trip of riders based on station name.*
> *Apparently, five stations mentioned in start and end station name worth to consider:*
> * *Millennium Park*
> * *Theater on the Lake*
> * *Lake Shore Dr & North Blvd*
> * *Clark St & Elm St*
> * *Streeter Dr & Grand Ave*

#### Investigate commute time
```python
# Create df6 as a copy of df2
df6 = df2.copy()

# Calculate another variable time commute
df6['t_comm'] = df6.loc[:,'ended_at'] - df6.loc[:,'started_at']

# Convert 't_comm' variable into hours commute
df6['t_comm'] = df6['t_comm'].dt.total_seconds() / 3600

# Quick check the population based on commute time
df6['t_comm'].value_counts(bins = 5)

# What is the typical commute hour for both member and casual rider?
#df6[(df6['t_comm']>521) & (df6['member_casual'] == 'member')].describe()
df6[(df6['t_comm']>521) & (df6['member_casual'] == 'casual')].describe()

# There are 121 riders commute > 521 hour, only 7 rider are member. 112 are casual rider.
# Pivot table of typical rider (commute time < 521 hour) check on 'rideable_type' classified by 'member_casual'
df6[df6['t_comm']<521].pivot_table(index = 'rideable_type', columns = ['member_casual'], 
                                   aggfunc = {'member_casual':'count'})

# Calculate statistics of total casual or member rider with t_comm < 521 hour
#df6[(df6['t_comm']<521) & (df6['member_casual'] == 'casual')].describe()
#df6[(df6['t_comm']<521) & (df6['member_casual'] == 'member')].describe()

# Pivot table of typical rider (commute time < 521 hour) check on 'rideable_type' classified by 'member_casual'
df6[df6['t_comm']<521].pivot_table(index = 'rideable_type', columns = ['member_casual'], values = ['t_comm'],
                                   aggfunc = {'t_comm':'mean'})

```

> #### *Key insight:*
> * *Among typical rider (commute time < 521 hour), There are 2,037,707 casual rider (40%) commute longer than member in average.*
> * *Docked bike still the most popular selection among them.*

#### Investigate 'member_casual' rider population among the top crowded station.

```python
# Create list of popular station
stat_pop = ['Millennium Park', 
            'Theater on the Lake', 
            'Lake Shore Dr & North Blvd', 
            'Clark St & Elm St', 
            'Streeter Dr & Grand Ave']

# Create function extract to return one of name in 'stat_pop'

def extract(x):
    if x in  stat_pop:
        return x
    else:
        return ''

df6a = df6[df6['t_comm']<521].copy()
df6a.shape

# Create new column 'station' to indicate frequency of each top 5 station.
# Check if any of station name is in 'start_station_name', then assign it onto 'station'
df6a.loc[df6a['start_station_name'].isin(stat_pop),'station'] = df6a.loc[:,'start_station_name'].apply(lambda x : extract(x))

# Check if any of station name is in 'end_station_name', then assign it onto 'station'
df6a.loc[df6a['end_station_name'].isin(stat_pop),'station'] = df6a['end_station_name'].apply(lambda x : extract(x))

df6a['station'].value_counts().sum()
df6a.pivot_table(index = 'station',
                 columns = ['member_casual'],
                 aggfunc = {'station':'count'}).style.bar()
#.style.background_gradient()
#.style.highlight_max(color="red")

# Find out the most popular route using combination of start and stop station name.
df6a.loc[(df6a['start_station_name'].isin(stat_pop)) & (df6a['end_station_name'].isin(stat_pop))]\
.groupby(['start_station_name','end_station_name']).size().reset_index().rename(columns={0:'count'})\
.sort_values('count', ascending = False).head(5)

```
![]()

> #### *Key insight:*
> * *8.5% of typical rider population (commute time < 521 hour) use top 5 station.*
> * *Casual rider most populated at Streeter Dr & Grand Ave station while member rider most populated at Clark St & Elm St station.*
> * *Only small population rider commute from the same location (start and end) across the top 5 station.*

#### Investigate the timeline aspect.

```python
# Create df67 as a copy of df2
df7 = df6.copy()

# Groupby 'member_casual' and days of week → find average value and reindex based on days category → plot
df7a = df7.groupby([df7['started_at'].dt.day_name(),'member_casual']).mean().reindex(dayscat, level=0)

fig = go.Figure()
fig = px.bar(df7a, x = df7a.index.get_level_values(0), y = 't_comm', color = df7a.index.get_level_values(1),barmode = 'group')
fig.update_layout(xaxis_title="Days of week",
                  yaxis_title="Commute time average [hour]",
                  title={'text': "Typical commute time on daily basis in 2020-2021",
                         'y':0.93,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'})
fig.show()
```
![]()

```python
# Groupby 'member_casual' and days of week → find total count and reindex based on days category → plot
df7b = df7.groupby([df7['started_at'].dt.day_name(),'member_casual']).size().to_frame()\
.rename(columns={0:'count'}).reindex(dayscat, level=0)

fig = go.Figure()
fig = px.bar(df7b, x = df7b.index.get_level_values(0), y = 'count', color = df7b.index.get_level_values(1),barmode = 'group')
fig.update_layout(xaxis_title="Days of week",
                  yaxis_title="Ride count",
                  title={'text': "Ride count on daily basis in 2020-2021",
                         'y':0.93,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'})
fig.show()
```
![]()

```python
# Groupby 'member_casual', year, and months → find total count and reindex based on months category → plot
df7c = df7.groupby([df7['started_at'].dt.year,df7['started_at'].dt.month_name(),'member_casual']).size().to_frame()\
.rename(columns={0:'count'}).reindex(monthscat, level=1)

fig = go.Figure()
fig = px.bar(df7c, x = df7c.index.get_level_values(1), y = 'count', 
             color = df7c.index.get_level_values(2),
             facet_col=df7c.index.get_level_values(0),
             category_orders={'df7c.index.get_level_values(0)': ['2020', '2021']},
             barmode = 'group')
fig.update_layout(xaxis_title="Months", xaxis_tickangle = 90,
                  xaxis2_title="Months", xaxis2_tickangle = 90,
                  yaxis_title="Ride count",
                  title={'text': "Ride count on monthly basis in 2020-2021",
                         'y':0.98,
                         'x':0.45,
                         'xanchor': 'center',
                         'yanchor': 'top'})
fig.show()
```
![]()

```python
# Groupby 'member_casual', year, and hour → find total count and reindex based on hour category → plot
df7d = df7.groupby([df7['started_at'].dt.year,df7['started_at'].dt.hour,'member_casual']).size().to_frame()\
.rename(columns={0:'count'})#.reindex(monthscat, level=1)

fig = go.Figure()

fig = px.bar(df7d, x = df7d.index.get_level_values(1), y = 'count', 
             color = df7d.index.get_level_values(2),
             facet_col=df7d.index.get_level_values(0),
             category_orders={'df7d.index.get_level_values(0)': ['2020', '2021']},
             barmode = 'group')
fig.update_layout(xaxis_title="Hour", xaxis_tickangle = 90,
                  xaxis2_title="Hour", xaxis2_tickangle = 90,
                  yaxis_title="Ride count",
                  title={'text': "Ride count on hour basis in 2020-2021",
                         'y':0.98,
                         'x':0.45,
                         'xanchor': 'center',
                         'yanchor': 'top'})
fig.show()
```
![]()

> #### *Key insight:*
> * *Based on 2020-2021 data, Casual rider bike progressingly increased from early year peak around mid year, then climb down.*
> * *Similar trend also seen from member ride, which indicate August is the best month for rent bike.*
> * *From daily basis, casual rider tend to bike on weekend time whereas member rider constantly bike over the week.*
> * *From hour basis, both casual and member rider typically start renting early morning at 5AM onward, peaked at 5PM, possibly after working hour commute.*


```python

```

![]()

> #### *Key insight:*
> * From context variable, we found potential 2,041,134 1TO casual rider (40% of 1TO population) to become member
