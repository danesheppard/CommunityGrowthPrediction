# This script assembles all the preprocessed provincial data files
# into a single file for the whole country.

import pandas as pd
import time

locationList = ['PEI', 'NS', 'NB', 'NL', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YK', 'NT', 'NU']
# locationList = ['PEI', 'NS', 'NB']
years = [2016, 2021]
encoding = 'ISO-8859-1'


# Read in the data for each province
start = time.time()
for year in years:
    provinceDFs = []
    for location in locationList:
            print(f"Reading in data for {location} {year}...")
            filePath = f'../processedData/processed_{location}_{year}.csv'
            df = pd.read_csv(filePath, encoding=encoding, low_memory=False)

            # Add a column for the province
            df['Province'] = location
            provinceDFs.append(df)

    # Concatenate the dataframes
    print(f'Concatenating dataframes for {year}...')
    allData = pd.concat(provinceDFs, ignore_index=False)
    # Sort by 'GEO_NAME'
    allData = allData.sort_values(by='GEO_NAME')

    # Let's add a column for the province so we can deal with duplicated community names

    # Write the data to a file
    allData.to_csv(f'../processedData/processed_Canada_{year}.csv', index=False)


end = time.time()
print(f'Data assembly took {end - start} seconds.')



