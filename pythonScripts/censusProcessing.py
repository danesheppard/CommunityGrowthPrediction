# This is a standalone file to process a Statistics Canada census file for machine learning
import pandas as pd
import time

# Since we learned what processing needs to be done in the previous file,
# we can now write a function to do this processing

def processCensusData(censusData, year = 2016):
    '''
    This function processes a Statistics Canada census data file for machine learning.
    It handles the different data structures for the 2016 and 2021 census files.
    The function returns a pandas dataframe with the processed data.
    Args:
        censusData: A pandas dataframe containing the census data as imported from the csv file.
        year: The year of the census data. Default is 2016.
    '''
    # If this is 2016 data, let's rename the 'DIM: Profile...' column
    if year == 2016:
        censusData.rename(columns={'DIM: Profile of Census Subdivisions (2247)': 'CHARACTERISTIC_NAME'}, inplace=True)
    # Next we trim the characteristic names
    censusData['CHARACTERISTIC_NAME'] = censusData['CHARACTERISTIC_NAME'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    censusData['CHARACTERISTIC_NAME'] = censusData['CHARACTERISTIC_NAME'].str.strip()

    # For the 2021 data, we need to split the commnity names on the comma
    # to drop additional subdivision info and match the names with any 2016 data
    if year == 2021:
        censusData['GEO_NAME'] = censusData['GEO_NAME'].str.split(',').str[0]

    # Trim the columns by keywords
    keyWords = ['population', 
                'household', 
                'employment', 
                'industry', 
                'labour', # Labour force information
                'years', # To catch age categories
                'decile', # To catch income decile
                'Employ', # Employment status
                'Unemploy', # Employment status
                'Occupation', # Industry of occupation
                'occupation' # Industry of occupation
                '$', # To catch income values. This will generate a lot of duplicates, but we will handle that later
                ]
    trimmedData = censusData[censusData['CHARACTERISTIC_NAME'].str.contains('|'.join(keyWords), case=False, na=False)]
    # Pivot the table on the 'GEO_NAME' column
    valueColumn = ''
    if year == 2016:
        valueColumn = 'Dim: Sex (3): Member ID: [1]: Total - Sex'
    else: # 2021 data
        valueColumn = 'C1_COUNT_TOTAL'
    pivotedData = trimmedData.pivot_table(index='GEO_NAME', 
                                         columns= 'CHARACTERISTIC_NAME', 
                                         values= valueColumn, 
                                         aggfunc='first')
    # Coerce numeric values
    pivotedData = pivotedData.apply(pd.to_numeric, errors='coerce')
    return pivotedData

# Import the census data from /statCanData
# Let's also record the time for processing
start = time.time()
print('Processing census data...')

years = [2016, 2021]
locationList = ['PEI', 'NS', 'NB', 'NL', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YK', 'NT', 'NU']

for location in locationList:
    for year in years:
        fileName = f'{location}_{year}.csv'
        encoding = 'ISO-8859-1'
        rawData = pd.read_csv(f'../statCanData/{fileName}', encoding=encoding, low_memory=False)
        processedData = processCensusData(rawData, year)

        # Save the processed data to a csv file, with a name based on the original file
        processedData.to_csv(f'../processedData/processed_{fileName}', index=True)
        print(f'Processed {fileName} and saved as processed_{fileName}')

end = time.time()
print(f'Processing complete. Time taken: {end-start} seconds')

    

# %%



