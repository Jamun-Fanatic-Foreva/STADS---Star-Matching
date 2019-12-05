import numpy as np
import pandas as pd
import time, gc
from GV_Catalogue_Gen import angularDistance

def genSigmaCatalogue(CATALOGUE, mag_limit = 6, FOV_limit = 20):
    '''
    Generates the mean of the sigma for each star in the catalogue.
    
    Sigma between star A and star B is defined as (1/6) of the angular 
    distance between the two stars.
    
    Such values of sigma are calculated for star A to every other star 
    in the catalogue that are its nearest neighbours, i.e., all those
    stars within a circular FOV defined by FOV_limit.
    This set of sigma values is defined as sigma_n.
    
    The mean of all the elements of sigma_n gives us mu_n.
    This mean value is paired with the corresponding star A.
    
    This process repeats for every star in the catalogue, and the star IDs
    the corresponding mu_n values are collated in a dataframe.
    
    Parameters
    ----------
    CATALOGUE : pd.Dataframe
        The 'master' star catalogue on which the function works

    mag_limit : floating-point number, default = 6
        The upper magnitude limit of stars that are required in the reference catalogue
        
    FOV_limit: floating-point number, default = 20
        Defines the circular radius (in degrees) which demarcates which stars from the 
        catalogue are to be considered as nearest neighbours for a given star
        
    Returns
    -------
    SIGMA_CATALOGUE : pd.Dataframe
        The dataframe collated from the star IDs and their corresponding mu_n
    '''
    
    # Start clock-1
    start1 = time.time()
    
    # Generate restricted catalogue based on upper magnitude limit
    temp0 = CATALOGUE[CATALOGUE.Mag <= mag_limit]
    
    # Number of rows in the resticted catalogue
    rows = temp0.shape[0]
    # Resets the index of <temp0>
    temp0.index = list(range(rows))
    
    # Prints total number of stars in <temp0> and the (n)X(n-1)- unique combinations per star
    print('Number of stars - ', rows)
    print('Number of unique combinations per star= ', (rows-1)*rows)
    
    # Initialize the number of iterations to take place
    no_iter = (rows) 
    
    # Initialize SIGMA_CATALOGUE
    SIGMA_CATALOGUE = pd.DataFrame(columns=['Star_ID', 'mu_n'])
    
    for i in range(no_iter):
        # Throws error if an iteration runs beyond number of available rows in <temp0>
        assert i<(rows), 'IndexError: iterating beyond available number of rows'
        
        # Generates <temp1> dataframe which has the (i - th) star of <temp0>
        # repetated (rows-1) times 
        temp1 = pd.DataFrame(columns = ['Star_ID1','RA_1', 'Dec_1', 'Mag_1'])
        s1, ra, dec, mag = temp0.iloc[i]
        temp1.loc[0] = [s1] + [ra] + [dec] + [mag]
        temp1 = pd.concat([temp1]*(rows-1), ignore_index=True)

        # Stores value of the star_ID for which mu_n will be calculated
        star_id_i = s1

        # Generates <temp2> dataframe by copying values of <temp0> and dropping the
        # (i -th) row from it
        temp2 = temp0
        temp2 = temp2.drop([i], axis = 0)
        # Resets the index 
        temp2.index = list(range(0, rows-1))

        # Concatenates <temp1> & <temp2> side-by-side such that resulting <temp3> has (8) columns altogether
        temp3 = pd.concat([temp1, temp2], axis=1)
        # Renaming columns of <temp3>
        temp3.columns = ['Star_ID1','RA_1', 'Dec_1', 'Mag_1', 'Star_ID2', 'RA_2', 'Dec_2', 'Mag_2']


        # Calculate angular distance between the two stars present in every row in <temp3>
        cols = ['RA_1', 'RA_2', 'Dec_1', 'Dec_2']
        temp3['Ang_Distance'] = temp3.apply(angularDistance, axis = 1, col_names = cols)

        # Generates <temp4> by selecting rows from <temp3> whose angular distances is
        # less than equal to the circular FOV limit
        temp4 = temp3[temp3.Ang_Distance <= FOV_limit]

        # Stores the value of the calculated mu_n for the current star
        mu_n_i = temp4.Ang_Distance.mean()
        
        # Multiply (mu_n_i) by (1/6) since sigma_i = Ang_distance_i, for all (i)
        mu_n_i = mu_n_i/6

        # Appends the entry to the SIGMA_CATALOGUE dataframe
        SIGMA_CATALOGUE = SIGMA_CATALOGUE.append({'Star_ID':star_id_i, 'mu_n':mu_n_i}, ignore_index=True)        
        
        # Releases memory back to OS
        if i%100 == 0:
            gc.collect()
            print(i/100)
            
    # Stop clock-1   
    end1 = time.time() - start1
    
    # Print time taken
    print('Time Taken - ', np.round(end1,3))
        
    return SIGMA_CATALOGUE

def main():
    '''
    main function
    '''
    # Reads 'Master' star catalogue
    CATALOGUE = pd.read_csv(r"F:\IIT Bombay\SatLab\Star Tracker\Programs\Catalogues\Modified Star Catalogue.csv")
    # StarID: The database primary key from a larger "master database" of stars
    # Mag: The star's apparent visual magnitude
    # RA, Dec: The star's right ascension and declination, for epoch 2000.0 (Unit: RA - hrs; Dec - degrees)
    
    # Sorts <CATALOGUE>
    CATALOGUE.sort_values('Mag', inplace=True)
    
    # Run function
    result = genSigmaCatalogue(CATALOGUE, mag_limit = 6, FOV_limit = 20)
    
    # Sort <result>
    result.sort_values('mu_n', inplace=True)
    
    
    # Generates CSV of <result>
    result.to_csv('SigmaCatalogue.csv', index = False)
    print('Done')
    
if __name__ == '__main__':
    main()
