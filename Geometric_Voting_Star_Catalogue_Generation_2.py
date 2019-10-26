import numpy as np
import pandas as pd
import time, gc
pi = np.pi 
cos = np.cos 
sin = np.sin
acos = np.arccos
degrees = np.degrees
radians = np.radians

def angularDistance(row):
    '''
    Computes the angular distance (degrees) between the reference Right-Ascension (RA)
    & Declination (Dec) value, and the corresponding RA - Dec value of the stars in 
    the dataframe - <OB_CATALOGUE>
    
    <Formula> - http://spiff.rit.edu/classes/phys373/lectures/radec/radec.html
    
    Parameters
    ----------
    row : pd.Dataframe - series
        Input RA/Dec in degrees from the <IMG_DF> dataframe
          
    Returns
    -------
    y : pd.Dataframe - series
        The corresponding angular distance in degree value.
    '''
    
    # Units of right-ascension is in (hours) format
    alpha1, alpha2 = radians(15*row['RA_1']), radians(15*row['RA_2'])
    
    # Units of declination is in (degrees) format
    delta1, delta2 = radians(row['Dec_1']), radians(row['Dec_2'])
    
    temp = cos(pi/2 - delta1)*cos(pi/2 - delta2) + sin(pi/2 - delta1)*sin(pi/2 - delta2)*cos(alpha1 - alpha2) 
    
    return np.degrees(acos(temp))


def genRefCatalogue(mag_limit, no_iter = -1, gen_csv = True):
    '''
    Generates the reference star catalogue for Geometric Voting Algorithm where each row
    of the table has two unique stars and the corresponding angular distance in degrees,
    for all pairs of stars with a specified upper magnitude limit
    
    Parameters
    ----------
    mag_limit : floating-point number
        The upper magnitude limit of stars that are required in the reference catalogue
        
    no_iter : integer, default = -1
        Specifies the number of iterations, thereby allowing it to be reduced
        Default value = -1, allows for the completion of the entire catalogue
        
    gen_csv : boolean, default = True
          If True generates csv files of the reference catalogues
          
    Returns
    -------
    OB_CATALOGUE : pd.Dataframe
        The corresponding angular distance in degree value.
    '''
    
    # Start clock-1
    start1 = time.time()
    
    # Generate restricted catalogue based on upper magnitude limit
    temp0 = CATALOGUE[CATALOGUE.Mag <= mag_limit]
    
    # Number of rows in the resticted catalogue
    rows = temp0.shape[0]
    # Resets the index of <temp0>
    temp0.index = list(range(rows))
    
    # Prints total number of stars in <temp0> and the (n)C(2) - combinations
    print('Number of stars - ', rows)
    print('Number of unique combinations = ', (rows-1)*rows/2)
    
    # Initialize the number of iterations to take place
    no_iter = (rows-1) if no_iter == -1 else no_iter
    
    for i in range(no_iter):
        # Throws error if an iteration runs beyond number of available rows in <temp0>
        assert i<(rows-1), 'iterating beyond available number of rows'
        
        # The final iteration is reduntant, as <temp2> will be zero rows
        '''
        if (rows-1-i)==0:
            continue
        '''
        
        # Generates <temp1> dataframe which has the (i - th) star of <temp0>
        # repetated (rows-1-i) times 
        temp1 = pd.DataFrame(columns = ['Star_ID1','RA_1', 'Dec_1', 'Mag_1'])
        s1, ra, dec, mag = temp0.iloc[i]
        temp1.loc[0] = [s1] + [ra] + [dec] + [mag]
        temp1 = pd.concat([temp1]*(rows-1-i), ignore_index=True)
        
        # Generates <temp2> dataframe by copying values of <temp0> and dropping the first
        # (i + 1) number of stars
        temp2 = temp0
        temp2 = temp2.drop(list(range(i+1)), axis = 0)
        # Resets the index 
        temp2.index = list(range(0, rows-1-i))
        
        # Concatenates <temp1> & <temp2> side-by-side such that resulting <temp3> has (8) columns altogether
        temp3 = pd.concat([temp1, temp2], axis=1)
        
        # Initializes <temp4> in the first iteratation
        if i == 0:
            temp4 = temp3
            
        # Append subsequent <temp4> with <temp3> after first iteration
        else:
            temp4 = pd.concat([temp4, temp3], axis = 0, ignore_index=True)
        
        # Releases memory back to OS
        if i%40 == 0:
            gc.collect()
    
    gc.collect()      
    # Rename columns
    temp4.columns = ['Star_ID1','RA_1', 'Dec_1', 'Mag_1', 'Star_ID2', 'RA_2', 'Dec_2', 'Mag_2']
    
    if gen_csv == True:
        #Generates CSV of <temp4>
        temp4.to_csv('Processed_Catalogue1.csv', index = False)
        
    # Stop clock-1   
    end1 = time.time() - start1
    
    # Print time taken
    print('Process 1 - ', end1)
    
    # Start clock-2
    start2 = time.time()
    
    # Initialize <OB_CATALOGUE>
    OB_CATALOGUE = temp4
    
    # Calculate angular distance between the two stars present in every row
    OB_CATALOGUE['Ang_Distance'] = OB_CATALOGUE.apply(angularDistance, axis = 1)
    
    if gen_csv == True:
        # Generates CSV of <OB_CATALOGUE>
        OB_CATALOGUE.to_csv('Processed_Catalogue2.csv', index = False)
        
    # Stop clock-2
    end2 = time.time() - start2
    
    # Print time taken
    print('Process 2 - ', end2)
    print('Total Process ', end1+ end2)
        
    return OB_CATALOGUE

CATALOGUE = pd.read_csv("Modified Star Catalogue.csv")
# StarID: The database primary key from a larger "master database" of stars
# Mag: The star's apparent visual magnitude
# RA, Dec: The star's right ascension and declination, for epoch 2000.0 (Unit: RA - hrs; Dec - degrees)

# Sorts <CATALOGUE>
CATALOGUE.sort_values('Mag', inplace=True)

# Run function
REF_DF = genRefCatalogue(mag_limit=4, no_iter=-1, gen_csv=False)

# Sort <REF_DF>
REF_DF.sort_values('Ang_Distance', inplace=True)
# Generates CSV of <REF_DF>
REF_DF.to_csv('Processed_Catalogue3.csv', index = False)