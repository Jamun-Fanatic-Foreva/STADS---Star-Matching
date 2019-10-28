import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import GV_Catalogue_Gen as gv_cg

def generateImageDataframe(CATALOGUE, ref_ra, ref_dec, ref_ang_dist, mag_limit = 6, ra_hrs = True):
    '''
    Generates a dataframe consisting of stars that lie within the circular boundary
    for a given max angular distance value for the generation of a star-image.
    The max magnitude limit that the stars possess can be set manually (Default = 6 Mv)
    
    Parameters
    ----------
    CATALOGUE : pd.Dataframe
        The 'master' star catalogue on which the function works
        
    ref_ra : floating-point number
        Input reference right-ascension value
        
    ref_dec : floating-point number
        Input reference declination value
        
    ref_ang_dist : floating-point number
        Input the circular field-of-view (FOV), the radius of which defines the conical
        boundary within which the stars from the catalogue should lie in
        
    mag_limit : floating-point number
        Input the maximum value of stars' magnitude that should be visible within with 
        circular FOV
        
    ra_hrs : boolean, default = True
        Input is True if unit of right ascension is in hour format
        Input is False if unit of right ascension is in degrees format 
                
        <Formula> - https://sciencing.com/calculate-longitude-right-ascension-6742230.html 
        
    Returns
    -------
    IMG_DF : pd.Dataframe
        This returns the dataframe consisting of stars that lie inside the specified circular FOV 
        that is sorted w.r.t the angular distance column in ascending order
    '''
    if ra_hrs == False:
        # Conversion of right-ascension from degrees to hours
        ref_ra = ref_ra/15
    
    # Generates image dataframe 
    IMG_DF = pd.DataFrame(columns=['Ref_RA', 'Ref_Dec', 'Star_ID', 'RA', 'Dec', 'Mag'])
    
    # Restricts stars to specified upper magnitude limit
    temp = CATALOGUE[CATALOGUE.Mag <= mag_limit]
    
    # Total number of rows in <temp>
    size = temp.StarID.shape[0]
    
    # Counter for rows in <IMG_DF>
    row_count = 0    
    for i in range(size):
        
        # Extracts data from (i - th) row of <temp>
        s_id, ra, dec, mag = temp.iloc[i] 
        
        # Copies data into (row_count - th) row of <IMG_DF>
        IMG_DF.loc[row_count] = [ref_ra] + [ref_dec] + [s_id] + [ra] + [dec] + [mag]
        
        # Increment row_count
        row_count = row_count + 1
        
        
    # Apply angularDistance> function on 'Ang_Dist' column of <IMG_DF> 
    cols = ['Ref_RA', 'RA', 'Ref_Dec', 'Dec']
    IMG_DF['Ang_Dist'] = IMG_DF.apply(gv_cg.angularDistance, axis=1, col_names = cols)
    
    # Sort <IMG_DF> based on 'Ang_Dist' column
    IMG_DF.sort_values('Ang_Dist', inplace = True, ascending = True)
    
    # Remove entries with angular distance in <IMG_DF> greater than that of <ref_ang_dist>
    IMG_DF = IMG_DF[IMG_DF.Ang_Dist <= ref_ang_dist]
    
    return IMG_DF

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
    
    
    # Generates example image frame centred around Orion's Belt
    result = generateImageDataframe(CATALOGUE, ref_ra=5.60355904, ref_dec=-1.20191725, ref_ang_dist=15, mag_limit=4.5, ra_hrs=True)
    
    # Plots stars with x-axis = (-ve) right-ascension; y-axis = (+ve) declination
    plt.figure()
    plt.scatter(-result.RA, result.Dec, c = result.Mag )
    plt.plot(-result.iloc[0].Ref_RA, result.iloc[0].Ref_Dec, 'ro', label = 'center')
    plt.legend(loc='upper right')
    plt.xlim(-7, -4)
    plt.colorbar()
    plt.grid()
    
if __name__ == '__main__':
    main()