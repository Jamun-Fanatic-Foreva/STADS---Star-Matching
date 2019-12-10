import numpy as np
import pandas as pd
import GV_Runtime as gv_r
import Star_Image_Generation as sig

pi = np.pi 
cos = np.cos 
sin = np.sin
acos = np.arccos
degrees = np.degrees
radians = np.radians

def cartesianVector_x(row, col_names):
    '''
    Returns the (x) component of the cartesian vector generated from right-ascension and declination of a star
    
    <Reference> - http://fmwriters.com/Visionback/Issue14/wbputtingstars.htm
    
    Parameters
    ----------
    row : pd.Dataframe - series
        Input the values of right-ascension and declination
        
    col_name : string
        The name of the column on which the function will be applied
    
    Returns
    -------
    x : floating-point number
        The (x) component of the cartesian vector     
    '''
    c1, c2 = col_names
    
    # Converting right-ascension into degrees format
    ra, dec = 15*row[c1], row[c2]
    
    # Given Formula
    x = cos(radians(dec))*cos(radians(ra))
    y = cos(radians(dec))*sin(radians(ra))
    z = sin(radians(dec))
    
    # Normalize vector
    temp = np.array([x,y,z])
    temp = temp/np.linalg.norm(temp)
    
    # Return (x) component
    return temp[0]

def cartesianVector_y(row, col_names):
    '''
    Returns the (y) component of the cartesian vector generated from right-ascension and declination of a star
    
    <Reference> - http://fmwriters.com/Visionback/Issue14/wbputtingstars.htm
    
    Parameters
    ----------
    row : pd.Dataframe - series
        Input the values of right-ascension and declination
        
    col_name : string
        The name of the column on which the function will be applied
    
    Returns
    -------
    y : floating-point number
        The (y) component of the cartesian vector     
    '''
    c1, c2 = col_names
    
    # Converting right-ascension into degrees format
    ra, dec = 15*row[c1], row[c2]
    
    # Given Formula
    x = cos(radians(dec))*cos(radians(ra))
    y = cos(radians(dec))*sin(radians(ra))
    z = sin(radians(dec))
    
    # Normalize vector
    temp = np.array([x,y,z])
    temp = temp/np.linalg.norm(temp)
    
    # Return (y) component
    return temp[1]

def cartesianVector_z(row, col_names):
    '''
    Returns the (z) component of the cartesian vector generated from right-ascension and declination of a star
    
    <Reference> - http://fmwriters.com/Visionback/Issue14/wbputtingstars.htm
    
    Parameters
    ----------
    row : pd.Dataframe - series
        Input the values of right-ascension and declination
        
    col_name : string
        The name of the column on which the function will be applied
    
    Returns
    -------
    z : floating-point number
        The (z) component of the cartesian vector     
    '''
    c1, c2 = col_names
    
    # Converting right-ascension into degrees format
    ra, dec = 15*row[c1], row[c2]
    
    # Given Formula
    x = cos(radians(dec))*cos(radians(ra))
    y = cos(radians(dec))*sin(radians(ra))
    z = sin(radians(dec))
    
    # Normalize vector
    temp = np.array([x,y,z])
    temp = temp/np.linalg.norm(temp)
    
    # Return (y) component
    return temp[2]

def norm(row, col_names):
    '''
    Returns the L2 - norm of the cartesian vectors of each star
        
    Parameters
    ----------
    row : pd.Dataframe - series
        Input the values of vector components
        
    col_name : string
        The name of the column on which the function will be applied
    
    Returns
    -------
    y : floating-point number
        The norm of the vector  
    '''    
    c1, c2, c3 = col_names
    
    # Extracts (x,y,z) vector components
    x, y, z = row[c1], row[c2], row[c3]
    
    # Return norm
    return np.sqrt(x*x + y*y + z*z)

def test_starVectorTransform():
    '''
    Function to test <starVectorTransform>
    '''
    cent, f = [0,0], 10
    result = [0,0,1]
    assert list(gv_r.starVectorTransform(cent, f)) == result, 'Error: Testcase - 1'
    
    cent, f = [5,5], 5
    # Round answer to neglect numerical error
    result = np.round( np.array([1,1,1])/np.sqrt(3) , 10 )
    result = list(result)
    # Round answer to neglect numerical error
    check = np.round( gv_r.starVectorTransform(cent, f) , 10 )
    check = list(check)
    assert check == result, 'Error: Testcase - 2'
    
    cent, f = [-5,10], 50
    # Round answer to neglect numerical error
    result = np.round( np.array([-1,2,10])/np.sqrt(105) , 10 )
    result = list(result)
    # Round answer to neglect numerical error
    check = np.round( gv_r.starVectorTransform(cent, f) , 10 )
    check = list(check)
    assert check == result, 'Error: Testcase - 3'
    
    print('starVectorTransform - Tested')
    
    
def test_vectorAngularDistance():
    '''
    Function to test <vectorAngularDistance>
    '''
    
    v1, v2 = np.array([1,0,0]), np.array([0,1,0])
    result = 90.0
    assert gv_r.vectorAngularDistance(v1, v2) == result, 'Error: Testcase - 1'
    
    v1, v2 = np.array([1,1,0])/np.sqrt(2), np.array([1,0,0])
    result = 45.0
    assert gv_r.vectorAngularDistance(v1, v2) == result, 'Error: Testcase - 2'
    
    v1, v2 = np.array([1,1,0])/np.sqrt(2), np.array([-1,-1,0])/np.sqrt(2)
    result = 180
    assert gv_r.vectorAngularDistance(v1,v2) == result, 'Error: Testcase - 3'
    
    
    print('vectorAngularDistance - Tested')

    
def test_gvAlgorithm_rigid():
    '''
    Funtion to test <gvAlgorithm> with just one testcase, with a constant global-sigma value    
    '''    
    # Initialize main and processed star - catalogues
    CATALOGUE = pd.read_csv(r"F:\IIT Bombay\SatLab\Star Tracker\Programs\Catalogues\Modified Star Catalogue.csv")
    REFERENCE= pd.read_csv(r'F:\IIT Bombay\SatLab\Star Tracker\Programs\Catalogues\Reduced_Catalogue.csv', usecols=['Star_ID1', 'Star_ID2', 'Ang_Distance'])

    # Converts reference catalogue to numpy array for faster implementation
    ref_array = REFERENCE.to_numpy()
    
    # Generate array of stars using <generateImageDataframe> about a given point, with a specified right-ascension & declination value,
    # with a max circular-FOV and maximum magnitude limit value
    # FIXED VALUES
    # >>>ref_ra=5, ref_dec=5, ref_ang_dist=10, mag_limit=4.5
    result = sig.generateImageDataframe(CATALOGUE, ref_ra=5, ref_dec=5, ref_ang_dist=20, mag_limit=4, ra_hrs=True) 
 
    # Generate <temp> dataframe which has following columns: Star_ID, RA/Dec
    temp = result
    temp = temp.drop(['Ref_RA', 'Ref_Dec', 'Mag'], axis = 1)
    temp.index = list(range(temp.shape[0]))

    # Generate the (x,y,z)-vect components for each star
    cols = ['RA', 'Dec']
    temp['x_vect'] = temp.apply(cartesianVector_x, axis = 1, col_names=cols)
    temp['y_vect'] = temp.apply(cartesianVector_y, axis = 1, col_names=cols)
    temp['z_vect'] = temp.apply(cartesianVector_z, axis = 1, col_names=cols)
    
    #  Generate the norm of the cartesian vector for each star
    cols = ['x_vect', 'y_vect', 'z_vect']
    temp['norm'] = temp.apply(norm, axis = 1, col_names = cols )
    
    # Generate <test_df> dataframe which has following columns: Star_ID, RA/Dec, (x,y,z)-vect components, norm of vector
    #test_df = temp   
    
    # Generate <temp1> dataframe which has following columns: (x,y,z)-vect components
    temp1 = temp.drop(['Star_ID', 'RA', 'Dec', 'norm','Ang_Dist'], axis = 1)
    
    # Generate np.arrays of the star vectors
    st_vect = temp1.to_numpy()
    
    # Initialize global_sigma 
    global_sigma = 2.21
   
    # Running Geometric Voting Algorithm on generated star-vectors and star uncertainty values
    final_result = gv_r.gvAlgorithm(ref_array, st_vect, global_sigma, return_VoteList_1=False)
    
    #check_arr1 = list(final_result[:, 1]==true_stars)
    #answer1 = [True, False, False, False, False, False, False, True, True, False, False, True, True, False, False, False]
    #assert check_arr1 == answer1, 'Error: Testcase - 1'
    
    #check_arr2 = list(final_result[:, 2])
    #answer2 = [4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0]
    #assert check_arr2 == answer2, 'Error: Testcase - 2'
    print(temp)
    print(final_result)
    print('gvAlgorithm_Rigid - Tested')
    

def test_gvAlgorithm():
    '''
    Funtion to test <gvAlgorithm>
    '''    
    # Initialize main and processed star - catalogues
    CATALOGUE = pd.read_csv(r"F:\IIT Bombay\SatLab\Star Tracker\Programs\Catalogues\Modified Star Catalogue.csv")
    REFERENCE= pd.read_csv(r'F:\IIT Bombay\SatLab\Star Tracker\Programs\Catalogues\Reduced_Catalogue.csv', usecols=['Star_ID1', 'Star_ID2', 'Ang_Distance'])

    # Converts reference catalogue to numpy array for faster implementation
    ref_array = REFERENCE.to_numpy()
    
    # Generate array of stars using <generateImageDataframe> about a given point, with a specified right-ascension & declination value,
    # with a max circular-FOV and maximum magnitude limit value
    # >>>The given RA/Dec value is centered about Orion's belt
    #5555555555result = sig.generateImageDataframe(CATALOGUE, ref_ra=5.60355904, ref_dec=-1.20191725, ref_ang_dist=5, mag_limit=5, ra_hrs=True)
    result = sig.generateImageDataframe(CATALOGUE, ref_ra=5.60355904, ref_dec=-1.20191725, ref_ang_dist=20, mag_limit=3, ra_hrs=True)
    print('1')
    # Generate <temp> dataframe which has following columns: Star_ID, RA/Dec
    temp = result
    temp = temp.drop(['Ref_RA', 'Ref_Dec', 'Mag'], axis = 1)
    temp.index = list(range(temp.shape[0]))

    # Generate the (x,y,z)-vect components for each star
    cols = ['RA', 'Dec']
    temp['x_vect'] = temp.apply(cartesianVector_x, axis = 1, col_names=cols)
    temp['y_vect'] = temp.apply(cartesianVector_y, axis = 1, col_names=cols)
    temp['z_vect'] = temp.apply(cartesianVector_z, axis = 1, col_names=cols)
    
    #  Generate the norm of the cartesian vector for each star
    cols = ['x_vect', 'y_vect', 'z_vect']
    temp['norm'] = temp.apply(norm, axis = 1, col_names = cols )
    
    # Generate <test_df> dataframe which has following columns: Star_ID, RA/Dec, (x,y,z)-vect components, norm of vector
    #test_df = temp
    
    # Generate array of Star_IDs alone
    true_stars = temp.Star_ID.to_numpy()    
    
    # Generate <temp1> dataframe which has following columns: (x,y,z)-vect components
    temp1 = temp.drop(['Star_ID', 'RA', 'Dec', 'norm','Ang_Dist'], axis = 1)
    
    # Generate np.arrays of the star vectors
    st_vect = temp1.to_numpy()
    
    # Initialize global_sigma 
    global_sigma = 1
    
    # Running Geometric Voting Algorithm on generated star-vectors and star uncertainty values
    final_result = gv_r.gvAlgorithm(ref_array, st_vect, global_sigma, return_VoteList_1=False)
    
    print(final_result)
    print(true_stars)
    
    check_arr = list(final_result[:, 1]==true_stars)
    print(check_arr)
    count_true = check_arr.count(True)
    count_false = check_arr.count(False)
    print('True = ', count_true)
    print('False = ', count_false)
    
def main():
    #test_starVectorTransform()
    #test_vectorAngularDistance()    
    #test_gvAlgorithm_rigid()
    test_gvAlgorithm()


if __name__ == "__main__":
    main()
