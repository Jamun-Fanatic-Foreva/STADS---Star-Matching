import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import mode

def starVectorTransform(centroid, focal_length=10):
    '''    
    Generates the unit 3D vectors from given 2D centroids of stars on the 
    image frame with the focal point as the origin
    
    <Formula> - CubeStar Doc - Appendix B
    
    Parameters
    ----------
    centroid : np.array
        Input (x,y) coordinates of stars in image frame
        
    focal_length : floating-point number, default = 10
        Input focal length of the optic system
          
    Returns
    -------
    y : np.array
        3D unit-vector of image star with the focal point as origin
    '''
    # Extracts (x,y) components
    x, y = centroid
    
    # Assert focal length != 0
    assert focal_length != 0, 'Error: focal_length = 0'
    
    # Given Formula
    temp = np.power(((x/focal_length)**2 + (y/focal_length)**2 + 1), -0.5)
    ux = (x/focal_length)
    uy = (y/focal_length)
    uz = 1
    
    return np.array([ux, uy, uz])*temp

def vectorAngularDistance(vect1, vect2):
    '''
    Returns the angle (in degrees) between two given vectors
    
    <Formula> - (Dot product of two vectors) / ( norm(vect1) * norm(vect2) )
    
    Parameters
    ----------
    vect1 : np.array
        Input unit vector of image star - 1
        
    vect2 : np.array
        Input unit vector of image star - 1
          
    Returns
    -------
    y : floating-point number
        Angle in degrees between the two vectors
    '''
    
    # Calulate Dot Product
    dot_product = np.sum(vect1*vect2)
    
    # Divide by norm of vectors
    dot_product = dot_product/(np.linalg.norm(vect1) * np.linalg.norm(vect2))
    
    # Return value in degrees
    return np.degrees(np.arccos(dot_product))

def starMatch(REF_ARR, STAR_CENTROIDS, GLOBAL_SIGMA, FOC_LEN=10, return_VoteList_1 = False):
    '''
    Generates list of the matched catalogue stars with its corresponding image star using
    the centroids of image stars
    
    Parameters
    ----------
    REF_ARR : np.array
        Input reference array against which star matching works
        #### Expected shape - (T_N, 3), T_N = number of stars, 3 = (star_ID1, star_ID2, angular distance (deg) )
        
    STAR_CENTROIDS : np.array
        Input (x,y) co-ordinates of stars on image plane as generated from feature extraction
        #### Expexted shape - (N,2), N = number of stars, 2 = (x,y) corrdinate components of each star
        
    GLOBAL_SIGMA : float/int
        Input the global sigma (a single number) in degrees, representing the error in angular distance between stars.
        This value will be the same for all the stars and is calculated on-ground         
    
    foc_len : floating-point number
        Input focal length of the optic system
    
    return_VoteList_1 : boolean, default = False
        <True> if final output should return both VOTE_LIST_1 and VOTE_LIST_2
        #### (IN THAT ORDER!) 
          
    Returns
    -------
    VOTE_LIST : np.array
        Returns an array (shape = (N,3), N = number of stars) where the columns represent the
        image star ID, matched catalogue star ID, and the number of votes received from the final 
        validation step
    '''
    # Assert shape of necessary arrays
    assert REF_ARR.shape[1] == 3, 'ShapeError: REF_ARR'
    assert STAR_CENTROIDS.shape[1] == 2, 'ShapeError: STAR_CENTROIDS'
    assert type(GLOBAL_SIGMA) == float or type(GLOBAL_SIGMA) == int, 'TypeError: GLOBAL_SIGMA'
    
    # Convert array of centroid data of image stars into corresponding 3D cartesian vector data
    STAR_VECTORS = np.apply_along_axis(starVectorTransform, 1, STAR_CENTROIDS, focal_length=FOC_LEN)
    
    return gvAlgorithm(REF_ARR, STAR_VECTORS, GLOBAL_SIGMA, return_VoteList_1)

    
def gvAlgorithm(REF_ARR, STAR_VECTORS, GLOBAL_SIGMA, return_VoteList_1 = False):
    '''
    Matches image star vectors to real stars from the catalogue using the Geometric Voting Algorithm
    
    <Reference>
        M. Kolomenkin, S. Pollak, I. Shimshoni and M. Lindenbaum, "Geometric voting algorithm for star trackers," in IEEE Transactions 
        on Aerospace and Electronic Systems, vol. 44, no. 2, pp. 441-456, April 2008.
        doi: 10.1109/TAES.2008.4560198
        keywords: {astronomical techniques;image sensors;stars;geometric voting algorithm;star trackers;satellite-based camera;star 
        catalogue;quaternion-based method;fast tracking algorithm;Voting;Cameras;Satellites;Gyroscopes;Robustness;Magnetic sensors;
        Calibration;Position measurement;System testing;Aerospace industry},
        URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4560198&isnumber=4560192
    
    Parameters
    ----------
    REF_ARR : np.array
        Input reference array against which the Geometric Voting Algorithm works
        #### Expected shape - (T_N, 3), T_N = number of stars, 3 = (star_ID1, star_ID2, angular distance (deg) )
        
    STAR_VECTORS : np.array
        Input (x,y,z) unit vectors of stars on image plane with origin at the focal point
        #### Expexted shape - (N,3), N = number of stars, 3 = (x,y,z) vector components of each star
        
    GLOBAL_SIGMA : float/int
        Input the global sigma (a single number) in degrees, representing the error in angular distance between stars.
        This value will be the same for all the stars and is calculated on-ground   
        
    return_VoteList_1 : boolean, default = False
        <True> if final output should return both VOTE_LIST_1 and VOTE_LIST_2 
        #### (IN THAT ORDER!) 
          
    Returns
    -------
    VOTE_LIST_2 : np.array
        Returns an array of N rows (N = number of image stars) and 3 columns, which represents the
        image star ID, matched catalogue star ID, and the number of votes received from the final 
        validation step
    '''
    # Assert shape of necessary arrays
    assert REF_ARR.shape[1] == 3, 'ShapeError: REF_ARR'
    assert STAR_VECTORS.shape[1] == 3, 'ShapeError: STAR_CENTROIDS'
    assert type(GLOBAL_SIGMA) == float or type(GLOBAL_SIGMA) == int, 'ShapeError: GLOBAL_SIGMA'    
    
    # Number of stars identified on sensor
    NUM_STARS = STAR_VECTORS.shape[0]
    
    # Generate an array of <VOTE_LIST>
    # Column 1 -> Integer numbers from 0 to NUM_STARS
    # Column 2 -> Empty Lists that will store the IDs of matched catalogue stars
    temp = [[1]]
    for i in range(NUM_STARS-1):
        temp.append([])
    temp = np.array(temp)
    temp[0].remove(1)
    VOTE_LIST_1 = np.vstack((np.arange(0, NUM_STARS), temp)).T
    

    # Run first iteration of Geometric Voting Algorithm 
    for i in range(NUM_STARS):
        
        # Range(i+1, NUM_STARS) to avoid processing on cases where (j == i) => angular distance between the same image star
        for j in range(i+1, NUM_STARS):
            d_ij = vectorAngularDistance(STAR_VECTORS[i], STAR_VECTORS[j])
            
            # Creates range <R_ij>
            r_ij = [d_ij - GLOBAL_SIGMA , d_ij + GLOBAL_SIGMA]
            
            # Finds indices of all the elements in <REF_ARR> whose angular distances lie within <R_ij>
            ind = np.where( (REF_ARR[:, 2] >= r_ij[0]) & (REF_ARR[:,2] <= r_ij[1]) )
            
            # Appends matched catalogue star IDs to the corresponding image star
            for k in REF_ARR[ind]:
                # Reads star IDs from indices that matched the condition above
                s1, s2 = k[0], k[1]
                
                # Appends star IDs to voting lists of corresponding sensor star
                VOTE_LIST_1[i, 1].append(s1)
                VOTE_LIST_1[i, 1].append(s2)
                VOTE_LIST_1[j, 1].append(s1)
                VOTE_LIST_1[j, 1].append(s2)


    # Generates array of <VOTE_LIST_2>
    # Column 1 -> Integer numbers from 0 to NUM_STARS
    # Column 2 -> ID of catalogue star that was repeated the most time for the coressponding image star
    # Column 3 -> Number of Votes when the validation step runs on the 'most probable' ID'ed catalogue star
    temp = np.arange(0, NUM_STARS)
    VOTE_LIST_2 = np.vstack((temp, np.zeros_like(temp),np.zeros_like(temp))).T

    # Appends the value of the most repeated catalogue star ID from list of voted stars
    for i in range(NUM_STARS):
        
        # If no catalogue star has voted on the image star, zero value is set
        VOTE_LIST_2[i,1] = mode(VOTE_LIST_1[i,1])[0][0] if mode(VOTE_LIST_1[i,1])[0].shape != (0,) else 0
        

    # Run second iteration of Geometric Voting Algorithm - <validation step>
    for i in range(NUM_STARS):
        for j in range(i+1, NUM_STARS):
            d_ij = vectorAngularDistance(STAR_VECTORS[i], STAR_VECTORS[j])
            r_ij = [d_ij - GLOBAL_SIGMA , d_ij + GLOBAL_SIGMA]
                        
            # Reads the 'most probable' catalogue star ID of corresponding image star
            s1, s2 = VOTE_LIST_2[i, 1], VOTE_LIST_2[j, 1]
            
            # Ignores the case when the corresponding catalogue star ID for a given image star == 0
            if s1 == 0 or s2 == 0:
                continue
            
            # Finds angular distance between the 'most probable' stars from <REF_ARR>
            ind1 = np.where( (REF_ARR[:, 0] == s1) & (REF_ARR[:,1] == s2) ) 
            
            # Case when <REF_ARR> does not have the angular distance between the specified catalogue star IDs
            if ind1[0].shape != (0,):
                if REF_ARR[ind1][0,2]>r_ij[0] and REF_ARR[ind1][0,2]<r_ij[1]:
                    VOTE_LIST_2[i,2] +=1
                    VOTE_LIST_2[j,2] +=1
                continue
            
            # Repeat the above step on interchanged-columns
            ind2 = np.where( (REF_ARR[:, 0] == s2) & (REF_ARR[:,1] == s1) )
            # Case when <REF_ARR> does not have the angular distance between the specified catalogue star IDs
            if ind2[0].shape != (0,):
                if REF_ARR[ind2][0,2]>r_ij[0] and REF_ARR[ind2][0,2]<r_ij[1]:
                    VOTE_LIST_2[i,2] +=1
                    VOTE_LIST_2[j,2] +=1
    
    if return_VoteList_1 == True:
        return VOTE_LIST_1, VOTE_LIST_2
    else:
        return VOTE_LIST_2

def main():
    '''
    main function
    '''
    # Initialize processed star - catalogue 
    REFERENCE= pd.read_csv(r'F:\IIT Bombay\SatLab\Star Tracker\Programs\Catalogues\Reduced_Catalogue.csv', usecols=['Star_ID1', 'Star_ID2', 'Ang_Distance'])
       
    # Converts reference catalogue to numpy array for faster implementation
    ref_array = REFERENCE.to_numpy()

    # Initialize centroid data and the corresponding centroid uncertainty data generated from running feature extraction
    ### Example Initialization - Random Values
    centroid = np.random.random((10,2))*2
    centroid_global_sigma = 2.21 #in degrees
    result = starMatch(REF_ARR=ref_array, STAR_CENTROIDS=centroid, GLOBAL_SIGMA = centroid_global_sigma)
    print(result)

if __name__ == '__main__':
    main()
    
###############################################
# --> STATS
'''
%timeit starMatch(ref_array, np.random.random((10,2))*2, 2.21)
5.35 s ± 504 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit starMatch(ref_array, np.random.random((5,2))*2, 2.21)
1.28 s ± 179 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
----------
%timeit starMatch(ref_array, np.random.random((10,2))*2, 1)
2.77 s ± 192 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit starMatch(ref_array, np.random.random((5,2))*2, 1)
639 ms ± 88.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
----------
%timeit starMatch(ref_array, np.random.random((10,2))*2, 0.5)
1.35 s ± 123 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit starMatch(ref_array, np.random.random((5,2))*2, 0.5)
282 ms ± 38.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
'''
###############################################
