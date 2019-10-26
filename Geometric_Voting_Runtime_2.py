import numpy as np
import pandas as pd
from scipy.stats import mode

def cos(row):
    '''
    Calulates the value of <cos(theta)> for the <theta> value in <Ang_Distance> column of dataframe
        
    Parameters
    ----------
    row : pd.Dataframe - series
        Input angular distance in degrees
          
    Returns
    -------
    y : pd.Dataframe - series
        The corresponding cos() value 
    '''
    return np.cos(np.radians(row['Ang_Distance']))

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
        Input focal length of the lens
          
    Returns
    -------
    y : np.array
        3D unit-vector of image star with the focal point as origin
    '''
    x, y = centroid
    
    temp = np.power(((x/focal_length)**2 + (y/focal_length)**2 + 1), -0.5)
    ux = (x/focal_length)
    uy = (y/focal_length)
    uz = 1
    return np.array([ux, uy, uz])*temp

def dotProduct(vect1, vect2):
    '''
    Returns the dot product, i.e, the angular distance [cos(theta)] between two <unit> vectors seperated by an angle theta
    
    <Formula> - Dot product of two vectors
    
    Parameters
    ----------
    vect1 : np.array
        Input unit vector of image star - 1
        
    vect2 : np.array
        Input unit vector of image star - 1
          
    Returns
    -------
    y : floating-point number
        Scalar value of dot product
    '''
    return np.sum(vect1*vect2)

def uncertaintyAngularDistance(u1, u2):
    '''
    Returns the value of net uncertainty between the centroid data of two image stars
    
    <Formula> - Simple Addition
    
    Parameters
    ----------
    u1 : np.array, floating-point number
        Input uncertainty value of image star - 1
        
    u2 : np.array, floating-point number
        Input uncertainty value of image star - 2
          
    Returns
    -------
    y : floating-point number
        Scalar value of net uncertainty
    '''
    return u1 + u2

def starMatchAlgo(STAR_CENTROIDS, STAR_CENTROIDS_UNCERTAINTY):
    '''
    Generates list of the matched catalogue stars with their corresponding image star and the votes
    each has received in the validation step  
    
    <Reference> - 
    
    Parameters
    ----------
    STAR_CENTROIDS : np.array
        Input (x,y) co-ordinates of stars on image plane as generated from feature extraction
        
    STAR_CENTROIDS_UNCERTAINTY : np.array
        Input uncertainty value of corresponding star for which the (x,y) co-ordinates are provided
          
    Returns
    -------
    VOTE_LIST_2 : np.array
        Returns (N X 3) array which stores the image star ID, matched catalogue star ID, and the 
        number of votes received from the final validation step
    '''
    
    # Number of stars identified on sensor
    NUM_STARS = STAR_CENTROIDS.shape[0]

    # Asserting if the number of rows in STAR_CENTROIDS_UNCERTAINTY & STAR_CENTROIDS match
    assert STAR_CENTROIDS.shape[0] == STAR_CENTROIDS_UNCERTAINTY.shape[0], 'STAR_CENTROIDS != STAR_CENTROIDS_UNCERTAINTY' 
    
    # Converting array of centroid data of image stars into corresponding 3D cartesian vector data
    STAR_VECTORS = np.apply_along_axis(starVectorTransform, 1, STAR_CENTROIDS, focal_length=10 )

    # Generating an array of <VOTE_LIST>
    # Column 1 -> Integer numbers from 0 to NUM_STARS
    # Column 2 -> Empty Lists that will store the IDs of matched catalogue stars
    temp = [[1]]
    for i in range(NUM_STARS-1):
        temp.append([])
    temp = np.array(temp)
    temp[0].remove(1)
    VOTE_LIST = np.vstack((np.arange(0, NUM_STARS), temp)).T

    # Running first iteration of Geometric Voting Algorithm 
    for i in range(NUM_STARS):
        
        # Range(i+1, NUM_STARS) to avoid processing on cases where (j == i) => angular distance between the same image star
        for j in range(i+1, NUM_STARS):
            d_ij = dotProduct(STAR_VECTORS[i], STAR_VECTORS[j])
            e_ij = uncertaintyAngularDistance(STAR_CENTROIDS_UNCERTAINTY[i], STAR_CENTROIDS_UNCERTAINTY[j])
            
            # Creating range <R_ij>
            r_ij = [d_ij - e_ij, d_ij + e_ij]
            
            # Finding indices of all the elements in <REF_ARR> whose angular distances lie within <R_ij>
            ind = np.where( (REF_ARR[:, 2] >= r_ij[0]) & (REF_ARR[:,2] <= r_ij[1]) )
            
            # Appending matched catalogue star IDs to the corresponding image star
            for k in REF_ARR[ind]:
                s1, s2 = k[0], k[1]
                VOTE_LIST[i, 1].append(s1)
                VOTE_LIST[i, 1].append(s2)
                VOTE_LIST[j, 1].append(s1)
                VOTE_LIST[j, 1].append(s2)


    # Generating array of <VOTE_LIST_2>
    # Column 1 -> Integer numbers from 0 to NUM_STARS
    # Column 2 -> ID of catalogue star that was repeated the most time for the coressponding image star
    # Column 3 -> Number of Votes when the validation step runs on the 'most probable' ID'ed catalogue star
    temp = np.arange(0, NUM_STARS)
    VOTE_LIST_2 = np.vstack((temp, np.zeros_like(temp),np.zeros_like(temp))).T


    # Appending the value of the most repeated catalogue star ID from list of voted stars
    for i in range(NUM_STARS):
        
        # If no catalgoe star has voted on the image star, zero value is set
        VOTE_LIST_2[i,1] = mode(VOTE_LIST[i,1])[0][0] if mode(VOTE_LIST[i,1])[0].shape != (0,) else 0
        

    # Running second iteration of Geometric Voting Algorithm - <validation step>
    for i in range(NUM_STARS):
        for j in range(i+1, NUM_STARS):
            d_ij = dotProduct(STAR_VECTORS[i], STAR_VECTORS[j])
            e_ij = uncertaintyAngularDistance(STAR_CENTROIDS_UNCERTAINTY[i], STAR_CENTROIDS_UNCERTAINTY[j])
            r_ij = [d_ij - e_ij, d_ij + e_ij]
            
            # Reading the 'most probable' catalogue star ID of corresponding image star
            s1, s2 = VOTE_LIST_2[i, 1], VOTE_LIST_2[j, 1]
            
            # Ignoring the case when the corresponding catalogue star ID for a given image star == 0
            if s1 == 0 or s2 == 0:
                continue
            
            # Finding angular distance between the 'most probable' stars from <REF_ARR>
            ind1 = np.where( (REF_ARR[:, 0] == s1) & (REF_ARR[:,1] == s2) ) 
            
            #Accounting for case when <REF_ARR> does not have the angular distance between the specified catalogue star IDs
            if ind1[0].shape != (0,):
                if REF_ARR[ind1][0,2]>r_ij[0] and REF_ARR[ind1][0,2]<r_ij[1]:
                    VOTE_LIST_2[i,2] +=1
                    VOTE_LIST_2[j,2] +=1
                continue
            
            # Repeating the above step on interchanged-columns
            ind2 = np.where( (REF_ARR[:, 0] == s2) & (REF_ARR[:,1] == s1) )
            #Accounting for case when <REF_ARR> does not have the angular distance between the specified catalogue star IDs
            if ind2[0].shape != (0,):
                if REF_ARR[ind2][0,2]>r_ij[0] and REF_ARR[ind2][0,2]<r_ij[1]:
                    VOTE_LIST_2[i,2] +=1
                    VOTE_LIST_2[j,2] +=1
                    
    return VOTE_LIST_2

# Initializing star - catalogue 
CATALOGUE = pd.read_csv("Modified Star Catalogue.csv")
PROCESSED_CATALOGUE= pd.read_csv('Reduced_Catalogue.csv')

# Generating reference catalogue with angular distance of form < cos(theta) >
REFERENCE  = pd.DataFrame(columns=['Star_ID1', 'Star_ID2', 'Ang_Distance'])
REFERENCE['Star_ID1'], REFERENCE['Star_ID2'] = PROCESSED_CATALOGUE['Star_ID1'], PROCESSED_CATALOGUE['Star_ID2']
REFERENCE['Ang_Distance'] = PROCESSED_CATALOGUE.apply(cos, axis = 1)

# Sorting reference catalogue w.r.t <Ang_Distance> in ascending order
REFERENCE.sort_values('Ang_Distance' ,ascending=True, inplace=True)

# Converting reference catalogue to numpy array for faster implementation
REF_ARR = REFERENCE.to_numpy()


# Initializing centroid data and the corresponding centroid uncertainty data generated from running feature extraction
### Example Initialization
centroid = np.random.random((5,2))*2
centroid_uncertainty = np.random.random(5)*0.0001

result = starMatchAlgo(centroid, centroid_uncertainty)
print(result)

###############################################
# --> STATS
'''
%timeit starMatchAlgo(np.random.random((40,2))*2, np.random.random(40)*0.0001)
3.55 s ± 112 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit starMatchAlgo(np.random.random((20,2))*2, np.random.random(20)*0.0001)
851 ms ± 40 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit starMatchAlgo(np.random.random((10,2))*2, np.random.random(10)*0.0001)
205 ms ± 9.44 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit starMatchAlgo(np.random.random((5,2))*2, np.random.random(5)*0.0001)
46.2 ms ± 1.13 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
'''
###############################################
