# @brief Contain test code for "outTisLib.py"

from myTisLib import *
import numpy as np


#######################################################################################
####################################### Padding #######################################
#######################################################################################

f=np.array([ 
    [ 1, 2, 3],
    [ 5, 6, 7]
           ])
mask=np.zeros( (5,5) ); 


# TEST MIRROR PADDING

theoreticalMirrorPaddedF=np.array([
    [6, 5, 5, 6, 7, 7, 6],
    [2, 1, 1, 2, 3, 3, 2],
    [2, 1, 1, 2, 3, 3, 2],
    [6, 5, 5, 6, 7, 7, 6],
    [6, 5, 5, 6, 7, 7, 6],
    [2, 1, 1, 2, 3, 3, 2]])

practicalPaddedF=imagePadding(f,mask,'mirror')

# compare results 
np.testing.assert_equal(theoreticalMirrorPaddedF, practicalPaddedF)

# TEST ZERO PADDING

theoreticalZeroPaddedF=np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 3, 0, 0],
    [0, 0, 5, 6, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]])

practicalPaddedF=imagePadding(f,mask,'zero')

# compare results 
np.testing.assert_equal(theoreticalZeroPaddedF, practicalPaddedF)

# TEST REPLICATE PADDING

theoreticalReplicatePaddedF=np.array([
    [1, 1, 1, 2, 3, 3, 3],
    [1, 1, 1, 2, 3, 3, 3],
    [1, 1, 1, 2, 3, 3, 3],
    [5, 5, 5, 6, 7, 7, 7],
    [5, 5, 5, 6, 7, 7, 7],
    [5, 5, 5, 6, 7, 7, 7]])

practicalPaddedF=imagePadding(f,mask,'replicate')

# compare results 
np.testing.assert_equal(theoreticalReplicatePaddedF, practicalPaddedF)

# TEST CIRCULAR PADDING

theoreticalCircularPaddedF=np.array([
    [2, 3, 1, 2, 3, 1, 2],
    [6, 7, 5, 6, 7, 5, 6],
    [2, 3, 1, 2, 3, 1, 2],
    [6, 7, 5, 6, 7, 5, 6],
    [2, 3, 1, 2, 3, 1, 2],
    [6, 7, 5, 6, 7, 5, 6]])

practicalPaddedF=imagePadding(f,mask,'circular')

# compare results 
np.testing.assert_equal(theoreticalCircularPaddedF, practicalPaddedF)


#######################################################################################
####################################### 2D Conv #######################################
#######################################################################################


f=np.array( [
    [0,1,2],
    [3,4,5],
    [6,7,8]] ,dtype='uint8') 


# ========================================================================
# test with a 1x1 blank mask (no change)
blank_mask=np.zeros( (1,1) , dtype='int8'); 
blank_mask[blank_mask.shape[0]//2,blank_mask.shape[1]//2]=1

g=conv2D(f,blank_mask,norm=False)
# compare results 
np.testing.assert_equal(f, g)


# ========================================================================
# test with a 3x3 blank mask (no change)
blank_mask=np.zeros( (3,3) , dtype='int8'); 
blank_mask[blank_mask.shape[0]//2,blank_mask.shape[1]//2]=1

g=conv2D(f,blank_mask,norm=False)
# compare results 
np.testing.assert_equal(f, g)

# ========================================================================
# test with a 3x7 blank mask (no change)
blank_mask=np.zeros( (3,7) , dtype='int8'); 
blank_mask[blank_mask.shape[0]//2,blank_mask.shape[1]//2]=1

g=conv2D(f,blank_mask,norm=False)
# compare results 
np.testing.assert_equal(f, g)

# ========================================================================
# test with a 7x5 blank mask (no change)
blank_mask=np.zeros( (7,5) , dtype='int8'); 
blank_mask[blank_mask.shape[0]//2,blank_mask.shape[1]//2]=1

g=conv2D(f,blank_mask,norm=False)
# compare results 
np.testing.assert_equal(f, g)

# ========================================================================
# test with a 3x3 mask for sum computation 
mask=np.ones( (3,3) , dtype='int8'); 

g=conv2D(f,mask,norm=False)

theoreticalG=np.array( [
    [ 8,15,12],
    [21,36,27],
    [20,33,24]] ,dtype='uint8') 

# compare results 
np.testing.assert_equal(theoreticalG, g)















print("All tests passed successfully!")