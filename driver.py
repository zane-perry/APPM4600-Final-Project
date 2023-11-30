######################################################################## imports
#######################################################################

## native python libraries
##

import numpy as np
import re
import math
import time
from operator import itemgetter

## our functions
##

from helpers import compareSVDandQR, generate_matrix

######################################################################### driver
#######################################################################

# create a random (m x n) matrix A with rank deficiency rDef
m = 500
n = 500
rDef = 420
A = generate_matrix(m, n, rDef)

# - set precision of factorizations (printed when fullOutput=True)
# - decrement this if the output looks fucked up
decimalPlaces = 3

# compare results of SVD and QR approximations
compareSVDandQR(A, precision=decimalPlaces, fullOutput=False)

