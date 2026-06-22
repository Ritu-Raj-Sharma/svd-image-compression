from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

A = imread('demo.jpeg')
X = np.mean(A, -1); #convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.title('Original image')
plt.show()

U, S, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

j = 0
for r in (5, 20, 100):
    #Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('rank = ' +str(r))
    plt.show()

#Singular value graph
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

#Cumulative sum graph
plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()