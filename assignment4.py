import numpy as np
import cv2
from mpi4py import MPI

kernel = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                   [0, 2, 3, 5, 5, 5, 3, 2, 0],
                   [3, 3, 5, 3, 0, 3, 5, 3, 3],
                   [2, 5, 3, -12, -23, -12, 3, 5, 2],
                   [2, 5, 0, -23, -40, -23, 0, 5, 2],
                   [2, 5, 3, -12, -23, -12, 3, 5, 2],
                   [3, 3, 5, 3, 0, 3, 5, 3, 3],
                   [0, 2, 3, 5, 5, 5, 3, 2, 0],
                   [0, 0, 3, 2, 2, 2, 3, 0, 0]])

filepath = 'pepper.ascii.pgm'

image = cv2.imread(filepath,0)


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    M = 64
    N = 64
    tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
    data = tiles
else:
    data = None

comm.Barrier()

data = comm.scatter(data)

convolvedTiles = cv2.filter2D(data,-1,kernel)

gatheredData = comm.gather(convolvedTiles)

comm.Barrier()

if rank == 0:
    finalImage = np.array(gatheredData)
    finalImage = np.hstack(finalImage)
    finalImage = np.hsplit(finalImage,4)
    finalImage = np.vstack(finalImage)

    cv2.imshow('rejoinedTiles',finalImage)
    cv2.imwrite('SampleOutput.ascii.pgm', finalImage)
    cv2.waitKey(30000)
    print(np.shape(finalImage))