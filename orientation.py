#%%
from matplotlib import markers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as m



def gen_arrow_head_marker(alpha,beta,gamma):
   
    arr = np.array([[.1, .3, .1], [.1, -.3, .1], [1, 0, 1]])  # arrow shape
    
    # alpha is the x axis angle
    # beta is the y axis angle
    # gamma is the z axis angle
    alpha = alpha / 180 * np.pi
    beta = beta / 180 * np.pi
    gamma = gamma / 180 * np.pi
    
    # three dimension rotation matrix
    Rx = np.matrix([[ 1, 0           , 0           ],
                    [ 0, m.cos(alpha),-m.sin(alpha)],
                    [ 0, m.sin(alpha), m.cos(alpha)]])
    
    Ry =  np.matrix([[ m.cos(beta), 0, m.sin(beta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(beta), 0, m.cos(beta)]])
    
    Rz =  np.matrix([[ m.cos(gamma), -m.sin(gamma), 0 ],
                   [ m.sin(gamma), m.cos(gamma) , 0 ],
                   [ 0           , 0            , 1 ]])
    
    # right hand rule calculate ZXY rotation matrix
    R =  Rz * Ry * Rx
    
    # rotates the arrow
    arr = np.matmul(arr, R)  

#%%
def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[.1, .3], [.1, -.3], [1, 0]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))

    arrow_head_marker = mpl.path.Path(arr)
    return arrow_head_marker, scale

#%%

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

point = np.array([x,y,z])

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.set_aspect('equal'
              
              )
for i in range(len(point[0])):
    mar = gen_arrow_head_marker(30)
    ax.scatter(point[0][i],point[1][i],point[2][i] )


# %%
