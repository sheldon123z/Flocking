## graph methods

import networkx as nx
import numpy as np
import shutil
import os
import matplotlib as mpl
import datetime

def if_strongly_connected(G):
    
    return nx.is_strongly_connected(G)

def if_connected():
    return nx.is_connected(G)

def node_connected_component(G, n):
    nodes_set = nx.node_connected_component(G,n)
    return nodes_set

def number_connected_components(G):
    
    return nx.number_connected_components(G)


def detect_colliding_node(G,N,D,nodes):
    
    collide_num = 0
    
    for i in range(0,N):
        for j in range(i+1,N):
            distance = np.linalg.norm(nodes[i] - nodes[j])
            if distance < D:
                pass
            
    pass

def detect_component():
    pass

def divide_communites():
    
    pass 
    
def deviation_energy():

    pass


# TODO leader需要进一步确定是否实时更改
def determine_leader(t,DISTANCE):
    
    d_max = max(DISTANCE[:,t])
    d_min = min(DISTANCE[:,t])
    d_middle = (d_max+d_min)/2
    
    temp = [(i,DISTANCE[i:t]-d_middle) for i in range(0,N)]
    
    #sort by Distance
    temp.sort(key=lambda y:y[1],reverse = True )
    
    return temp[0][0]
    
def leader_pos(t,nodes):
    
    lattice_leader = determine_leader(t)
    
    leader_pos = nodes[lattice_leader]
    
    return leader_pos
    

# Change the marker shape
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


def get_file_path():
    ISOTIMEFORMAT = '%Y-%m-%d-%H-%M-%S'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    img_path = './img/img_' + theTime
    flocking_path = img_path + '/flocking'
    attributes_path =img_path + '/attributes'
    properties_path = img_path + '/properties'
    speed_path = img_path + '/speed'
    if not os.path.exists(img_path):
        os.mkdir(img_path)
        
    return img_path,flocking_path,attributes_path,properties_path,speed_path

def clear_img_path(*file_path):
    ISOTIMEFORMAT = '%Y-%m-%d-%H-%M-%S'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    for path in file_path:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)