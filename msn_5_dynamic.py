from math import sqrt
from networkx.algorithms.cluster import average_clustering, triangles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import networkx as nx
import matplotlib.animation as animation
import networkx.algorithms.community as nx_comm

# Parameters start
central_position = 100
EPSILON = 0.1
H_alpha = 0.2
H_beta = 0.9
C1_ALPHA = 10
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)
C1_BETA = 1500
C2_BETA = 2*np.sqrt(C1_BETA)
C1_GAMMA = 15
C2_GAMMA = 2 * np.sqrt(C1_GAMMA)


# --------------algorithm properties-----------------------#

N = 100  # Number of sensor nodes
M = 2  # Space dimensions
D = 10  # Desired distance among nodes,i.e. algebraic constraints
K = 1.2 # Scaling factor
R = K*D  # Interaction range

D_prime = D * 0.6 # desired distance between obstacles and nodes
R_prime = K * D_prime # Interaction range with obstacles

DELTA_T = 0.01
A = 5
B = 5
C = np.abs(A-B)/np.sqrt(4*A*B)
ITERATION = 2000
POSITION_X = np.zeros([N, ITERATION])
POSITION_Y = np.zeros([N, ITERATION])

AVERAGE_VELOCITY = np.zeros([1,ITERATION]) # the average speed for each iter
MAX_V = np.zeros([1,ITERATION])# the max speed of points for each iter
AVERAGE_X_POSITION = np.zeros([1,ITERATION]) # average X 
MAX_X_VELOCITY = np.zeros([1,ITERATION]) # max speed X
ORIENTATION = np.random.rand(N,ITERATION)*360

markersize=5
SNAPSHOT_INTERVAL = 25

nodes = np.random.rand(N, M) * central_position -50 # nodes initial position,and instant position 
nodes_velocity = np.zeros([N, M])
nodes_velocity[:,0] = 20


# adjacency_matrix = np.zeros([N, N])
a_ij_matrix = np.zeros([N, N])
velocity_magnitudes = np.zeros([N, ITERATION])
connectivity = np.zeros([ITERATION, 1])
fig_counter = 0

# target
q_mt = np.array([350, 55])

# position of obstacles, could be adding more obstacles
obstacles = np.array([[200, 100],[200,0]] )
# Obstacle radius
Rk = np.array([[20],[20]])
num_obstacles = obstacles.shape[0]

# target_points = np.zeros([ITERATION, M])
center_of_mass = np.zeros([ITERATION, M])


#-----------Build networkx graph ---------------------------#
G = nx.Graph()
nodes_list = [i for i in range(len(nodes))]
G.add_nodes_from(nodes_list)
for i in range(0,N):
    G.nodes[i]['pos'] = (nodes[i][0],nodes[i][1])

#----------------network properties------------------------#
DEGREE_CENTRALITY = np.zeros([N, ITERATION])
MAX_DEGREE = np.zeros([1, ITERATION])
MAX_DEGREE_NODE  = np.zeros([1, ITERATION])
MAX_DEGREE_CENTRALITY = np.zeros([1, ITERATION])
MAX_DEGREE_CENTRALITY_NODE = np.zeros([1, ITERATION])

CLUS_COEF = np.zeros([N, ITERATION])
AVERAGE_CLUSTERING = np.zeros([1, ITERATION])
MAX_CLUSTERING = np.zeros([1, ITERATION])
MAX_CLUSTERING_NODE = np.zeros([1, ITERATION])

TRANGLES = np.zeros([N, ITERATION])
NUM_TRIANGLES  = np.zeros([1, ITERATION])
MAX_TRANGLE_NUM = np.zeros([1, ITERATION])
MAX_TRANGLE_NODE = np.zeros([1, ITERATION])



# Parameters end


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

def update_orientation(i,t):
    if t == 0:
        theata = ORIENTATION[i,t]
    else:   
        delta_x = POSITION_X[i,t]-POSITION_X[i,t-1]
        delta_y = POSITION_Y[i,t]-POSITION_Y[i,t-1]
        theata = math.atan2(delta_y,delta_x)*360/(2*np.pi)
        
    return theata

def sigma_norm(z):
    norm = np.linalg.norm(z)
    val = EPSILON*(norm**2)
    val = np.sqrt(1 + val) - 1
    val = val/EPSILON
    return val

r_beta = sigma_norm(R_prime)
d_beta = sigma_norm(D_prime)
s = 1


def create_adjacency_matrix():
    adjacency_matrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                # val = nodes[i] - nodes[j]
                distance = np.linalg.norm(nodes[i] - nodes[j])
                if distance <= R:
                    adjacency_matrix[i, j] = 1
    return adjacency_matrix


def plot_deployment():
    fig = plt.figure('initial deployment')
    ax = fig.add_subplot()
    for i in range(N):
        theata_i = ORIENTATION[i,0]
        marker,scale = gen_arrow_head_marker(theata_i)
        ax.plot(nodes[i, 0], nodes[i, 1], marker = marker,ms = markersize)

def get_edge_list():
    nodes_edge_list =[]
    for i in range(0, N):
        for j in range(i+1, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                nodes_edge_list.append((i,j,round(distance))) # adding tuple edges
                
    return nodes_edge_list
    

def plot_neighbors(t,f,nodes_edge_list):
    ax = f.add_subplot(221)
    ax.title.set_text('time {} s'.format(t*DELTA_T))
    ax.plot(q_mt[0], q_mt[1], 'ro', color='green')
    ax.plot(center_of_mass[0:t, 0], center_of_mass[0:t, 1], color='black')
    for i in range(0, num_obstacles):
        ax.add_artist(plt.Circle((obstacles[i, 0],obstacles[i, 1]), Rk[i], color='red'))
        
    # plot agents
    for i in range(0, N):
        theata_i = ORIENTATION[i,t]
        marker,scale = gen_arrow_head_marker(theata_i)
        ax.plot(nodes[i, 0], nodes[i, 1], marker = marker,ms = markersize)
        
    # plot edges
    for e in nodes_edge_list:
        x = e[0]
        y = e[1]
        ax.plot([nodes[x, 0], nodes[y,0]],
                [nodes[x, 1], nodes[y,1]],'b-',lw=0.5)
        


def get_triangles_properties():
    
    # triangles
    trangles = nx.triangles(G)
    num_triangles = sum(triangles(G).values())
    max_trangle_num = max(trangles.values())
    max_trangle_node = max(trangles, key = lambda k : trangles.get(k))  
    
    return trangles,num_triangles,max_trangle_num,max_trangle_node

def get_clustering_property():
        # clustering coefficient
    clus_coef = nx.clustering(G) 
    average_clustering = nx.average_clustering(G)
    max_clustering = max(clus_coef.values())
    max_clustering_node =max(clus_coef, key = lambda k : clus_coef.get(k))  
    
    return clus_coef,average_clustering, max_clustering, max_clustering_node

def get_degree_property():
    # max node degree
    degree = dict(G.degree())
    max_degree = max(degree.values())
    max_degree_node = max(degree, key = lambda k : degree.get(k))
    
    # degree centrality 
    degree_centrality = nx.degree_centrality(G)
    max_degree_centrality = max(degree_centrality.values())
    max_degree_centrality_node = max(degree_centrality,  key = lambda k : degree_centrality.get(k))
    
    return degree_centrality,max_degree, max_degree_node, max_degree_centrality, max_degree_centrality_node

def record_graph_properties(t):
        #-------------- get graph properties --------------#
    
    # triangles
    trangles,num_triangles,max_trangle_num,max_trangle_node = get_triangles_properties()
    TRANGLES[:,t] = list(trangles.values())
    NUM_TRIANGLES[:,t] = num_triangles
    MAX_TRANGLE_NUM[:,t] =  max_trangle_num
    MAX_TRANGLE_NODE[:,t] = max_trangle_node
    
    # clustering coefficient
    clus_coef,average_clustering, max_clustering, max_clustering_node = get_clustering_property()
    CLUS_COEF[:,t] = list(clus_coef.values())
    AVERAGE_CLUSTERING[:,t] = average_clustering
    MAX_CLUSTERING[:,t] = max_clustering
    MAX_CLUSTERING_NODE[:,t] =  max_clustering_node
    
    # max node degree
    degree_centrality,max_degree, max_degree_node, max_degree_centrality, max_degree_centrality_node = get_degree_property()
    DEGREE_CENTRALITY[:,t] = list(degree_centrality.values())
    MAX_DEGREE[:,t] = max_degree
    MAX_DEGREE_NODE[:,t] = max_degree_node
    MAX_DEGREE_CENTRALITY[:,t] = max_degree_centrality
    MAX_DEGREE_CENTRALITY_NODE[:,t] =max_degree_centrality_node
    
    return trangles,num_triangles,max_trangle_num,max_trangle_node,\
            clus_coef,average_clustering, max_clustering, max_clustering_node,\
                degree_centrality,max_degree, max_degree_node, max_degree_centrality, max_degree_centrality_node
    
    
    
def draw_network_interactive(t,fig):
    
    trangles,num_triangles,max_trangle_num,max_trangle_node,\
                clus_coef,average_clustering, max_clustering, max_clustering_node,\
                    degree_centrality,max_degree, max_degree_node, max_degree_centrality, max_degree_centrality_node = record_graph_properties(t)
    #--------------Drawing----------------------------#
    
    ax1 = fig.add_subplot(234)
    ax1.title.set_text('network shape')
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx(G,pos = pos,ax=ax1,node_size = 100,font_size = 6 ) 
    
    # ------ Draw labels of the weight -----------------#
    
    # labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    
    # -------------draw text box------------------------
    
    # textstr = 'shape'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,fontsize=14,
    #         verticalalignment='top', bbox=props)

    ax2 = fig.add_subplot(235) 
    ax2.title.set_text('network shape')
    ax2.axis('off')
    # column labels
    col_labels =['attributes','value','max value node'] 
    
    # attributes data
    data = [num_triangles,
            average_clustering,
            max_clustering,
            max_degree,
            max_degree_centrality
            ]
    # the data of the table
    celltext=np.array([['number of triangles', data[0], max_trangle_node],
                       ['average clustering coefficient', data[1], max_clustering_node],
                       ['max clustering coefficient', data[2], max_clustering_node],
                       ['maximum node degree', data[3], max_degree_node],
                       ['max_degree_centrality', data[4],max_degree_centrality_node]
                       ])
    # utilize pandas DataFrame
    df = pd.DataFrame(celltext, columns=col_labels)
    # draw the table
    ax2.table(cellText=df.values, colLabels=df.columns, loc='center')

    
    
    # plot attributes
    ax3 = fig.add_subplot(236)
    #------- details of clustering coefficient--------------#
    ax3.title.set_text('clustering coefficietn for each node')
    plt.xlabel('node number')
    plt.ylabel('clustering coefficient for each node')
    plt.bar(list(clus_coef.keys()),clus_coef.values(),color='b')
    
    #-----------------triangles for each node----------------------#
    # ax3.title.set_text('triangles for each node')
    # plt.xlabel('node number')
    # plt.ylabel('number of traingles pass through')
    # plt.bar(list(trangles.keys()),trangles.values(),color='g')

def if_strongly_connected(G):
    
    return nx.is_strongly_connected(G)

def if_connected():
    return nx.is_connected(G)

def node_connected_component(G, n):
    nodes_set = nx.node_connected_component(G,n)
    return nodes_set

def number_connected_components(G):
    
    return nx.number_connected_components(G)


def detect_colliding_node(G):
    
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
    
    
    

    
# update the graph, extract the properties of the flock
def update_graph(edge_list):
    
    # extract the pure edges relationship 
    edges = [(e[0], e[1] ) for e in edge_list]
    
    edges_old = list(G.edges())
    edges_remove= [ e for e in edges_old if e not in edges] #edges to be removed
    
    G.update(edges = edges,nodes =nodes_list) # update graph
    
    G.remove_edges_from(edges_remove)
    
    # update nodes position 
    for i in range(0,N):
        G.nodes[i]['pos'] = tuple(nodes[i])
        
    # update edge weight, must update edge first then weight, update will overwrite the edge attributes
    for e in edge_list:
        x = e[0] # source_node
        y = e[1] # targe_node
        weight = e[2]
        G[x][y]['weight'] = weight
        
        
  

def bump_function(z,H):
    if 0 <= z < H:
        return 1
    elif H <= z < 1:
        val = (z-H)/(1-H)
        val = np.cos(np.pi*val)
        val = (1+val)/2
        return val
    else:
        return 0


def sigma_1(z):
    val = 1 + z **2
    val = np.sqrt(val)
    val = z/val
    return val


def phi(z):
    val_1 = A + B
    val_2 = sigma_1(z + C)
    val_3 = A - B
    val = val_1 * val_2 + val_3
    val = val / 2
    return val


def phi_alpha(z):
    input_1 = z/sigma_norm(R)  # Sigma norm of R is R_alpha
    input_2 = z - sigma_norm(D)  # Sigma norm of D is D_alpha
    val_1 = bump_function(input_1,H_alpha)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val


def phi_beta(z):
    val1 = bump_function(z/d_beta,H_beta)
    val2 = sigma_1(z-d_beta) - 1
    return val1 * val2


def get_a_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm)/sigma_norm(R)
    val = bump_function(val_2,H_alpha)
    return val


def get_n_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm**2
    val = val_1/np.sqrt(val_2)
    return val


# get ui_beta from lemma 4
def get_ui_beta(i,q_i, p_i):
    
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    ui_beta = 0
    # for each obstacles
    for k in range(num_obstacles):
        yk = obstacles[k]
        a_k = (q_i - yk) / np.linalg.norm(q_i-yk)
        mu = Rk[k] / np.linalg.norm(q_i-yk)
        P = 1 - np.matmul(a_k.T, a_k)
        
        q_i_k = mu*q_i + (1-mu) * yk
        p_i_k = mu * P * p_i
        
        distance = np.linalg.norm(q_i_k - q_i)
        # if distance < R_prime:
        n_i_k = (q_i_k - q_i) /(np.sqrt(1 + EPSILON * (np.linalg.norm(q_i_k-q_i))**2))
        b_i_k = bump_function(sigma_norm(q_i_k-q_i)/d_beta,H_beta)
            
        sum_1 +=  phi_beta(sigma_norm(q_i_k-q_i)) * n_i_k 
        
        sum_2 +=  b_i_k * (p_i_k-p_i)
            
    ui_beta = C1_BETA * sum_1 + C2_BETA * sum_2
    
    return ui_beta
    

def get_u_i(i,q_i,p_i):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        distance = np.linalg.norm(nodes[j] - nodes[i])
        if distance <= R:
            phi_alpha_val = phi_alpha(sigma_norm(nodes[j] - nodes[i]))
            sum_1 += phi_alpha_val * get_n_ij(i, j)
            sum_2 += get_a_ij(i, j) * (nodes_velocity[j] - nodes_velocity[i])
            
    ui_alpha = C1_ALPHA * sum_1 + C2_ALPHA * sum_2
                         
    ui_gamma = - C1_GAMMA * sigma_1(nodes[i] - q_mt) # - C2_GAMMA * (p_i - 0)
    
    ui_beta = get_ui_beta(i,q_i,p_i) # ui_beta 不参与j循环
           
    ui =  ui_alpha + ui_beta + ui_gamma

    return ui




def get_positions_interactive():
               
    fig = plt.figure('attributes',figsize=(18,20))
    ani = []   
    flag = False
    counter = 0
    plt.ion()
    for t in range(0, ITERATION):
        # print(np.linalg.matrix_rank(adjacency_matrix))
        adjacency_matrix = create_adjacency_matrix()
        # print(np.linalg.matrix_rank(adjacency_matrix))
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)
        center_of_mass[t] = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1])])
        
        if t == 0:
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
                nodes_edge_list= get_edge_list()
                plot_neighbors(t,fig,nodes_edge_list)
        else:
            for i in range(0, N):
                # p_i == old_velocity in  the paper
                # q_i === old_position
                old_velocity = nodes_velocity[i, :]
                old_position = np.array([POSITION_X[i, t-1],
                                         POSITION_Y[i, t-1]])
                
                u_i = get_u_i(i, old_position, old_velocity)
                
                #update position
                new_position = old_position + DELTA_T * old_velocity + (DELTA_T ** 2 / 2) * u_i
                [POSITION_X[i, t], POSITION_Y[i, t]] = new_position
                nodes[i, :] = new_position

                # update velocity
                new_velocity = (new_position - old_position) / DELTA_T
                nodes_velocity[i, :] = new_velocity
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)
                
                #update orientation
                ORIENTATION[i,t]=update_orientation(i,t)
                record_graph_properties(t)
                
        nodes_edge_list= get_edge_list()       
        plot_neighbors(t,fig,nodes_edge_list)
        update_graph(nodes_edge_list)
        draw_network_interactive(t,fig)
        plot_dynamic_speed(t,fig)
        
        
        if (t) % SNAPSHOT_INTERVAL == 0:
            plt.savefig('../img/time_{}_.png' .format(counter))
            counter += 1
            
        plt.pause(0.001)
        fig.clf() 
    plt.ioff()
    plt.show()
    
    
    


    
def plot_dynamic_speed(t,fig):
    vel_plot = fig.add_subplot(222,title='maximum speed point in the flock',xlabel='the floak average position',ylabel='speed')
    MAX_V[:,t] =max(velocity_magnitudes[:, t])
    AVERAGE_VELOCITY[:,t] = np.average(velocity_magnitudes[:, t])
    
    # return max velocity node index
    max_v_index = velocity_magnitudes[:, t].argmax()
    AVERAGE_X_POSITION[:,t] = np.average(POSITION_X[:,t])
    MAX_X_VELOCITY[:,t] = POSITION_X[max_v_index,t]
    
    vel_plot.plot(AVERAGE_X_POSITION[0,:],MAX_V[0,:],'bo',label = 'max speed')
    vel_plot.plot(AVERAGE_X_POSITION[0,:],AVERAGE_VELOCITY[0,:],'rx',label = 'average speed')
    vel_plot.legend(loc='upper right')

def plot_trajectory():
    for i in range(0, N):
        # arr = np.array([POSITION_X[i, :], POSITION_Y[i, :]])
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])
        # plt.show()
    plt.show()
    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)


def plot_velocity():
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]
        # velocity_i = 1/velocity_i
        plt.plot(velocity_i)
    plt.show()
    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)


def plot_connectivity():
    plt.plot(connectivity)
    plt.show()
    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)
    # l=1
# plot_initial_deployment()
# create_adjacency_matrix()


def plot_center_of_mass():
    plt.plot(center_of_mass[:, 0], center_of_mass[:, 1])
    plt.show()

if __name__ == '__main__':
    plot_deployment()
    get_positions_interactive()
    plot_trajectory()
    plot_velocity()
    plot_connectivity()
    plot_center_of_mass()

#%%


