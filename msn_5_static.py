import gm
from math import sqrt
from networkx.algorithms.cluster import average_clustering, triangles
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import matplotlib.animation as animation
import networkx.algorithms.community as nx_comm
import warnings
from matplotlib.patches import Rectangle
warnings.filterwarnings('ignore')

####################################################################
####################################################################
#-----------------------PARAMETERS BEGIN---------------------------#
####################################################################
####################################################################

# ------------------------ALGORITHM SETTINGS--------------------------#
N = 100  # Number of sensor nodes
M = 2  # Space dimensions
D = 10  # Desired distance among nodes,i.e. algebraic constraints
K = 1.2 # Scaling factor
R = K*D  # Interaction range
A = 5
B = 5
C = np.abs(A-B)/np.sqrt(4*A*B)
D_prime = D * 3 # desired distance between obstacles and node
R_prime = K * D_prime # Interaction range with obstacles
EPSILON = 0.1
H_alpha = 0.2
H_beta = 0.9
C1_ALPHA = 10
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)
C1_BETA = 150
C2_BETA = 2*np.sqrt(C1_BETA)
C1_GAMMA = 120
C2_GAMMA = 2 * np.sqrt(C1_GAMMA)
DELTA_T = 0.01 # time interval for calculating speed
ITERATION = 5000 # total step number

# ---------------------STATISTICAL SETTINGS---------------------------#
# whole process parameters recording
POSITION_X = np.zeros([N, ITERATION])  # X position of each agent
POSITION_Y = np.zeros([N, ITERATION]) # Y position of each agent
AVERAGE_VELOCITY = np.zeros([1,ITERATION]) # the average speed for each iter
MAX_V = np.zeros([1,ITERATION])# the max speed of points for each iter
AVERAGE_X_POSITION = np.zeros([1,ITERATION]) # average X 
MAX_VELOCITY_X = np.zeros([1,ITERATION]) # max speed X
VELOCITY_MAGNITUDES = np.zeros([N, ITERATION]) # velocity of each agent 
# acceleration 
ACCELERACTION_X = np.zeros([N, ITERATION])
AVERAGE_X_ACC = np.zeros([1, ITERATION])
AVERAGE_Y_ACC = np.zeros([1, ITERATION])
ACCELERACTION_Y = np.zeros([N, ITERATION])
ACCELERACTION_MAGNITUDES = np.zeros([N, ITERATION])
AVERAGE_ACC_MAGNITUDES = np.zeros([1, ITERATION])
CONNECTIVITY = np.zeros([ITERATION, 1]) # connectivity for the network
ORIENTATION = np.random.rand(N,ITERATION)*360 # direction of each agent while flying
DEGREE_CENTRALITY = np.zeros([N, ITERATION])
DEGREES = np.zeros([N, ITERATION])
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
DISTANCE = np.array([N, ITERATION])

#---------------------AGENT/OBSTACLES SETTINGS------------------------#

# target position
target_position = np.array([650, 50])
# position of obstacles, could be adding more obstacles
obstacles = np.array([[400, 90],[400,20]] )
# Obstacle radius
Rk = np.array([[1],[1]])
num_obstacles = obstacles.shape[0]

# target_points = np.zeros([ITERATION, M])
center_of_mass = np.zeros([ITERATION, M])
# initial position 
formation_range = 200
nodes = np.random.rand(N, M) * formation_range # nodes initial position,and instant position 
nodes_velocity = np.zeros([N, M])

initial_x = 0
initial_y = 0
for i in range(0,N):
    nodes[i][0] -= initial_x
    nodes[i][1] -= initial_y
        
#-------------------------GRAPH SETTINGS -----------------------------#
G = nx.Graph()
nodes_list = [i for i in range(len(nodes))] # must build the graph from nodes rather than edges
G.add_nodes_from(nodes_list)
# adding pos to each node
for i in range(0,N):
    G.nodes[i]['pos'] = (nodes[i][0],nodes[i][1])
# adjacency_matrix = np.zeros([N, N])
a_ij_matrix = np.zeros([N, N])

#---------------------------UTILITIES--------------------------------#
SNAPSHOT_INTERVAL = 100 # screenshot interval 
markersize=8 # node marker size
fig_counter = 0
img_path,flocking_path,attributes_path,properties_path,speed_path = gm.get_file_path()

####################################################################
####################################################################
#-----------------------PARAMETERS END-----------------------------#
####################################################################
####################################################################




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

def get_edge_list():
    nodes_edge_list =[]
    for i in range(0, N):
        for j in range(i+1, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                nodes_edge_list.append((i,j,round(distance))) # adding tuple edges
                
    return nodes_edge_list
    
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
    DEGREES[:,t] = np.array(list(dict(G.degree()).values()))
    MAX_DEGREE[:,t] = max_degree
    MAX_DEGREE_NODE[:,t] = max_degree_node
    MAX_DEGREE_CENTRALITY[:,t] = max_degree_centrality
    MAX_DEGREE_CENTRALITY_NODE[:,t] =max_degree_centrality_node

    
    AVERAGE_X_POSITION[:,t] = np.average(POSITION_X[:,t])
    AVERAGE_X_ACC[:,t] =np.average(ACCELERACTION_X[:,t])
    AVERAGE_Y_ACC[:,t] =np.average(ACCELERACTION_Y[:,t])
    
    # draw max velocity node
    max_v_index = VELOCITY_MAGNITUDES[:, t].argmax() # return max velocity node index
    MAX_VELOCITY_X[:,t] = POSITION_X[max_v_index,t]
    MAX_V[:,t] = max(VELOCITY_MAGNITUDES[:, t])
    
    AVERAGE_VELOCITY[:,t] = np.average(VELOCITY_MAGNITUDES[:, t])
    # CONNECTIVITY[t] = nx.average_node_connectivity(G)
    
    
    # distance for each node to destination
    for i in range(0,N):
        distance = np.linalg.norm(nodes[i] - target_position)
    DISTANCE[i:t] = distance

    return     [trangles, num_triangles, max_trangle_num, max_trangle_node,\
                clus_coef, average_clustering, max_clustering, max_clustering_node,\
                degree_centrality, max_degree, max_degree_node, max_degree_centrality,\
                max_degree_centrality_node]



        
# plot stastitical properites
def plot_properties():
    
    fig = plt.figure('Properties',figsize=(20,20))
    # trajectory 
    traject_plot = fig.add_subplot(title ='Trajectory',xlabel='Position X',ylabel='Position Y')
    for i in range(0, N):
        traject_plot.plot(POSITION_X[i, :], POSITION_Y[i, :])
    fig_name = properties_path + '/Trajectory.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    fig.clf()
    
    
    velocity_plot = fig.add_subplot(title='Velocity',xlabel='Iteration',ylabel = 'Velocity')
    for i in range(0, N):
        velocity_plot.plot(VELOCITY_MAGNITUDES[i, :])
        
    fig_name = properties_path + '/Velocity.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    fig.clf()
    
    orientation_plot = fig.add_subplot(121,title='Orientation vs Position',xlabel = 'Position X',ylabel='Orientation')
    orientation2_plot = fig.add_subplot(122,title='Orientation vs Iteration',xlabel = 'Iteration',ylabel='Orientation')
    for i in range(0, N):
        orientation_plot.plot(POSITION_X[i, :], ORIENTATION[i, :])
        orientation2_plot.plot(ORIENTATION[i, :])
        
    fig_name = properties_path + '/Orientation.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    fig.clf()
        
    # triangles_plot = fig.add_subplot(223)
    # triangles_plot.title.set_text('triangles_plot')
    
    # clus_coef_plot = fig.add_subplot(224)
    # clus_coef_plot.title.set_text('clus_coef_plot')

    
    # for i in range(0, N):
        # traject_plot.plot(POSITION_X[i, :], POSITION_Y[i, :])
        # triangles_plot.plot(TRANGLES[i,:]) 
        # clus_coef_plot.plot(CLUS_COEF[i,:])            
        # degree_centrality_plot.plot(DEGREE_CENTRALITY[i,:])   
        # acceleraction_plot.plot(ACCELERACTION_MAGNITUDES[i,:])
        
        
        
                
    # degree_centrality_plot = fig.add_subplot(235)
    # degree_centrality_plot.title.set_text('degree_centrality_plot')
    
    # acceleraction_plot = fig.add_subplot(236)
    # acceleraction_plot.title.set_text('acceleraction_plot')   

    
 

    # acceleraction_plot.plot(MAX_DEGREE)
    # acceleraction_plot.plot(MAX_CLUSTERING)
    # acceleraction_plot.plot(MAX_TRANGLE_NUM)
    # acceleraction_plot.plot(MAX_V)
    # acceleraction_plot.plot(AVERAGE_VELOCITY)
    
    

    #save property image
    # fig_name = properties_path + '/proterties.png'
    # fig.savefig(fig_name, dpi=fig.dpi)


def plot_deployment():
    fig = plt.figure('initial deployment')
    ax = fig.add_subplot()
    for i in range(0,N):
        theata_i = ORIENTATION[i,0]
        marker,scale = gm.gen_arrow_head_marker(theata_i)
        ax.plot(nodes[i, 0], nodes[i, 1], marker = marker,ms = markersize)


# plot neighbors and edges
def plot_neighbors(t,f,nodes_edge_list):
    ax = f.add_subplot()
    ax.set_xlim(0,1000)
    ax.set_ylim(-500,500)
    
    ax.title.set_text('time {} s'.format(t*DELTA_T))
    ax.plot(target_position[0], target_position[1], 'ro', color='green')
    ax.plot(center_of_mass[0:t, 0], center_of_mass[0:t, 1], color='black')
    # ax.add_patch(Rectangle((450,-50),100,100,fc='none',lw = 1,ec ='g' ))
    ax.add_patch(Rectangle((300,-200),400,400,fc='none',lw = 1,ec ='g' ))
    for i in range(0, num_obstacles):
        # ax.add_artist(plt.Circle((obstacles[i, 0],obstacles[i, 1]), Rk[i], color='red'))
        for k in range(len(obstacles[i])):
            ax.scatter(obstacles[i][0],obstacles[i][1],color = 'red',s = D_prime)
        
        
    # plot agents
    for i in range(0, N):
        theata_i = ORIENTATION[i,t]
        marker,scale = gm.gen_arrow_head_marker(theata_i)
        ax.plot(nodes[i, 0], nodes[i, 1], marker = marker,ms = markersize)
        
    # plot edges
    for e in nodes_edge_list:
        start_node = e[0]
        end_node = e[1]
        ax.plot([nodes[start_node, 0], nodes[end_node,0]],
                [nodes[start_node, 1], nodes[end_node,1]],'b-',lw=0.5)


def plot_position_speed(fig):
    
    vel_plot = fig.add_subplot(211,title='maximum speed point in the flock',xlabel='the floak average position',ylabel='speed')
    vel_plot.set_xlim(-100,2000)
    vel_plot.set_ylim(-100,2000)
    vel_plot.plot(AVERAGE_X_POSITION[0,:],MAX_V[0,:],'bo',label = 'max speed')
    vel_plot.plot(AVERAGE_X_POSITION[0,:],AVERAGE_VELOCITY[0,:],'rx',label = 'average speed')
    
    
    acc_x_plot = fig.add_subplot(223,title='accelerations',xlabel='the floak average position',ylabel='accleartion')
    acc_x_plot.plot(AVERAGE_X_POSITION[0,:],AVERAGE_X_ACC[0,:],'gx',label = 'average x acceleration')
    
    acc_y_plot = fig.add_subplot(224,title='accelerations',xlabel='the floak average position',ylabel='accleartion')
    acc_y_plot.plot(AVERAGE_X_POSITION[0,:],AVERAGE_Y_ACC[0,:],'gx',label = 'average y acceleration')
    
    vel_plot.legend(loc='upper right')
    acc_x_plot.legend(loc='upper right')
    acc_y_plot.legend(loc='upper right')
    
    

def draw_network(G,data,fig,t):
    triangles = data[0]
    num_triangles = data[1]
    max_trangle_node = data[3]
    clus_coef = data[4]
    average_clustering = data[5]
    max_clustering = data[6]
    max_clustering_node = data[7]
    max_degree = data[9]
    max_degree_node = data[10]
    max_degree_centrality = data[11]
    max_degree_centrality_node = data[12]
    
    #--------------Drawing----------------------------#
    network_plot = fig.add_subplot(221)
    network_plot.title.set_text('network shape')
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx(G,pos = pos,ax=network_plot,node_size = 100,font_size = 6 ) 
    fig.suptitle('This is the time {} s network'.format(t * DELTA_T))
    
    
    
    degrees_plot =fig.add_subplot(222)
    degrees_plot.title.set_text('degrees')
    node_degrees = dict(G.degree())
    plt.bar(node_degrees.keys(),node_degrees.values(),color='r')
    

    
    # ------ Draw labels of the weight -----------------#
    
    # labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    
    # -------------draw text box------------------------
    # textstr = 'shape'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,fontsize=14,
    #         verticalalignment='top', bbox=props)

    attributes_plot = fig.add_subplot(223) 
    attributes_plot.title.set_text('network attributes and connectivity')
    plt.xlabel('node number')
    plt.ylabel('number of traingles pass through')
    plt.bar(list(triangles.keys()),triangles.values(),color='g')
    attributes_plot.get_xaxis().set_visible(False)
    # column labels
    col_labels =['attributes','value','node num'] 
    
    # attributes data
    data = [num_triangles,
            average_clustering,
            max_degree_centrality
            ]
    # the data of the table
    celltext=np.array([['number of triangles', data[0], max_trangle_node],
                       ['average clustering coefficient', round(data[1],2), max_clustering_node],
                       ['max_degree_centrality', round(data[2],2),max_degree_centrality_node]
                       ])
    # utilize pandas DataFrame
    df = pd.DataFrame(celltext, columns=col_labels)
    # draw the table
    
    table = attributes_plot.table(cellText=df.values, colLabels=df.columns, loc='bottom')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(list(range(len(col_labels))))

    
    
    # plot attributes
    clu_coef_plot = fig.add_subplot(224)
    #------- details of clustering coefficient--------------#
    clu_coef_plot.title.set_text('clustering coefficietn for each node')
    plt.xlabel('node number')
    plt.ylabel('clustering coefficient for each node')
    plt.bar(list(clus_coef.keys()),clus_coef.values(),color='b')
    
    #-----------------triangles for each node----------------------#
    # ax3.title.set_text('triangles for each node')
    # plt.xlabel('node number')
    # plt.ylabel('number of traingles pass through')
    # plt.bar(list(trangles.keys()),trangles.values(),color='g')


# update the graph, extract the properties of the flock
def update_graph(G,edge_list):
    
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

    return G
        
  

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

def sigma_1_gamma(z):
    val = 1 + np.linalg.norm(z)**2
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
    val1 = bump_function(z/D_BETA,H_beta)
    val2 = sigma_1(z-D_BETA) - 1
    return val1 * val2


def get_a_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    val_2 = sigma_norm(val_1)/sigma_norm(R)
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
        if distance < R_prime:
            n_i_k = (q_i_k - q_i) /(np.sqrt(1 + EPSILON * (np.linalg.norm(q_i_k-q_i))**2))
            
            b_i_k = bump_function(sigma_norm(q_i_k-q_i) / D_BETA, H_beta)
                
            sum_1 +=  phi_beta(sigma_norm(q_i_k-q_i)) * n_i_k 
            
            sum_2 +=  b_i_k * (p_i_k-p_i)
                
    ui_beta = C1_BETA * sum_1 + C2_BETA * sum_2
    
    return ui_beta




def get_ui_beta_hyper(i,q_i, p_i):
    
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    ui_beta = 0
    # for each obstacles
    for k in range(num_obstacles):
        
        yk = obstacles[k]
        a_k = (q_i - yk) / np.linalg.norm(q_i-yk)
        P = 1 - np.matmul(a_k.T, a_k)
        q_i_k = P*q_i + (1-P) * yk
        p_i_k = P * p_i
        
        distance = np.linalg.norm(q_i_k - q_i)
        if distance < R_prime:
            n_i_k = (q_i_k - q_i) /(np.sqrt(1 + EPSILON * (np.linalg.norm(q_i_k-q_i))**2))
            
            b_i_k = bump_function(sigma_norm(q_i_k-q_i) / D_BETA, H_beta)
                
            sum_1 += C1_BETA *  phi_beta(sigma_norm(q_i_k-q_i)) * n_i_k 
            
            sum_2 += C2_BETA * b_i_k * (p_i_k-p_i)
                
    ui_beta = sum_1 + sum_2
    
    return ui_beta
    

def get_u_i(i,q_i,p_i,target_pos):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):

        distance = np.linalg.norm(nodes[j] - nodes[i])
        if distance <= R:
            phi_alpha_val = phi_alpha(sigma_norm(nodes[j] - nodes[i]))
            sum_1 += phi_alpha_val * get_n_ij(i, j)
            sum_2 += get_a_ij(i, j) * (nodes_velocity[j] - nodes_velocity[i])
    
    ui_alpha = C1_ALPHA * sum_1 + C2_ALPHA * sum_2
                         
    ui_gamma = - C1_GAMMA * sigma_1_gamma(nodes[i] - target_pos)  - C2_GAMMA * (p_i - 0)
    
    ui_beta = get_ui_beta(i,q_i,p_i) # ui_beta 不参与j循环
           
    ui =  ui_alpha + ui_beta + ui_gamma

    return ui


def get_positions_static(G):
    counter = 0
    graph_data = []

    gm.clear_img_path(flocking_path,attributes_path,properties_path,speed_path)
    for t in range(0, ITERATION):
        # print(np.linalg.matrix_rank(adjacency_matrix))
        adjacency_matrix = create_adjacency_matrix()
        # print(np.linalg.matrix_rank(adjacency_matrix))
        # CONNECTIVITY[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)
        
        center_of_mass[t] = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1])])
        nodes_edge_list= get_edge_list() 
        G = update_graph(G,nodes_edge_list)
        if t == 0:
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
        else:
            for i in range(0, N):
                # p_i == old_velocity in  the paper
                # q_i === old_position
                old_velocity = nodes_velocity[i]
                old_position = np.array([POSITION_X[i, t-1],
                                         POSITION_Y[i, t-1]])
                
                # TODO add target_pos
                u_i = get_u_i(i, old_position, old_velocity,target_position)
                ACCELERACTION_X[i,t] = u_i[0]
                ACCELERACTION_Y[i,t] = u_i[1]
                
                #update position
                new_position = old_position + DELTA_T * old_velocity + (DELTA_T ** 2 / 2) * u_i
                [POSITION_X[i, t], POSITION_Y[i, t]] = new_position

                ACCELERACTION_MAGNITUDES[i,t] = np.linalg.norm(u_i)
                
                nodes[i, :] = new_position

                # update velocity
                new_velocity = (new_position - old_position) / DELTA_T
                nodes_velocity[i] = new_velocity
                VELOCITY_MAGNITUDES[i, t] = np.linalg.norm(new_velocity)
                
                #update orientation
                ORIENTATION[i,t] = update_orientation(i,t)
                
        graph_data = record_graph_properties(t)
        
        if (t) % SNAPSHOT_INTERVAL == 0:
            fig_flock = plt.figure('flocking',figsize=(12,10))
            plot_neighbors(t,fig_flock,nodes_edge_list)
            f_path = flocking_path + '/step {} _flock.png'.format(counter)
            plt.savefig(f_path)
            fig_flock.clf()
            
            fig_main = plt.figure('attributes',figsize=(12,12))            
            draw_network(G,graph_data,fig_main,t)
            a_path = attributes_path + '/step {} _attribute.png'.format(counter)
            plt.savefig(a_path)
            fig_main.clf()
            
            fig_speed = plt.figure('/Speed and position',figsize=(12,12))
            plot_position_speed(fig_speed)
            s_path = speed_path + '/step {} _speed.png'.format(counter)
            plt.savefig(s_path)
            fig_speed.clf()
            counter += 1
            # plt.show()

       
def save_parameter():
    path = img_path + '/parameter.txt'
    f = open(path,'w')
    a = 'C1_ALPHA: {}\n C2_ALPHA: {} \n\
        C1_BETA:  {}\n C2_BETA:  {} \n\
        C1_GAMMA: {}\n C2_GAMMA:{}\n\
        N : {}\n M:{}\nD:{} \n\
        K : {} \n R:{} \n A:{}\n\
        B:{} \n C:{} \n\
        D_prime:{}\n R_prime{} \n\
        EPSILON:{} \n H_alpha{} \n\
        DELTA:{}\n ITERATION:{}\n\
        '.format(C1_ALPHA,C2_ALPHA,C1_BETA,C2_BETA,C1_GAMMA,C2_GAMMA,N,M,D,K,R,A,B,C,D_prime,R_prime,EPSILON,H_alpha,DELTA_T,ITERATION)
    a.lstrip('\t')
    f.write(a)
    f.close()        
    
def build_square_obstacles(x1,y1,x2,y2,x_n,y_n):
    
    dl_y = np.linspace(y1,y2,y_n)
    dl_x = np.linspace(x1,x2,x_n)
    
    left = np.zeros((y_n,2))
    right = np.zeros((y_n,2))
    top = np.zeros((x_n,2))
    bottom = np.zeros((x_n,2))
    left[:,0] = x1
    left[:,1] = dl_y
    right[:,0] = x2
    right[:,1] = dl_y
    top[:,0] = dl_x
    top[:,1] =y2
    bottom[:,0] = dl_x
    bottom[:,1] = y1
    return np.concatenate((left,right,top,bottom))


def parameter_changer(i,itr_num,obst=True):
    global N 
    global M 
    global D 
    global K 
    global R 
    global A
    global B
    global C
    global D_prime 
    global R_prime
    global R_BETA 
    global D_BETA
    global EPSILON
    global H_alpha,H_beta  
    global C1_ALPHA 
    global C2_ALPHA
    global C1_BETA 
    global C2_BETA 
    global C1_GAMMA 
    global C2_GAMMA 
    global ITERATION
    global POSITION_X 
    global POSITION_Y 
    global AVERAGE_VELOCITY 
    global MAX_V 
    global AVERAGE_X_POSITION 
    global MAX_VELOCITY_X
    global VELOCITY_MAGNITUDES 
    global ACCELERACTION_X 
    global AVERAGE_X_ACC 
    global AVERAGE_Y_ACC
    global ACCELERACTION_Y 
    global ACCELERACTION_MAGNITUDES
    global AVERAGE_ACC_MAGNITUDES 
    global CONNECTIVITY
    global ORIENTATION
    global DEGREE_CENTRALITY 
    global DEGREES 
    global MAX_DEGREE
    global MAX_DEGREE_NODE  
    global MAX_DEGREE_CENTRALITY
    global MAX_DEGREE_CENTRALITY_NODE
    global CLUS_COEF 
    global AVERAGE_CLUSTERING 
    global MAX_CLUSTERING 
    global MAX_CLUSTERING_NODE
    global TRANGLES 
    global NUM_TRIANGLES  
    global MAX_TRANGLE_NUM 
    global MAX_TRANGLE_NODE 
    global DISTANCE
    global target_position,obstacles,Rk,num_obstacles
    global center_of_mass,formation_range
    global nodes,nodes_velocity,nodes_list
    global fig_counter
    global a_ij_matrix
    global img_path,flocking_path,attributes_path,properties_path,speed_path
    
    
    N = 100  # Number of sensor nodes
    M = 2  # Space dimensions
    D = 10  # Desired distance among nodes,i.e. algebraic constraints
    K = 1.2 # Scaling factor
    R = K*D  # Interaction range
    A = 5
    B = 5
    C = np.abs(A-B)/np.sqrt(4*A*B)
    ITERATION = itr_num
    POSITION_X = np.zeros([N, ITERATION])  # X position of each agent
    POSITION_Y = np.zeros([N, ITERATION]) # Y position of each agent
    AVERAGE_VELOCITY = np.zeros([1,ITERATION]) # the average speed for each iter
    MAX_V = np.zeros([1,ITERATION])# the max speed of points for each iter
    AVERAGE_X_POSITION = np.zeros([1,ITERATION]) # average X 
    MAX_VELOCITY_X = np.zeros([1,ITERATION]) # max speed X
    VELOCITY_MAGNITUDES = np.zeros([N, ITERATION]) # velocity of each agent 
    # acceleration 
    ACCELERACTION_X = np.zeros([N, ITERATION])
    AVERAGE_X_ACC = np.zeros([1, ITERATION])
    AVERAGE_Y_ACC = np.zeros([1, ITERATION])
    ACCELERACTION_Y = np.zeros([N, ITERATION])
    ACCELERACTION_MAGNITUDES = np.zeros([N, ITERATION])
    AVERAGE_ACC_MAGNITUDES = np.zeros([1, ITERATION])
    CONNECTIVITY = np.zeros([ITERATION, 1]) # connectivity for the network
    ORIENTATION = np.random.rand(N,ITERATION)*360 # direction of each agent while flying
    DEGREE_CENTRALITY = np.zeros([N, ITERATION])
    DEGREES = np.zeros([N, ITERATION])
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
    DISTANCE = np.array([N, ITERATION])
    RELATIVE_CONNECTIVITY = np.zeros([ITERATION, 1])
    
    D = 15
    D_prime =  D  # desired distance between obstacles and node
    R_prime = 1.414 * D_prime # Interaction range with obstacles
    R_BETA = sigma_norm(R_prime)
    D_BETA = sigma_norm(D_prime)
    
    
    # param lists
    EPSILONl = [0.1,0.1,0.1]
    H_alphal = [0.3,0.2,0.2]
    H_betal = [0.9,0.9,0.9]
    C1_ALPHAl = [40, 10, 10]
    C1_BETAl = [1500,1500,1500]
    C1_GAMMAl = [50,20,1]
    target_positionl = [np.array([500, 0]),np.array([650, 55]),np.array([650, 55])]
    
    x1 = 450
    x2 = 550
    y1 = -50
    y2 = 50
    x_n = int((x2-x1)/(0.6*D)) + 1
    y_n = int((y2-y1)/(0.6*D)) + 1
    smarll_rec = build_square_obstacles(x1,y1,x2,y2,x_n,y_n)
    x1 = 300
    x2 = 700
    y1 = -200
    y2 = 200
    x_n = int((x2-x1)/(0.6*D)) + 1
    y_n = int((y2-y1)/(0.6*D)) + 1
    big_rec = build_square_obstacles(x1,y1,x2,y2,x_n,y_n)
    rec = np.concatenate((smarll_rec,big_rec))
    obstaclesl = [big_rec,np.array([[650, 50]]),np.array([[400, 55]]),np.array([[400, 55]])]
    Rkl = [np.array([[30]]),np.array([[25]]),np.array([[50]])]
    
    # target_points = np.zeros([ITERATION, M])
    center_of_mass = np.zeros([ITERATION, M])
    formation_range = 390
    
    nodes = np.random.rand(N, M) * formation_range - formation_range/2 # nodes initial position,and instant position 
    
    target_position = target_positionl[i]
    initial_x = 500
    initial_y = 0
    
    for k in range(0,N):
        nodes[k][0] += initial_x
        nodes[k][1] += initial_y
    
    nodes_velocity = np.zeros([N, M])

    # assign values
    EPSILON = EPSILONl[i]
    H_alpha = H_alphal[i]
    H_beta = H_betal[i]
    C1_ALPHA = C1_ALPHAl[i] 
    C1_BETA = C1_BETAl[i]
    C1_GAMMA = C1_GAMMAl[i]
    C2_ALPHA = 2 * np.sqrt(C1_ALPHA)
    C2_BETA = 2*np.sqrt(C1_BETA)
    C2_GAMMA = 2 * np.sqrt(C1_GAMMA)
    
    if obst == False:
        obstacles=[]
        num_obstacles = 0
    else:
        obstacles = obstaclesl[i]
        num_obstacles = obstacles.shape[0]
        Rk = Rkl[i]
        

    G = nx.Graph()
    nodes_list = [i for i in range(len(nodes))] # must build the graph from nodes rather than edges
    G.add_nodes_from(nodes_list)
    # adding pos to each node
    for i in range(0,N):
        G.nodes[i]['pos'] = (nodes[i][0],nodes[i][1])
    # adjacency_matrix = np.zeros([N, N])
    a_ij_matrix = np.zeros([N, N])
    
    fig_counter = 0
    img_path,flocking_path,attributes_path,properties_path,speed_path = gm.get_file_path()




###==-------------------- main function -----------------------------==##

if __name__ == '__main__':
    

    for i in range(0,1):
        parameter_changer(i,5000,False)
        save_parameter()
        plot_deployment()
        get_positions_static(G)
        plot_properties()
        print('finished')

#%%


