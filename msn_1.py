from matplotlib import markers, scale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
# Parameters start
X = 100
Y = 100
EPSILON = 0.1
H = 0.2 #parameter for bump function 1<H<=1
C1_ALPHA = 30
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)
N = 50  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor node
K = 1.2  # Scaling factor
R = K*D  # Interaction range
DELTA_T = 0.009
A = 5
B = 5
C = np.abs(A-B)/np.sqrt(4*A*B)
ITERATION = 300
SNAPSHOT_INTERVAL = 50
POSITION_X = np.zeros([N, ITERATION])
POSITION_Y = np.zeros([N, ITERATION])
Orientation = np.random.rand(N,ITERATION)*360
markersize=10


nodes = np.random.rand(N, M) * X
nodes_old = nodes
nodes_velocity_p = np.zeros([N, M])
# adjacency_matrix = np.zeros([N, N])
a_ij_matrix = np.zeros([N, N])
velocity_magnitudes = np.zeros([N, ITERATION])
connectivity = np.zeros([ITERATION, 1])
fig_counter = 0
# Parameters end


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

#get the orientation for each point
def get_orientation(i,t):
    if t == 0:
        theata = Orientation[i,t]
    else:   
        delta_x = POSITION_X[i,t]-POSITION_X[i,t-1]
        delta_y = POSITION_Y[i,t]-POSITION_Y[i,t-1]
        theata = math.atan2(delta_y,delta_x)*360/(2*np.pi)
        
    return theata


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
    plt.figure('deployment')
    for i in range(N):
        marker, scale= gen_arrow_head_marker(Orientation[i,0]) #plot initial orientation of each point
        plt.plot(nodes[i, 0], nodes[i, 1], marker = marker,ms=markersize)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show(block=False)


def plot_neighbors(t):
    # draw points
    for i in range(0, N):
        theata_i = get_orientation(i,t)
        marker,scale= gen_arrow_head_marker(theata_i)
        plt.plot(nodes[i, 0], nodes[i, 1], marker = marker,ms =markersize)
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                # draw edges
                plt.plot([nodes[i, 0], nodes[j, 0]],
                         [nodes[i, 1], nodes[j, 1]],
                         'b-', lw=1)   
    plt.xlabel('X ')
    plt.ylabel('Y ')



def sigma_norm(z):
    val = np.sqrt(1 + EPSILON*(z**2)) - 1
    val = val/EPSILON
    return val


def gradient_sigma_norm(z):
    val = z/(1+EPSILON*sigma_norm(z))
    return val
    
    
def bump_function(z):
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
    val_1 = bump_function(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val

#spatial adjacency matrix A(q)
def get_a_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm)/sigma_norm(R)
    val = bump_function(val_2)
    return val


def get_n_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm**2
    val = val_1/np.sqrt(val_2)
    return val


def get_u_i(i):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        if i == j:
            pass
        else:
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                val_1 = nodes[j] - nodes[i]
                norm = np.linalg.norm(val_1)
                val = phi_alpha(sigma_norm(norm)) * get_n_ij(i, j)
                sum_1 += val

                val_2 = nodes_velocity_p[j] - nodes_velocity_p[i]
                sum_2 += get_a_ij(i, j) * val_2
    val = C1_ALPHA * sum_1 + C2_ALPHA * sum_2
    return val


    


def get_positions():
    for t in range(0, ITERATION):
        # print(np.linalg.matrix_rank(adjacency_matrix))
        adjacency_matrix = create_adjacency_matrix()
        # print(np.linalg.matrix_rank(adjacency_matrix))
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)
        # print(t)
        if t == 0:
            f = plt.figure('initial distribution')
            plot_neighbors(t)
            plt.show(block=False)
            
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
        else:
            for i in range(0, N):
                u_i = get_u_i(i)
                old_velocity = nodes_velocity_p[i, :]
                # old_position = np.array([POSITION_X[i, t-1],
                #                          POSITION_Y[i, t-1]])
                old_position = nodes[i]
                #更新速度
                new_velocity = old_velocity + u_i * DELTA_T
                #更新位置
                new_position = old_position + DELTA_T * new_velocity + (DELTA_T ** 2 / 2) * u_i
                [POSITION_X[i, t], POSITION_Y[i, t]] = new_position
                get_orientation(i,t)
                nodes_velocity_p[i, :] = new_velocity
                nodes[i, :] = new_position
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)
                # velocity_magnitudes[i, :, t] = new_velocity
        plt.figure('Snapshots',figsize=[15,10])
        if (t+1) % SNAPSHOT_INTERVAL == 0:
            Num_snapshots =ITERATION/SNAPSHOT_INTERVAL
            n_plt = (t+1)/SNAPSHOT_INTERVAL
            plt.subplot(2,3,int(n_plt))
            plt.title('T = {} '.format(n_plt))
            plot_neighbors(t)   
    plt.show(block=False)
    plt.figure('final')
    plt.subplot(221)
    plot_neighbors(t)
    plt.title('final position')

def plot_trajectory():
    plt.subplot(222)
    plt.title('trajectory')
    for i in range(0, N):
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])

    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)


def plot_velocity():
    plt.subplot(223)
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]
        # velocity_i = 1/velocity_i
        plt.plot(velocity_i)
        plt.show(block=False)
        plt.title('velocity')



def plot_connectivity():
    plt.subplot(224)
    plt.plot(connectivity)
    plt.show(block =False)
    plt.title('connectivity')
    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)
    # l=1
# plot_initial_deployment()
# create_adjacency_matrix()


plot_deployment()
get_positions()
plot_trajectory()
plot_velocity()
plot_connectivity()
plt.show()

#%%
