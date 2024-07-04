
#%%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# Parameters start
X = 150
Y = 150
EPSILON = 0.1
H = 0.2
C1_ALPHA = 70
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)
N = 100  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor node
K = 1.2  # Scaling factor
R = K*D  # Interaction range
DELTA_T = 0.009
A = 5
B = 5
C = np.abs(A-B)/np.sqrt(4*A*B)
ITERATION_VALUES = np.arange(0, 7, DELTA_T)
ITERATION = ITERATION_VALUES.shape[0]
SNAPSHOT_INTERVAL = 5
POSITION_X = np.zeros([N, ITERATION])
POSITION_Y = np.zeros([N, ITERATION])

# initial orientation following average distribution 
ave_v = np.zeros([1,ITERATION]) # the average speed for each iter
max_v = np.zeros([1,ITERATION])# the max speed of points for each iter
ave_X = np.zeros([1,ITERATION]) # average X 
max_v_X = np.zeros([1,ITERATION]) # max speed X
Orientation = np.random.rand(N,ITERATION)*360
markersize=10


n_x = np.random.rand(N) * X
n_y = np.random.rand(N) * X
nodes = np.array([n_x, n_y]).T
# nodes = np.random.rand(N, M) * X + 150
nodes_velocity_p = np.zeros([N, M])
# adjacency_matrix = np.zeros([N, N])
a_ij_matrix = np.zeros([N, N])
velocity_magnitudes = np.zeros([N, ITERATION])
connectivity = np.zeros([ITERATION, 1])
# fig = plt.figure()
# fig_counter = 0
c1_mt = 20
c2_mt = 2 * np.sqrt(c1_mt)
q_mt_x1 = 50
q_mt_y1 = 295
q_mt_x1_old = q_mt_x1
q_mt_y1_old = q_mt_y1
target_points = np.zeros([ITERATION, M])
center_of_mass = np.zeros([ITERATION, M])
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

def update_orientation(i,t):
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
    fig_deployment = plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    return fig_deployment
    # plt.show()
    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)


def plot_neighbors(t,f):
    ax =f.add_subplot(211)
    # ax =f.add_subplot(211,xlim=[-450,450],ylim=[-450,450])
    ax.plot(target_points[0:t, 0], target_points[0:t, 1])
    # draw the mass point of the target
    ax.plot(q_mt_x1_old, q_mt_y1_old, 'ro', color='green')
    # draw the points by red 
    # plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    
    for i in range(0, N):
        # get the orientation of the ith point
        theata_i = Orientation[i,t]
        marker,scale = gen_arrow_head_marker(theata_i)
        ax.plot(nodes[i, 0], nodes[i, 1], marker = marker,ms =markersize)
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                #draw the 2D lines between neighbors
                ax.plot([nodes[i, 0], nodes[j, 0]],
                         [nodes[i, 1], nodes[j, 1]],
                         'b-', lw=0.5)
    



def sigma_norm(z):
    val = EPSILON*(z**2)
    val = np.sqrt(1 + val) - 1
    val = val/EPSILON
    return val


def bump_function(z):
    if 0 <= z <= H:
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


def get_u_i(i, q_mt, p_mt):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        distance = np.linalg.norm(nodes[j] - nodes[i])
        if distance <= R:
            val_1 = nodes[j] - nodes[i]
            norm = np.linalg.norm(val_1)
            phi_alpha_val = phi_alpha(sigma_norm(norm))
            val = phi_alpha_val * get_n_ij(i, j)
            sum_1 += val

            val_2 = nodes_velocity_p[j] - nodes_velocity_p[i]
            sum_2 += get_a_ij(i, j) * val_2
    val = C1_ALPHA * sum_1 + C2_ALPHA * sum_2 - c1_mt * (nodes[i] - q_mt) - c2_mt * (nodes_velocity_p[i] - p_mt) 
          
    return val


def get_positions():
    global q_mt_x1_old, q_mt_y1_old
    fig = plt.figure('tracking static point',figsize=(8, 12))
    plt.ion()
    for t in range(0, ITERATION):
        
        fig.suptitle('tracking static point')
        # print(np.linalg.matrix_rank(adjacency_matrix))
        adjacency_matrix = create_adjacency_matrix()
        # print(np.linalg.matrix_rank(adjacency_matrix))
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)
        center_of_mass[t] = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1])])
        # update each point's position
        if t == 0:
            target_points[t] = np.array([q_mt_x1_old, q_mt_y1_old])
            plot_neighbors(t,fig)
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
        else:
            q_mt_x1 = 50 + 50 * ITERATION_VALUES[t]
            q_mt_y1 = 295 - 50 * np.sin(ITERATION_VALUES[t])
            q_mt = np.array([q_mt_x1, q_mt_y1])
            target_points[t] = q_mt
            q_mt_old = np.array([q_mt_x1_old, q_mt_y1_old])
            # get the new speed
            p_mt = (q_mt - q_mt_old) / DELTA_T
            q_mt_x1_old = q_mt_x1
            q_mt_y1_old = q_mt_y1
            
            # update each point's position
            for i in range(0, N):
                u_i = get_u_i(i, q_mt, p_mt)
                old_velocity = nodes_velocity_p[i, :]
                old_position = np.array([POSITION_X[i, t-1],
                                         POSITION_Y[i, t-1]])
                new_velocity = old_velocity + u_i * DELTA_T
                new_position = old_position + DELTA_T * new_velocity + (DELTA_T ** 2 / 2) * u_i
                # new_velocity = (new_position - old_position) / DELTA_T
                # new_velocity = old_velocity + u_i * DELTA_T
                [POSITION_X[i, t], POSITION_Y[i, t]] = new_position
                nodes_velocity_p[i, :] = new_velocity
                nodes[i, :] = new_position
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)
                #update orientation
                Orientation[i,t]=update_orientation(i,t)
                # velocity_magnitudes[i, :, t] = new_velocity
        # if t % SNAPSHOT_INTERVAL == 0:
        plot_neighbors(t,fig)
        plot_dynamic_speed(t,fig)
        plt.pause(0.001)
        fig.clf()
    plt.ioff()
    plt.show()

def plot_dynamic_speed(t,fig):
    vel_plot = fig.add_subplot(212,title='maximum speed point in the flock',xlabel='the floak average position',ylabel='speed',xlim=[-450,450])
    max_v[:,t] =max(velocity_magnitudes[:, t])
    ave_v[:,t] = np.average(velocity_magnitudes[:, t])
    max_v_index = velocity_magnitudes[:, t].argmax()
    ave_X[:,t] = np.average(POSITION_X[:,t])
    max_v_X[:,t] = POSITION_X[max_v_index,t]
    vel_plot.plot(ave_X[0,:],max_v[0,:],'bo',label = 'max speed')
    vel_plot.plot(ave_X[0,:],ave_v[0,:],'rx',label = 'average speed')
    vel_plot.legend(loc='upper right')


def plot_trajectory():
    for i in range(0, N):
        # arr = np.array([POSITION_X[i, :], POSITION_Y[i, :]])
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])
        # plt.show()
    plt.show()
    s = 1
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
    s = 1
    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)


def plot_connectivity():
    m = connectivity
    plt.plot(connectivity)
    plt.show()
    s = 1
    # global fig_counter
    # fig_name = 'figure_' + str(fig_counter) + '.png'
    # fig_counter += 1
    # fig.savefig(fig_name, dpi=fig.dpi)
    # l=1
# plot_initial_deployment()
# create_adjacency_matrix()


def plot_center_of_mass():
    plt.plot(target_points[:, 0], target_points[:, 1])
    plt.plot(center_of_mass[:, 0], center_of_mass[:, 1])
    plt.show()

if __name__ == '__main__':

    # plot_deployment()
    get_positions()
    # plot_trajectory()
    # plot_velocity()
    # plot_connectivity()
    # plot_center_of_mass()

#%%