import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter



def add_noise_to_values(values, noise_percentage):
    """
    Add random noise to an array of values.
    The noise is both positive and negative, given as a percentage of the actual values.

    :param values: Numpy array of original values.
    :param noise_percentage: The percentage of the original value by which to vary.
    :return: Numpy array with noise added.
    """
    # Calculate noise as a percentage of each value
    noise = np.random.uniform(-noise_percentage, noise_percentage, values.shape) / 100.0
    noisy_values = values + (values * noise) # Add noise to the original values
    return noisy_values

def generate_normal_distribution(size, mean, std, lower_limit):
    """
    Generate an array of values with a given size, mean, and standard deviation,
    with the constraint that all values are above a specified lower limit.

    :param size: Size of the array to generate.
    :param mean: Mean value of the normal distribution.
    :param std: Standard deviation of the normal distribution.
    :param lower_limit: The lower limit for the values.
    :return: Numpy array of generated values.
    """
    values = np.random.normal(mean, std, size) # Generate values from a normal distribution

    # Ensure all values are above the lower limit
    values = np.maximum(values, lower_limit)

    return values

def redistribute_force(F, L, alpha, F_thr, F_res, neighborhood='moore'):
    """
    Redistribute force among the square lattice according to OFC model for moore and von nueman neighborhood criteria.
    Returns the new force array and the list of active sites.

    :param F: array of forces stored in the lattice.
    :param L: Size of the lattice (number of cells along one side of the square grid).
    :param alpha: fraction of force to redistribute.
    :param F_thr: array of thresholds for lattice cells.
    :param F_res: array of residual forces for lattice cells.
    :param neighborhood: Specifies the type of neighborhood for force redistribution
    :return: Numpy array of new forces array and number of active sites of the previous configuration.
    """
    F_new = np.copy(F) # Copying the force array to a new array
    # active_sites = [] # array of active sites(forces above threshold)

    # for i in range(L):
    #     for j in range(L):
    #         if F[i, j] >= F_thr[i, j]:  # Use individual cell thresholds
    #             active_sites.append((i, j))

    active_mask = F >= F_thr

    # Use np.argwhere to find the indices where the condition is True
    active_sites = np.argwhere(active_mask)

    # Convert the indices to a list of tuples
    active_sites = list(map(tuple, active_sites))

    for site in active_sites:
        i, j = site
        force_to_redistribute = F[i, j] - F_res[i, j] # Force to redistribute

        if neighborhood == 'moore':
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1),
                         (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
            neighbors_count = 8
        else:
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            neighbors_count = 4

        neighbors_count = sum(1 for ni, nj in neighbors if 0 <= ni < L and 0 <= nj < L) # Counting the number of neighbors
        
        for ni, nj in neighbors:
            if 0 <= ni < L and 0 <= nj < L:
                F_new[ni, nj] += (alpha * force_to_redistribute) / neighbors_count # Redistributing the force to the neighbors

        F_new[i, j] = F_res[i,j]  # Resetting the active site to a residual force

    return F_new, active_sites

def simulate_ca(L, alpha, mean_thr, std_thr, min_thr, mean_res, std_res, min_res, n_additions, neighborhood='moore'):
    """
    Simulate a Cellular Automaton model over a lattice.

    :param L: Size of the lattice (number of cells along one side of the square grid).
    :param alpha: Scaling factor used in the force redistribution process.
    :param mean_thr, std_thr, min_thr: Parameters for the distribution of threshold forces.
    :param mean_res, std_res, min_res: Parameters for the distribution of residual forces.
    :param n_additions: Number of iterations to perform in adding force to the system.
    :param neighborhood: Specifies the type of neighborhood for force redistribution.
    :return: Initial state, final state, and array of active site counts.
    """
    F = np.random.rand(L, L) * mean_thr
    initial_state = np.copy(F)
    total_additions = n_additions
    states = []
    i = 0
    # F_thr = add_noise_to_values(np.full((L, L), mean_thr), noise_thr)
    # F_res = add_noise_to_values(np.full((L, L), mean_res), noise_res)

    F_thr = generate_normal_distribution((L, L), mean_thr, std_thr, min_thr)
    F_res = generate_normal_distribution((L, L), mean_res, std_res, min_res)

    active = True
    active_sites_count = 0
    active_sites_array = []
    avalanche_sequences = []
    current_avalanche = []
    while active:
        F, active_sites = redistribute_force(F, L, alpha, F_thr, F_res, neighborhood)
        if i % 100 == 0:
            states.append(np.copy(F))
        i += 1
        active = len(active_sites) > 0
        active_sites_count += len(active_sites)
        current_avalanche.append(len(active_sites))

        if not active and n_additions > 0:
            i, j = np.unravel_index(np.argmin(F_thr - F), F.shape)
            F += (F_thr[i, j] - F[i, j])
            
            if n_additions != total_additions:
                active_sites_array.append(active_sites_count)
                avalanche_sequences.append(current_avalanche)
            n_additions -= 1
            active_sites_count = 0
            current_avalanche = []
            active = True

    # add the data for last addition
    active_sites_array.append(active_sites_count)
    avalanche_sequences.append(current_avalanche)

    return initial_state, F, active_sites_array, avalanche_sequences, states


# Parameters
L = 50  # Size of the lattice
alpha = 0.5  # Coupling constant
mean_thr = 5  # Mean threshold
std_thr = 0.0  # std dev of threshold
min_thr = 3.0  # Minimum threshold
mean_res = 0  # Mean residual force
std_res = 0 # std dev of residual force
min_res = 0  # Minimum residual force
n_additions = 20000  # Number of additions


# Run a simulation for Moore neighborhood
initial_state, final_state, active_sites_array, avalanche_sequences, states = simulate_ca(L, alpha, mean_thr, std_thr, min_thr, mean_res, std_res, min_res, n_additions, neighborhood='von')


fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(states[0], cmap='viridis', interpolation='nearest')
fig.colorbar(im, ax=ax)  # Add a colorbar to the initial image

def update(frame):
    ax.clear()
    im = ax.imshow(states[frame], cmap='viridis', interpolation='nearest')
    ax.set_title(f"Step {frame}")
    return im,

ani = FuncAnimation(fig, update, frames=range(len(states)), interval=1/125, blit=False, repeat=False)

# Save the animation process as .gif
writer = PillowWriter(fps=15, metadata=dict(artist='Group17'), bitrate=1800)
ani.save('earthquake_animation.gif', writer=writer)

plt.show()