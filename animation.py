import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def redistribute_force(F, L, alpha, F_thr, neighborhood='moore'):
    F_new = np.copy(F)
    active_sites = []

    for i in range(L):
        for j in range(L):
            if F[i, j] >= F_thr:
                active_sites.append((i, j))

    for site in active_sites:
        i, j = site
        force_to_redistribute = F[i, j]

        if neighborhood == 'moore':
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1),
                         (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
        else:  # von Neumann neighborhood
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

        neighbors_count = 0
        for n in neighbors:
            ni, nj = n
            if 0 <= ni < L and 0 <= nj < L:
                neighbors_count += 1

        for n in neighbors:
            ni, nj = n
            if 0 <= ni < L and 0 <= nj < L:
                F_new[ni, nj] += (alpha * force_to_redistribute) / neighbors_count

        F_new[i, j] = 1

    return F_new, active_sites

def simulate_ca(L, alpha, F_thr, n_additions, neighborhood='moore'):
    F = np.random.rand(L, L)
    initial_state = np.copy(F)
    states = []

    active = True
    active_sites_count = 0
    active_sites_array = []
    active_sites_group = []
    while active:
        F, active_sites = redistribute_force(F, L, alpha, F_thr, neighborhood)
        states.append(np.copy(F))
        active_sites_group.append(active_sites)
        active = len(active_sites) > 0
        active_sites_count += len(active_sites)

        if not active and n_additions > 0:
            i, j = np.unravel_index(np.argmin(F_thr - F), F.shape)
            F += (F_thr - F[i, j])
            n_additions -= 1
            active_sites_array.append(active_sites_count)
            active_sites_count = 0
            active = True

    return initial_state, F, active_sites_array, active_sites_group, states

# Parameters
L = 25  # Size of the lattice
alpha = 0.5  # Coupling constant
F_thr = 4.0  # Threshold force
n_additions = 10000  # Number of additions of force to the system

# Run a simulation for Moore neighborhood
initial_state_moore, final_state_moore, active_sites_moore, active_sites_group, states = simulate_ca(L, alpha, F_thr, n_additions, neighborhood='moore')

fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(states[0], cmap='viridis', interpolation='nearest')
fig.colorbar(im, ax=ax)  # Add a colorbar to the initial image

def update(frame):
    ax.clear()
    im = ax.imshow(states[frame], cmap='viridis', interpolation='nearest')
    ax.set_title(f"Step {frame}")
    return im,

ani = FuncAnimation(fig, update, frames=len(states), interval=50, blit=True, repeat=False)
plt.show()
