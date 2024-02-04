import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

def visualize_states(initial_state, final_state, title='Cellular Automata States'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Determine the color scale limits based on the maximum value in both states
    vmax = max(np.max(initial_state), np.max(final_state))

    # Initial state visualization
    ax1 = axes[0]
    im1 = ax1.imshow(initial_state, cmap='viridis', interpolation='nearest', vmax=vmax)
    ax1.set_title('Initial State')
    fig.colorbar(im1, ax=ax1, orientation='vertical')

    # Final state visualization
    ax2 = axes[1]
    im2 = ax2.imshow(final_state, cmap='viridis', interpolation='nearest', vmax=vmax)
    ax2.set_title('Final State')
    fig.colorbar(im2, ax=ax2, orientation='vertical')

    # Overall title and show
    plt.suptitle(title)
    plt.show()

def visualize_states_3d(initial_state, final_state, title='Cellular Automata States'):
    fig = plt.figure(figsize=(12, 6))

    # Create 3D subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Create meshgrid for 3D plot
    x, y = np.meshgrid(np.arange(initial_state.shape[0]), np.arange(initial_state.shape[1]))

    # Plot 3D surface for initial state
    ax1.plot_surface(x, y, initial_state, cmap='viridis', vmax=np.max(initial_state), rstride=1, cstride=1, alpha=0.8)
    ax1.set_title('Initial State')

    # Plot 3D surface for final state
    ax2.plot_surface(x, y, final_state, cmap='viridis', vmax=np.max(final_state), rstride=1, cstride=1, alpha=0.8)
    ax2.set_title('Final State')

    # Overall title and show
    plt.suptitle(title)
    plt.show()
    
def plot_arrays_on_spheres(array_2d_1, array_2d_2, title1='2D Array 1 on Sphere', title2='2D Array 2 on Sphere'):
    # Number of grid points
    ntheta, nphi = array_2d_1.shape

    # Define spherical coordinates
    theta = np.linspace(0, np.pi, ntheta)
    phi = np.linspace(0, 2 * np.pi, nphi)

    # Create meshgrid for spherical coordinates
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)

    # Create a 3D plot with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

    # Plot the first 2D array on the first sphere
    axes[0].plot_surface(X, Y, Z, facecolors=plt.cm.viridis(array_2d_1), rstride=1, cstride=1, alpha=0.8)
    axes[0].set_title(title1)

    # Plot the second 2D array on the second sphere
    axes[1].plot_surface(X, Y, Z, facecolors=plt.cm.viridis(array_2d_2), rstride=1, cstride=1, alpha=0.8)
    axes[1].set_title(title2)

    # Set common labels and adjust layout
    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.suptitle('2D Arrays on Spheres')
    plt.show()