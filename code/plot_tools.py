import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_with_std(mean_population, std_population, t_array, color, label):
    """ Used for NO diffusion simulations.
    Plot mean population with shaded area representing standard deviation.
    
    Parameters:
        mean_population (array): Mean population array.
        std_population (array): Standard deviation population array.
        t_array (array): Time array.
    """
    plt.plot(t_array, mean_population, label=label,color=color)
    plt.fill_between(t_array, mean_population - std_population, mean_population + std_population, alpha=0.2,color=color)
    plt.xlabel('Time')
    plt.title('Evolution with Standard Deviation')
    plt.legend()

def plot_pop_traj(array,t_array):
  '''Used for NO diffusion simulations
  Plots individual trajectories'''
  plt.plot(t_array, array)
  plt.xlabel('time [sec]')
  plt.ylabel('Number of molecules in system')
  plt.show()

def plot_1d(array, h):
  '''Used for 1D simulations. Plots histogram: x-axis the indices of box, y-axis the population of each box. '''
  x_array = [h*element*1e3 for element in range(len(array))]
  plt.bar(x_array, array, width=h*1e3, edgecolor='black')
  plt.xlabel('x [mm]')
  plt.ylabel('Number of molecules in compartment')
  plt.show()

def plot_2d(array):
  plt.imshow(array, cmap='viridis')  # Change 'viridis' to any other colormap you prefer
  plt.colorbar()
  plt.show()


def plot_movie(array, t_array, increment):
  '''Used for 2D simuations'''
  fig = plt.figure()

  mod_array, mod_t = select_elements_with_dt(array, t_array, increment)

  # Function to update the plot for each frame
  def update_plot(frame):
      plt.clf()  # Clear the current plot
      plt.imshow(mod_array[frame], cmap='viridis', vmin=vmin, vmax=vmax)  # Change 'viridis' to any other colormap you prefer
      plt.title(f'Timestep: {mod_t[frame]:.4g}')
      plt.colorbar()  # Show colorbar for each frame

  # Set up fixed colorbar
  vmin = min(np.min(arr) for arr in mod_array)
  vmax = max(np.max(arr) for arr in mod_array)
  norm = plt.Normalize(vmin=vmin, vmax=vmax)

  # Create animation
  ani = animation.FuncAnimation(fig, update_plot, frames=len(mod_array), repeat=False)

  # Save the animation to a file
  ani.save('../one_movie.mp4', writer='ffmpeg', fps=2)

def select_elements_with_dt(A, B, dt):
    '''Used for 2D simulations. Function used to make the movie. 
    It adapts the increment of the arrays to a given one dt by discarding the intermediate ones.'''
    # Find the indices of elements in B with time step dt
    selected_indices = np.arange(0, len(B), int(dt / np.diff(B)[0]))
    # Select corresponding elements from A and B
    selected_A = A[selected_indices]
    selected_B = B[selected_indices]
    return selected_A, selected_B

def plot_2_movies(array1, t_array1, array2, t_array2, increment):
    '''Used for 2D simulations. Plots the movies for both population and entropy production side by side'''
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    mod_array1, mod_t1 = select_elements_with_dt(array1, t_array1, increment)
    mod_array2, mod_t2 = select_elements_with_dt(array2, t_array2, increment)

    # Pre-calculate minimum and maximum values
    vmin1 = min(np.min(arr) for arr in mod_array1)
    vmax1 = max(np.max(arr) for arr in mod_array1)
    vmin2 = min(np.min(arr) for arr in mod_array2)
    vmax2 = max(np.max(arr) for arr in mod_array2)

    # Set up fixed colorbar
    norm1 = plt.Normalize(vmin=vmin1, vmax=vmax1)
    norm2 = plt.Normalize(vmin=vmin2, vmax=vmax2)

    # Boolean flags to track if colorbars have been added
    colorbars_added = [False, False]

    # Function to update the plot for each frame
    def update_plot(frame):
        axes[0].clear()  # Clear the subplot for movie 1
        im1 = axes[0].imshow(mod_array1[frame], cmap='viridis', norm=norm1)
        axes[0].set_title(f'Population, Timestep: {mod_t1[frame]:.4g}')
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].clear()  # Clear the subplot for movie 2
        im2 = axes[1].imshow(mod_array2[frame], cmap='viridis', norm=norm2)
        axes[1].set_title(f'Entropy production, Timestep: {mod_t2[frame]:.4g}')
        axes[1].grid(False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Add colorbars
        if not colorbars_added[0]:  # Only add colorbar for movie 1 once
            cbar1 = fig.colorbar(im1, ax=axes[0])
            colorbars_added[0] = True
        if not colorbars_added[1]:  # Only add colorbar for movie 2 once
            cbar2 = fig.colorbar(im2, ax=axes[1])
            colorbars_added[1] = True

    # Create animation
    ani = animation.FuncAnimation(fig, update_plot, frames=len(mod_array1), repeat=False)

    # Save the animation to a file
    ani.save('../two_movies.mp4', writer='ffmpeg', fps=2)
    
def plot_num_traj(array,t_array):
  '''Used for >1D simulations. Sums all elemenst of an array and plots individual trajectory'''
  num_array = [np.sum(element) for element in array]
  plt.plot(t_array, num_array)
  plt.xlabel('time [sec]')
  plt.ylabel('Number of molecules in system')
  plt.show()
