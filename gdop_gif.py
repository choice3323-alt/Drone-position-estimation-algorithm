import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# GDOP calculation function (unit: meters)
def calculate_gdop_all_pairs(target_pos, stations):
    """
    Compute GDOP for TDOA using all base-station pairs (no reference station)

    Parameters:
        target_pos: np.array([x, y, z]) target position (m)
        stations: np.array (N, 3) base station coordinates (m)

    Returns:
        float: GDOP value
    """
    N = stations.shape[0]
    vecs = stations - target_pos
    dists = np.linalg.norm(vecs, axis=1)
    
    if np.any(dists < 1e-6):
        return np.inf
    
    u = vecs / dists[:, None]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    H = np.zeros((len(pairs), 3))
    
    for idx, (i, j) in enumerate(pairs):
        H[idx, :] = u[i] - u[j]
    
    try:
        M = H.T @ H
        if np.linalg.matrix_rank(M) < 3:
            return np.inf
        cov_geo = np.linalg.inv(M)
        gdop = np.sqrt(np.trace(cov_geo))
        return gdop if np.isfinite(gdop) else np.inf
    except np.linalg.LinAlgError:
        return np.inf


# Create simulation grid points
x_min, x_max = -750, 850
y_min, y_max = -750, 850
n_points = 100

x = np.linspace(x_min, x_max, n_points)
y = np.linspace(y_min, y_max, n_points)
X, Y = np.meshgrid(x, y)

Z_target = 100.0  # Target height


# Animation frame generation function
def animate(frame):

    center_x = 50
    base_offset = 75  # Original distance |x − 50|
    
    stretch_factor = 1.0 + frame * (3.0 - 1.0) / 19

    ant_pos_stretched = np.array([
        [50, -25, 0],                       
        [center_x - base_offset * stretch_factor, 50, 0],
        [50, 125, 0],
        [center_x + base_offset * stretch_factor, 50, 0],
    ], dtype=float)
    
    GDOP = np.full_like(X, np.nan)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = np.array([X[i, j], Y[i, j], Z_target])
            GDOP[i, j] = calculate_gdop_all_pairs(pt, ant_pos_stretched)
    
    ax.clear()

    levels = np.r_[np.arange(0, 15, 0.5),
                   np.arange(15, 41, 2),
                   np.arange(41, 81, 5)]

    contour_filled = ax.contourf(X, Y, GDOP, levels=levels, cmap='jet', alpha=0.6)
    contour_lines = ax.contour(X, Y, GDOP, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
    
    ax.clabel(contour_lines, fmt="%.1f", inline=True, fontsize=8)

    ax.plot(
        ant_pos_stretched[:, 0],
        ant_pos_stretched[:, 1],
        'wo',
        markersize=9,
        markeredgecolor='black',
        markeredgewidth=1.5
    )

    ax.set_title(f'GDOP Contour Map - Stretch Factor: {stretch_factor:.2f}', fontsize=14)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


# Create figure object
fig, ax = plt.subplots(figsize=(8, 8))


# Create animation object
ani = FuncAnimation(fig, animate, frames=range(20), interval=500, repeat=False)


# Save animation as GIF file
ani.save('gdop_evolution.gif', writer='pillow', dpi=100)

plt.close()

print("Animation saved as 'gdop_evolution.gif'")