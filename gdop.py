import numpy as np
import matplotlib.pyplot as plt

# # ----------------------------
# # GDOP calculation function (unit: meters)
# # ----------------------------
# def calculate_gdop(target_pos, stations):
#     """
#     Compute the GDOP of a TDOA system.
#     stations: (4, 3) array, where the first station is used as the reference station.
#     """
#     ref = stations[0]
#     others = stations[1:]
    
#     ranges = np.linalg.norm(stations - target_pos, axis=1)
#     r0 = ranges[0]
    
#     H = np.zeros((3, 3))
#     for i in range(3):
#         u_ref = (ref - target_pos) / r0
#         u_i = (others[i] - target_pos) / ranges[i + 1]
#         H[i, :] = u_ref - u_i
        
#     try:
#         cov = np.linalg.inv(H.T @ H)
#         gdop = np.sqrt(np.trace(cov))
#     except np.linalg.LinAlgError:
#         gdop = np.nan  # Set to NaN when singular, for easier plotting
#     return gdop

# Fully redundant method
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
    # Compute unit direction vectors (from the target to the base stations)
    vecs = stations - target_pos          # (N, 3)
    dists = np.linalg.norm(vecs, axis=1)  # (N,)
    
    # Safety check: avoid division by zero
    if np.any(dists < 1e-6):
        return np.inf
    
    u = vecs / dists[:, None]  # (N, 3)
    
    # Generate all unordered pairs (i, j), i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    H = np.zeros((len(pairs), 3))
    
    # Construct the H matrix: each row = u_i - u_j
    for idx, (i, j) in enumerate(pairs):
        H[idx, :] = u[i] - u[j]
    
    # Compute GDOP = sqrt(trace(inv(H^T H)))
    try:
        M = H.T @ H
        # Check rank (avoid ill-conditioning)
        if np.linalg.matrix_rank(M) < 3:
            return np.inf
        cov_geo = np.linalg.inv(M)
        gdop = np.sqrt(np.trace(cov_geo))
        return gdop if np.isfinite(gdop) else np.inf
    except np.linalg.LinAlgError:
        return np.inf

# ----------------------------
# Base station configurations (unit: meters)
# ----------------------------
# ant_pos = np.array([[ 50,    50,   0], [-86.60254038 + 50.0, 0.0, 0.0], [ 50,   150,   0],[86.60254038 + 50.0, 0.0, 0.0]], dtype=float) # Y-sharp
# ant_pos = np.array([[110, 50, 5],[-10, 50, 0], [40,50,5], [60, 50, 0]], dtype=float) # Straight line
# ant_pos = np.array([[50,-25,0], [-100, 50, 0], [50, 125, 0], [200, 50, 0]], dtype=float) # Lozenge
# ant_pos = np.array([[-5, -5, 0], [-5, 105, 5], [105, -5, 0], [105, 105, 5]], dtype=float) # Square
# ant_pos = np.array([[50,50,5],[-5, -5, 0], [-5, 105, 0], [105, -5, 0],  [105, 105, 0]], dtype=float) # 5-station square
ant_pos = np.array([[-100, -150, 5], [40,-150,0], [60, -150, 5], [200, -150, 0]], dtype=float) # Straight

# ----------------------------
# Simulation region settings (unit: meters)
# ----------------------------
x_min, x_max = -350, 450
y_min, y_max = -350, 450
n_points = 100

x = np.linspace(x_min, x_max, n_points)
y = np.linspace(y_min, y_max, n_points)
X, Y = np.meshgrid(x, y)
Z_target = 100.0  # Target height: 100 meters

# ----------------------------
# Compute the GDOP grid
# ----------------------------
print("Computing GDOP grid...")
GDOP = np.full_like(X, np.nan)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pt = np.array([X[i, j], Y[i, j], Z_target])
        GDOP[i, j] = calculate_gdop_all_pairs(pt, ant_pos)

# ----------------------------
# Plot: white background + colored contours (paper-style imitation)
# ----------------------------
plt.figure(figsize=(8, 8))

# Define contour levels (focus on low-GDOP regions, papers usually focus on < 2–3)
# 0–10: one line every 0.25; 10–30: every 1; 30–80: every 5
levels = np.r_[np.arange(0, 15, 0.5),
               np.arange(15, 41, 2),
               np.arange(41, 81, 5)]

# Use contourf for light filling + contour for emphasized lines (optional)
contour_filled = plt.contourf(X, Y, GDOP, levels=levels, cmap='jet', alpha=0.6)
contour_lines = plt.contour(X, Y, GDOP, levels=levels, colors='k', linewidths=0.3, alpha=0.5)

# Add contour labels (optional; often omitted in papers)
plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

# Plot base station locations (white circles with black edges for emphasis)
plt.plot(ant_pos[:, 0], ant_pos[:, 1], 'wo', markersize=9, markeredgecolor='black', markeredgewidth=1.5, label='Antennas')

# Set plot style: white background, axes, labels
plt.gca().set_facecolor('white')
plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)
# plt.title('GDOP Contour Map (Y-sharp-2 Configuration)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()

# # Add color bar
# cbar = plt.colorbar(contour_filled, pad=0.02)
# cbar.set_label('GDOP', fontsize=12)

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()