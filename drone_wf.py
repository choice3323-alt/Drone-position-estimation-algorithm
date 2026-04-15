import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal

def trajectory(flying_dia, start_pos):
    # Parameters
    timeinterval = 0.01
    x_limit = flying_dia
    y_limit = flying_dia
    z_limit = 100  # kept for parity; unused in current motion

    v_drone = 5
    circle_range = np.pi
    s = v_drone * timeinterval
    r = 5.0
    angle_interval = s / r

    # Initialize
    d_pos_r = [np.array(start_pos, dtype=float)] # 创建列表以存储轨迹点
    x_drone, y_drone, z_drone = map(float, start_pos) # 

    while True:
        if x_drone >= x_limit:
            break

        # 1) line (increasing y)
        while True:
            y_drone = y_drone + v_drone * timeinterval
            if x_drone >= x_limit or y_drone > y_limit - r:
                break
            d_pos_r.append(np.array([x_drone, y_drone, z_drone])) 

        if x_drone >= x_limit:
            break

        # 2) quarter circle turning upward (counter-clockwise)
        theta = 0.0
        x_st = x_drone
        y_st = y_drone
        while True:
            theta = theta + angle_interval
            if theta > circle_range:
                break
            x_drone = x_st + r - r * np.cos(theta)
            y_drone = y_st + r * np.sin(theta)
            if x_drone >= x_limit:
                break
            d_pos_r.append(np.array([x_drone, y_drone, z_drone]))

        if x_drone >= x_limit:
            break

        # 3) line (decreasing y)
        while True:
            y_drone = y_drone - v_drone * timeinterval
            if x_drone >= x_limit or y_drone < r:
                break
            d_pos_r.append(np.array([x_drone, y_drone, z_drone]))

        if x_drone >= x_limit:
            break

        # 4) quarter circle turning downward (clockwise)
        theta = 0.0
        x_st = x_drone
        y_st = y_drone
        while True:
            theta = theta + angle_interval
            if theta > circle_range:
                break
            x_drone = x_st + r - r * np.cos(theta)
            y_drone = y_st - r * np.sin(theta)
            if x_drone >= x_limit:
                break
            d_pos_r.append(np.array([x_drone, y_drone, z_drone]))

        if x_drone >= x_limit:
            break

    d_pos_r = np.vstack(d_pos_r)

    return d_pos_r


def compute_error(r, R, R_diff):
    # r: (3,), R: (n,3), R_diff: (n,n)
    d = np.linalg.norm(R - r, axis=1)
    D_mat = d[:, None] - d[None, :]  # d_i - d_j
    diff = D_mat - R_diff           # desired value should be zero
    return np.sum(diff**2)


def compute_gradient(r, R, R_diff):
    n = R.shape[0]
    g = np.zeros(3, dtype=float)
    d = np.linalg.norm(R - r, axis=1) + 1e-6  # avoid zero
    for j in range(n):
        deltas_j = R_diff[:, j]
        for i in range(n):
            grad_term = (r - R[i, :]) / d[i] - (r - R[j, :]) / d[j]
            g = g + 2.0 * (d[i] - d[j] - deltas_j[i]) * grad_term
    return g


def linesearch(compute_error_fcn, compute_gradient_fcn, r, direction, slope0, of):
    c1 = 1e-6
    c2 = 0.1

    alpha_max = 100.0
    alpha_0 = 0.0
    alpha_1 = 1.0

    of_r = of
    of_0 = of
    it = 0

    def noc_zoom(alpha_lo, alpha_hi, of_lo):
        while True:
            alpha = 0.5 * (alpha_lo + alpha_hi)
            rc = r + alpha * direction
            of_c = compute_error_fcn(rc)

            if of_c > of_0 + c1 * alpha * slope0 or of_c >= of_lo:
                alpha_hi = alpha
            else:
                slopec = float(np.dot(compute_gradient_fcn(rc), direction))
                if abs(slopec) <= -c2 * slope0:
                    return alpha
                if slopec * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha
                of_lo = of_c

    while True:
        rc = r + alpha_1 * direction
        of_c = compute_error_fcn(rc)
        slopec = float(np.dot(compute_gradient_fcn(rc), direction))

        # Armijo condition or sufficient decrease
        if (of_c > of_0 + c1 * alpha_1 * slope0) or ((of_c >= of_r) and (it > 0)):
            return noc_zoom(alpha_0, alpha_1, of_r)

        if abs(slopec) <= -c2 * slope0:
            return alpha_1

        if slopec >= 0:
            return noc_zoom(alpha_1, alpha_0, of_0)

        alpha_0 = alpha_1
        alpha_1 = min(alpha_max, alpha_1 * 3.0)
        of_r = of_c
        it += 1


def steepest_descent(r0, R, R_diff, max_time=2.0):
    r = r0.astype(float).copy()
    max_iter = 80000
    tol = 9e-6 # m^2
    start_time = time.time()
    prev_error = float('inf')
    no_improve_count = 0

    for it in range(1, max_iter + 1):
        if time.time() - start_time > max_time:
            print("Optimization timeout")
            break

        gradE = compute_gradient(r, R, R_diff)
        grad_norm = np.linalg.norm(gradE)
        if grad_norm < 1e-8:
            break

        direction = -gradE
        slope0 = float(np.dot(gradE, direction))
        of = compute_error(r, R, R_diff)

        if of < tol:
            break

        # check whether optimization is stuck
        if of >= prev_error:
            no_improve_count += 1
            if no_improve_count > 50:
                print("No improvement for 50 iterations, breaking")
                break
        else:
            no_improve_count = 0
        prev_error = of

        compute_error_fcn = lambda x: compute_error(x, R, R_diff)
        compute_gradient_fcn = lambda x: compute_gradient(x, R, R_diff)
        alpha = linesearch(compute_error_fcn, compute_gradient_fcn, r, direction, slope0, of)
        r = r + alpha * direction

    return r, it


def awgn_measured(x, snr_db):
    """
    Adds Additive White Gaussian Noise (AWGN) to a complex signal to achieve a specified measured SNR.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    y = np.empty_like(x, dtype=np.complex128)
    for i in range(x.shape[0]):
        sig = x[i]
        p_sig = np.mean(np.abs(sig)**2)
        snr_lin = 10**(snr_db / 10.0)
        p_noise = p_sig / snr_lin
        sigma = np.sqrt(p_noise / 2.0)
        noise = sigma * (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape))
        y[i] = sig + noise
    return y if y.shape[0] > 1 else y[0]


def unwrap_phase_to_reference(phi_meas, phi_ref, max_dist=10.0, delta_f=1e6, c=3e8):
    """
    Improved version: only allow adjustment within [-π, π] plus a small integer-cycle correction
    to prevent large phase jumps.

    max_dist: maximum allowed position change distance (meters)
    """
    # maximum allowed time difference
    max_td = max_dist / c
    max_phi_change = 2.0 * np.pi * delta_f * max_td

    # basic phase unwrapping
    k = np.round((phi_ref - phi_meas) / (2.0 * np.pi))
    phi_unwrapped = phi_meas + 2.0 * np.pi * k

    # check for excessive phase jumps
    delta_phi = phi_unwrapped - phi_ref
    if abs(delta_phi) > max_phi_change:
        # fallback to nearest 2π neighborhood
        k_safe = np.round((phi_ref - phi_meas) / (2.0 * np.pi))

        # restrict k change within ±1
        k_prev = (phi_ref - phi_meas) / (2.0 * np.pi)
        if abs(k_safe - k_prev) > 1.5:
            k_safe = np.floor(k_prev + 0.5)

        phi_unwrapped = phi_meas + 2.0 * np.pi * k_safe

    return phi_unwrapped


def main():
    """
    Simulates the trajectory and localization of a drone using multi-antenna phase-difference measurements.
    """
    # 1. Parameters
    
    # 4 antennas
    # ant_pos = np.array([[50, 50, 5], [-36.6, 0, 0], [50, 150, 0], [136.6, 0, 5]], dtype=float)  # Y-sharp
    # ant_pos = np.array([[-5, -5, 0], [-5, 105, 5], [105, -5, 0], [105, 105, 5]], dtype=float)  # Square
    # ant_pos = np.array([[50, -25, 5], [-100, 50, 0], [50, 125, 0], [200, 50, 5]], dtype=float)  # Lozenge
    ant_pos = np.array([[-100, 50, 5], [40, 50, 0], [60, 50, 5], [200, 50, 0]], dtype=float)  # Straight

    # 5 antennas
    # ant_pos = np.array([[-100, -50, 0], [200, -50, 5], [50, 200, 0], [50, 50, 0], [50, 20, 5]], dtype=float)
    # ant_pos = np.array([[50, 50, 5], [-5, -5, 0], [-5, 105, 0], [105, -5, 0], [105, 105, 0]], dtype=float)  # Square
    c = 2.99792458e8
    num_ant = len(ant_pos)
    f_base = 1e9
    delta_f = 1e6  # Hz spacing between tones
    f = f_base + delta_f * np.arange(2 * num_ant)

    f_off = 1e3
    fs = 1e11  # Sampling rate: 100 GHz
    total_time = 5e-6
    t = np.arange(0, total_time, 1.0 / fs)
    N = t.size
    snr = 60  # dB
    flying_dia = 100  # meters

    # Trajectory
    sam_interval = 1  # Sampling interval
    start_pos = np.array([0.0, 0.0, 100.0])
    traj_points = trajectory(flying_dia, start_pos)  # Continuous trajectory points
    step = max(1, int(round(sam_interval / 0.01)))
    d_pos_r = traj_points[::step]  # Sampled trajectory points only
    num_steps = len(d_pos_r) - 1
    d_pos = np.zeros((len(d_pos_r), 3), dtype=float)
    d_pos[0, :] = start_pos

    # Initial phase difference at the starting position
    # Use the time-delay difference computed from the true geometry as the initial reference
    p_time = np.zeros(num_ant)
    for i in range(num_ant):
        p_time[i] = np.linalg.norm(start_pos - ant_pos[i, :]) / c  # τ_i

    # Previous-point time difference: pp_td_ref[i,j] = τ_i - τ_j
    pp_td_ref = np.zeros((num_ant, num_ant))
    for j in range(num_ant):
        pp_td_ref[:, j] = p_time - p_time[j]

    # # GDOP
    # gdop = np.zeros((len(d_pos_r), 1), dtype=float)
    # gdop[0] = compute_tdoa_gdop(ant_pos, start_pos)

    # 2. Slow-time sampling over trajectory steps
    for k in range(num_steps):
        current_pos = d_pos_r[k + 1, :]

        # Distances to antennas
        R_dist = np.zeros(num_ant)
        for i in range(num_ant):
            R_dist[i] = np.linalg.norm(current_pos - ant_pos[i, :])

        # 3. Fast-time sampling: received signal from each antenna
        r_s = np.zeros((num_ant, N), dtype=np.complex128)
        for i in range(num_ant):
            idx1 = 2 * i
            idx2 = 2 * i + 1
            s1 = (1.0 / R_dist[i]) * np.exp(1j * 2 * np.pi * (f[idx1] + f_off) * (t - R_dist[i] / c))
            s2 = (1.0 / R_dist[i]) * np.exp(1j * 2 * np.pi * (f[idx2] + f_off) * (t - R_dist[i] / c))
            r_s[i, :] = s1 + s2

        r_s = awgn_measured(r_s, snr)

        # 4. Analog-signal-processing stage
        # 4.1 Use frequency-domain filtering to extract the two carrier components corresponding to each antenna
        beat_signals = np.zeros((num_ant, N), dtype=np.complex128)
        for i in range(num_ant):
            R_fft = np.fft.fftshift(np.fft.fft(r_s[i, :]))
            f_a = np.fft.fftshift(np.arange(N) - N / 2) * fs / N + fs / 2

            # Find the locations of the two largest peaks in the power spectrum
            mag_spectrum = np.abs(R_fft)
            peak_indices = np.argpartition(mag_spectrum, -4)[-4:]  # Take the top 4 peaks, then select the strongest 2
            top2 = peak_indices[np.argsort(mag_spectrum[peak_indices])[-2:]]  # Sort and select the strongest two
            f_est = np.sort(f_a[top2])  # Arrange in ascending frequency order
            f_est1, f_est2 = f_est

            # # Frequency-offset estimation: difference from the theoretical frequencies
            # f_expected1 = f[2 * i] + f_off
            # f_expected2 = f[2 * i + 1] + f_off
            # delta_f_est1 = f_est1 - f_expected1
            # delta_f_est2 = f_est2 - f_expected2
            # print(f"[Antenna {i}] Estimated freq offsets: {delta_f_est1:.2f} Hz, {delta_f_est2:.2f} Hz")

            BW = 3e5
            mask1 = np.abs(f_a - f_est1) < BW
            mask2 = np.abs(f_a - f_est2) < BW

            H1 = np.zeros(N)
            H1[mask1] = 1.0
            H2 = np.zeros(N)
            H2[mask2] = 1.0

            R1_fft = (R_fft * H1) / N
            R2_fft = (R_fft * H2) / N
            s1 = np.fft.ifft(np.fft.ifftshift(R1_fft))
            s2 = np.fft.ifft(np.fft.ifftshift(R2_fft))

            # Beat signal: phase ~ 2π Δf (τ_i - τ_j) when compared across antennas
            # Normalize the beat-signal amplitude to eliminate path-loss effects and ensure
            # numerical stability of phase estimation at all distances:
            # (R_dist²) * (1/R_dist * s1) * conj(1/R_dist * s2) = s1 * conj(s2), whose amplitude remains 1
            sig_after = (R_dist[i] ** 2) * s1 * np.conj(s2)

            # Use the phase-halved signal
            # phi = np.unwrap(np.angle(sig_after))
            # phi_half = phi / 2.0
            # sig_half = np.abs(sig_after) * np.exp(1j * phi_half)

            beat_signals[i, :] = sig_after

        # 4.2 Digital-signal-processing stage (downsampling)
        f_beat = delta_f  # 1 MHz (theoretical beat-frequency center)

        # (1) Downconvert to baseband
        t_beat = np.arange(N) / fs  # Note: fs is still 10 GHz here!
        lo = np.exp(-1j * 2 * np.pi * f_beat * t_beat)
        baseband = beat_signals * lo  # Shift to 0 Hz for all antennas

        # (2) Low-pass filtering (cutoff ≈ 300 kHz)
        nyq = fs / 2
        lpf = signal.firwin(101, 300e3 / nyq, window='hamming')
        baseband_filtered = signal.lfilter(lpf, 1.0, baseband, axis=1)

        # (3) Downsample to match the bandwidth
        target_fs = 20e6  # 20 MHz
        decim = int(round(fs / target_fs))  # 10e9 / 1.2e6 ≈ 8333
        decim = max(decim, 1)
        baseband_decim = signal.decimate(
            baseband_filtered, decim, ftype="fir", zero_phase=True, axis=1
        )
        # At this point, the sampling rate of baseband_decim is approximately 1.2 MHz,
        # and the bandwidth is approximately 600 kHz, so it can be directly used
        # for subsequent cross-correlation

        # 5. Estimate the phase difference between antenna pairs (mod 2π)
        phase_diff_est = np.zeros((num_ant, num_ant))
        for i in range(num_ant):
            for j in range(num_ant):
                cij = np.vdot(baseband_decim[j], baseband_decim[i])  # sum(conj(j) * i)
                phase_diff_est[i, j] = np.angle(cij)  # φ_ij ≈ 2π Δf (τ_i - τ_j) (mod 2π)
        # print(f"\nStep {k} phase_diff_est (deg):\n{np.rad2deg(phase_diff_est)}")

        # 6. Use the time-difference reference from the previous time step
        # to perform integer-cycle phase unwrapping
        phi_ref_mat = 2.0 * np.pi * delta_f * pp_td_ref  # Full reference phase matrix
        # print(f"Step {k} phi_ref_mat (deg):\n{np.rad2deg(phi_ref_mat)}")
        unwrapped_phase_diff = np.zeros_like(phase_diff_est)
        for i in range(num_ant):
            for j in range(num_ant):
                if i == j:
                    continue
                # Reference phase φ_ref = 2π Δf (τ_i - τ_j)^{ref}, while prohibiting large jumps
                unwrapped_phase_diff[i, j] = unwrap_phase_to_reference(
                    phase_diff_est[i, j], phi_ref_mat[i, j]
                )
                # unwrapped_phase_diff[i, j] = phi_ref_mat[i, j] + np.angle(np.exp(1j * (phase_diff_est[i, j] - phi_ref_mat[i, j])))
        # print(f"Step {k} unwrapped_phase_diff (deg):\n{np.rad2deg(unwrapped_phase_diff)}")

        # 7. Phase difference -> time difference -> range difference
        # φ_ij ≈ 2π Δf (τ_i - τ_j)
        pp_td = unwrapped_phase_diff / (2.0 * np.pi * delta_f)  # τ_i - τ_j
        # print(f"Step {k}, max |pp_td| = {np.max(np.abs(pp_td)):.6e} s")

        # # Optional: enforce antisymmetry to reduce noise
        # # (theoretically, τ_i - τ_j = -(τ_j - τ_i))
        # pp_td = 0.5 * (pp_td - pp_td.T)
        # np.fill_diagonal(pp_td, 0.0)

        # Use the current round's time differences as the reference for the next round
        pp_td_ref = pp_td.copy()

        # Time-difference matrix -> range-difference matrix
        R = ant_pos.copy()
        R_diff = c * pp_td.copy()  # d_i - d_j

        # 8. Position estimation: gradient-descent method
        # Initial iteration point: previous estimated position
        r = d_pos[k, :].copy()
        t_start = time.time()
        r_est, iters = steepest_descent(r, R, R_diff)
        d_pos[k + 1, :] = r_est
        elapsed_time = time.time() - t_start
        print(f"time: {elapsed_time:.4f} sec, iteration numbers: {iters}")

    # RMSE
    errors = np.sqrt(np.sum((d_pos - d_pos_r) ** 2, axis=1))
    max_error = np.max(errors)
    min_error = np.min(errors)

    print(f"Maximum: {max_error:.6f} m")
    print(f"Minimum: {min_error:.6f} m")
    valid_idx = np.all(d_pos != 0, axis=1)
    rmse = np.sqrt(np.mean(np.sum((d_pos[valid_idx, :] - d_pos_r[valid_idx, :]) ** 2, axis=1)))
    print(f"RMSE: {rmse:.6f} m")
    # np.savez("to verify sampling rate effect.npz", d_pos=d_pos, snr=snr, d_pos_r=d_pos_r, ant_pos=ant_pos, errors=errors, rmse=rmse)

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.axis('equal')

    # Antenna positions: black squares
    plt.plot(ant_pos[:, 0], ant_pos[:, 1], 'ks', linewidth=3, markersize=10, label='Antenna positions')
    for i, (x, y, _) in enumerate(ant_pos):
        plt.text(x + 3, y + 3, f"Antenna{i+1}", fontsize=12, fontweight="bold", color="black")

    # Connect all antenna pairs with black dashed lines
    first_link = True
    for i in range(len(ant_pos)):
        for j in range(i + 1, len(ant_pos)):
            x_pair = [ant_pos[i, 0], ant_pos[j, 0]]
            y_pair = [ant_pos[i, 1], ant_pos[j, 1]]
            if first_link:
                plt.plot(x_pair, y_pair, 'k--', linewidth=1, label='Antenna baselines')
                first_link = False
            else:
                plt.plot(x_pair, y_pair, 'k--', linewidth=1)

    # True trajectory
    plt.plot(traj_points[:, 0], traj_points[:, 1], color="k", linestyle="-", linewidth=1)

    # Estimated trajectory points: colored by error, with constrained colorbar range
    sc = plt.scatter(
        d_pos[:, 0], d_pos[:, 1],
        c=errors, s=60, marker="x",
        cmap="turbo",  # bright endpoints for clearer distinction between small and large errors
        vmin=errors.min(), vmax=errors.max(),
        label='Estimated positions'
    )

    cb = plt.colorbar(sc)
    cb.set_label('Error (m)', fontsize=16, fontweight='bold', family='DejaVu Sans')
    cb.ax.tick_params(labelsize=14)

    plt.xlabel('X (m)', fontsize=28)
    plt.ylabel('Y (m)', fontsize=28)
    # plt.title(f"Position Error Map in square antenna layout (SNR={snr}dB)", fontsize=28, fontweight="bold")
    plt.legend(loc='upper right', fontsize=14, labelcolor="linecolor")
    plt.tick_params(labelsize=14)
    plt.show()

if __name__ == "__main__":
    main()
