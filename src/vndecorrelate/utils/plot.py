from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import ConvexHull

from vndecorrelate.utils.dsp import (
    check_stereo,
    polar_coordinates,
    polar_to_cartesian,
)


def plot_correlogram(
    correlogram: NDArray,
    lag_seconds: float,
    time_seconds: int,
    title: str = 'Cross Correlogram',
) -> plt.Figure:
    fig = plt.figure(figsize=(15, 6))
    plt.imshow(
        np.abs(correlogram),
        extent=[-lag_seconds, lag_seconds, time_seconds, 0],
        aspect='auto',
        cmap='Blues',
        origin='upper',
    )
    plt.colorbar(label='Correlation Coefficient (Normalized)')
    plt.xlabel('Lag (s)')
    plt.ylabel('Time (s)')
    plt.title(title)
    return fig


def plot_signal(input_signal: NDArray, title: str = 'Signal') -> plt.Figure:
    """Plot the time domain input signal."""
    fig = plt.figure()
    plt.plot(input_signal)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(title)
    return fig


def plot_spectrogram(
    spectrogram: NDArray,
    title: str = 'Spectrogram',
) -> plt.Figure:
    """Plot the spectrogram of the input signal."""
    fig = plt.figure(figsize=(15, 6))
    plt.imshow(
        10 * np.log10(spectrogram + 1e-10),
        aspect='auto',
        cmap='Blues',
        origin='lower',
    )
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title(title)
    return fig


def _draw_scope_frame(
    axes,
    show_grid: bool = True,
    grid_color: str = '#1a2a2a',
    baseline_color: str = '#1a3333',
    label_color: str = '#4a8a8a',
    decibel_label_color: str = '#2a4a4a',
):
    """
    Draw the standard 180-degree vectorscope frame onto *axes*.

    Renders radial arcs with decibel labels, angular spokes, the horizontal
    baseline, channel labels (L / L45 / C / R45 / R), and the origin dot.
    Both polar_vectorscope and polar_heatmap call this so the two views
    share an identical frame.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Target axes. Must already have aspect=equal, xlim/ylim set,
        and axis('off') called.
    show_grid : bool
        Whether to draw the arcs and spokes.
    grid_color : str
        Color for arcs and spokes.
    baseline_color : str
        Color for the horizontal diameter line.
    label_color : str
        Color for the L/C/R channel labels and ticks.
    decibel_label_color : str
        Color for the dB level text at each arc.
    """
    if show_grid:
        arc_sweep_angles = np.linspace(-90.0, 90.0, 300)

        # Radial arcs at -12, -6, -3, 0 dBFS (r = 0.25, 0.50, 0.75, 1.0)
        for reference_radius in [0.25, 0.5, 0.75, 1.0]:
            arc_x, arc_y = polar_to_cartesian(arc_sweep_angles, reference_radius)
            axes.plot(arc_x, arc_y, color=grid_color, linewidth=0.7, zorder=1)
            decibel_value = 20.0 * np.log10(reference_radius + 1e-9)
            axes.text(
                0,
                reference_radius + 0.02,
                f'{decibel_value:.0f} dB',
                ha='center',
                va='bottom',
                color=decibel_label_color,
                fontsize=7,
                zorder=2,
            )

        # Angular spokes every 15 degrees
        for spoke_angle in range(-90, 91, 15):
            spoke_tip_x, spoke_tip_y = polar_to_cartesian(spoke_angle, 1.0)
            axes.plot(
                [0, spoke_tip_x],
                [0, spoke_tip_y],
                color=grid_color,
                linewidth=0.7,
                zorder=1,
            )

        # Horizontal baseline spanning the full folded angular range
        # (from -90 deg / fully anti-phase on one side to +90 deg on the other)
        left_x, left_y = polar_to_cartesian(-90.0, 1.0)
        right_x, right_y = polar_to_cartesian(90.0, 1.0)
        axes.plot(
            [left_x, right_x],
            [left_y, right_y],
            color=baseline_color,
            linewidth=1.0,
            zorder=1,
        )

    # Channel labels and tick marks at key angles
    label_kwargs = dict(
        color=label_color,
        fontsize=8,
        ha='center',
        va='center',
        zorder=10,
    )
    for label_angle, label_text in [
        (-90, 'Anti-\u03c6'),
        (-45, 'L'),
        (0, 'C'),
        (45, 'R'),
        (90, 'Anti-\u03c6'),
    ]:
        label_x, label_y = polar_to_cartesian(label_angle, 1.15)
        axes.text(label_x, label_y, label_text, **label_kwargs)

        tick_base_x, tick_base_y = polar_to_cartesian(label_angle, 1.0)
        tick_tip_x, tick_tip_y = polar_to_cartesian(label_angle, 1.05)
        axes.plot(
            [tick_base_x, tick_tip_x],
            [tick_base_y, tick_tip_y],
            color='#2a5a5a',
            linewidth=1.0,
            zorder=2,
        )

    # Origin dot
    axes.plot(0, 0, 'o', color=label_color, markersize=3, zorder=11)


def plot_polar_sample(
    input_signal: np.ndarray,
    sample_rate_hz: int = 44100,
    title: str = 'Polar Heatmap',
    figure_size: tuple = (8, 5.5),
    # Render mode
    mode: Literal[
        'heatmap', 'scatter', 'both'
    ] = 'heatmap',  # "heatmap" | "scatter" | "both"
    # Heatmap resolution and smoothing
    num_angle_bins: int = 240,  # angular resolution
    num_radius_bins: int = 150,  # radial resolution
    smoothing_sigma: float = 0.6,  # Gaussian blur on the count grid (bin units)
    heatmap_alpha: float = 0.72,  # overall opacity of the heatmap layer
    # Scatter options
    max_scatter_points: int = 20_000,  # subsample cap for the scatter layer
    scatter_point_size: float = 2.5,  # marker size in points^2
    scatter_alpha: float = 0.65,  # opacity of scatter markers
    # Color
    colormap_name: str = 'cyan_scope',  # "cyan_scope" or any matplotlib colormap name
    # Frame
    show_grid: bool = True,
    axes: plt.Axes = None,
) -> plt.Figure:
    """
    Plot a 180-degree polar density heatmap and/or scatter for a stereo signal.

    Each sample (L, R) is converted to the same polar coordinates used by
    polar_vectorscope:

        theta  =  arctan2(R, L) - 45 degrees    in [-90, +90]
        radius =  sqrt(L^2 + R^2) / sqrt(2)     in [0, 1]

    Heatmap mode
    ------------
    Samples are binned into a (angle x radius) grid.  Counts are log-scaled
    and lightly Gaussian-smoothed, then rendered as filled arc-wedge patches
    that respect the polar geometry.  The heatmap alpha is set below 1 so the
    grid arcs, spokes, and labels remain visible through the data layer.

    Scatter mode
    ------------
    Individual sample positions are drawn as points, colored by their local
    2-D density so dense regions glow and isolated points remain visible.
    The color range is mapped to the actual density range of the data, not to
    the full theoretical range, so even the outermost sparse points receive a
    visible non-zero color value.

    Parameters
    ----------
    input_signal : np.ndarray
        Stereo audio, shape (num_samples, 2) or (2, num_samples),
        with values in [-1, 1].
    sample_rate_hz : int
        Sample rate in Hz. Used for display annotations only.
    title : str
        Figure title.
    figure_size : tuple of (float, float)
        Width and height of the figure in inches.
    mode : {"heatmap", "scatter", "both"}
        Which layer(s) to render.  "both" overlays scatter on the heatmap.
    num_angle_bins : int
        Number of angular bins across the 180-degree range.
        Higher values give finer angular resolution.
    num_radius_bins : int
        Number of radial bins from 0 to 1.
    smoothing_sigma : float
        Standard deviation of the Gaussian blur applied to the bin counts,
        in units of bins. Smaller values preserve sharper detail.
    heatmap_alpha : float
        Opacity of the heatmap wedge layer (0-1). Lower values let the
        grid arcs and labels show through more clearly.
    max_scatter_points : int
        Maximum number of points drawn in scatter mode (random subsample).
    scatter_point_size : float
        Marker area in Matplotlib points^2 (the ``s`` parameter of scatter).
    scatter_alpha : float
        Opacity of scatter markers.
    colormap_name : str
        "cyan_scope" (built-in dark-to-cyan palette matching the vectorscope
        style) or any Matplotlib colormap name.
    show_grid : bool
        Draw angular spokes, radial arcs, and decibel labels.
    axes : matplotlib.axes.Axes, optional
        Existing axes to draw into. If None a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    check_stereo(input_signal)

    sample_radii, sample_angles_rad = polar_coordinates(
        input_signal[:, 0], input_signal[:, 1], compute_weights=False
    )

    sample_angles_deg = np.degrees(sample_angles_rad)

    # ------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------
    if colormap_name == 'cyan_scope':
        colormap = mcolors.LinearSegmentedColormap.from_list(
            'cyan_scope',
            [
                '#0a0a0f',
                '#001a22',
                '#003344',
                '#006688',
                '#00b4d8',
                '#00e5ff',
                '#80f4ff',
                '#ffffff',
            ],
        )
    else:
        colormap = plt.get_cmap(colormap_name)

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    background_color = '#0a0a0f'
    if axes is None:
        figure = plt.figure(figsize=figure_size, facecolor=background_color)
        axes = figure.add_subplot(111, facecolor=background_color)
    else:
        figure = axes.get_figure()

    axes.set_aspect('equal')
    axes.set_xlim(-1.25, 1.25)
    axes.set_ylim(-0.12, 1.25)
    axes.axis('off')

    _draw_scope_frame(axes, show_grid=show_grid)

    # ------------------------------------------------------------------
    # Heatmap layer
    # ------------------------------------------------------------------
    if mode in ('heatmap', 'both'):
        angle_bin_edges = np.linspace(-90.0, 90.0, num_angle_bins + 1)
        radius_bin_edges = np.linspace(0.0, 1.0, num_radius_bins + 1)

        bin_counts, _, _ = np.histogram2d(
            sample_angles_deg,
            sample_radii,
            bins=[angle_bin_edges, radius_bin_edges],
        )
        # shape: (num_angle_bins, num_radius_bins)

        # Smooth first, then log-scale to compress the dynamic range.
        if smoothing_sigma > 0:
            bin_counts = gaussian_filter(bin_counts, sigma=smoothing_sigma)
        log_bin_counts = np.log1p(bin_counts)
        count_maximum = log_bin_counts.max() or 1.0

        # Build one Wedge patch per occupied cell.
        #
        # Coordinate mapping:
        #   Vectorscope angle alpha (0 = up, -90 = left, +90 = right)
        #   maps to Matplotlib Wedge angle w = 90 - alpha
        #   (Wedge measures degrees CCW from the +x axis).
        #   So a scope bin [alpha_low, alpha_high] becomes
        #   the wedge arc [90 - alpha_high, 90 - alpha_low].
        wedge_patches = []
        wedge_colors = []

        for angle_index in range(num_angle_bins):
            for radius_index in range(num_radius_bins):
                cell_log_count = log_bin_counts[angle_index, radius_index]
                if cell_log_count < 1e-6:
                    continue

                inner_radius = radius_bin_edges[radius_index]
                outer_radius = radius_bin_edges[radius_index + 1]

                wedge_theta_1 = 90.0 - angle_bin_edges[angle_index + 1]
                wedge_theta_2 = 90.0 - angle_bin_edges[angle_index]

                wedge_patches.append(
                    Wedge(
                        center=(0.0, 0.0),
                        r=outer_radius,
                        theta1=wedge_theta_1,
                        theta2=wedge_theta_2,
                        width=outer_radius - inner_radius,
                    )
                )
                wedge_colors.append(cell_log_count / count_maximum)

        if wedge_patches:
            patch_collection = PatchCollection(
                wedge_patches,
                cmap=colormap,
                norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
                linewidth=0,
                antialiased=True,
                alpha=heatmap_alpha,
                zorder=3,
            )
            patch_collection.set_array(np.array(wedge_colors))
            axes.add_collection(patch_collection)

    # ------------------------------------------------------------------
    # Scatter layer
    # ------------------------------------------------------------------
    if mode in ('scatter', 'both'):
        num_samples = len(sample_angles_deg)

        if num_samples > max_scatter_points:
            selected_indices = np.random.choice(
                num_samples, max_scatter_points, replace=False
            )
            scatter_angles = sample_angles_deg[selected_indices]
            scatter_radii = sample_radii[selected_indices]
        else:
            scatter_angles = sample_angles_deg
            scatter_radii = sample_radii

        scatter_x, scatter_y = polar_to_cartesian(scatter_angles, scatter_radii)

        # Compute per-point density by looking up each point in a 2-D histogram
        # built from all samples (not just the subsample).
        density_angle_edges = np.linspace(-90.0, 90.0, 200)
        density_radius_edges = np.linspace(0.0, 1.0, 200)

        density_histogram, _, _ = np.histogram2d(
            sample_angles_deg,
            sample_radii,
            bins=[density_angle_edges, density_radius_edges],
        )
        density_histogram = gaussian_filter(np.log1p(density_histogram), sigma=1.5)

        angle_bin_indices = np.clip(
            np.searchsorted(density_angle_edges, scatter_angles) - 1,
            0,
            density_histogram.shape[0] - 1,
        )
        radius_bin_indices = np.clip(
            np.searchsorted(density_radius_edges, scatter_radii) - 1,
            0,
            density_histogram.shape[1] - 1,
        )
        point_densities = density_histogram[angle_bin_indices, radius_bin_indices]

        # Normalize density to the actual range of the data so that even the
        # sparsest points receive a visible color value, rather than collapsing
        # near zero on the colormap.
        density_minimum = point_densities.min()
        density_range = point_densities.max() - density_minimum
        if density_range < 1e-9:
            density_range = 1.0
        normalized_density = (point_densities - density_minimum) / density_range

        axes.scatter(
            scatter_x,
            scatter_y,
            c=normalized_density,
            cmap=colormap,
            s=scatter_point_size,
            alpha=scatter_alpha,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )

    # ------------------------------------------------------------------
    # Stats annotation
    # ------------------------------------------------------------------
    center_angle = np.average(sample_angles_deg, weights=sample_radii)
    rms_level = np.sqrt(np.mean(sample_radii**2))
    rms_decibels = 20.0 * np.log10(rms_level + 1e-12)
    width_degrees = np.std(sample_angles_deg)

    stats_text = (
        f'Center: {center_angle:+.1f} deg   '
        f'RMS: {rms_decibels:.1f} dBFS   '
        f'Width sigma: {width_degrees:.1f} deg'
    )
    axes.text(
        0,
        -0.09,
        stats_text,
        ha='center',
        va='top',
        color='#3a7a7a',
        fontsize=8,
        zorder=10,
        fontfamily='monospace',
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    num_samples = len(input_signal)
    duration_sec = num_samples / sample_rate_hz
    mode_label = {
        'heatmap': 'heatmap',
        'scatter': 'scatter',
        'both': 'heatmap + scatter',
    }[mode]
    axes.set_title(
        f'{title}\n'
        f'{num_samples:,} samples  {duration_sec:.2f} s  {sample_rate_hz} Hz'
        f'  |  {mode_label}',
        color='#66aaaa',
        fontsize=10,
        pad=8,
    )

    figure.tight_layout()
    return figure


def plot_lissajous(
    input_signal: NDArray,
    sample_rate_hz: int = 44100,
    title: str = 'Lissajous',
    figsize: tuple = (7, 7),
    color_mode: str = 'density',  # "density" | "time" | "solid"
    line_color: str = '#00ff88',  # used when color_mode="solid"
    alpha: float = 0.6,
    dot_size: float = 0.3,
    max_points: int = 100_000,
    show_axes: bool = True,
    axes: plt.Axes = None,
) -> plt.Figure:
    """
    Create a Lissajous (vectorscope) plot from a stereo audio numpy array.

    Parameters
    ----------
    input_signal : NDArray
        Stereo audio data. Shape: (num_samples, 2) or (2, num_samples).
        Values should be in the range [-1, 1].
    sample_rate_hz : int
        Sample rate in Hz (used for display purposes only).
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    color_mode : str
        "density"  – 2-D histogram heat map (shows where signal spends most time)
        "time"     – points coloured from start (blue) to end (red)
        "solid"    – single colour, scatter / line
    line_color : str
        Hex colour used when color_mode="solid".
    alpha : float
        Point/line opacity for "solid" and "time" modes.
    dot_size : float
        Point size for scatter-based modes.
    max_points : int
        Subsample to this many points to keep rendering fast.
    show_axes : bool
        Whether to draw the ±1 reference lines and axis labels.
    ax : plt.Axes, optional
        Existing axes to draw into. If None, a new figure is created.

    Returns
    -------
    fig : plt.Figure
    """
    check_stereo(input_signal)

    left = input_signal[:, 0]
    right = input_signal[:, 1]

    # Subsample for performance
    n = len(left)
    if n > max_points:
        step = n // max_points
        left = left[::step]
        right = right[::step]

    # Build figure / axes
    if axes is None:
        fig, axes = plt.subplots(figsize=figsize, facecolor='#0d0d0d')
    else:
        fig = axes.get_figure()

    axes.set_facecolor('#0d0d0d')
    axes.set_aspect('equal')
    axes.set_xlim(-1.1, 1.1)
    axes.set_ylim(-1.1, 1.1)

    # Reference guides
    if show_axes:
        guide_kw = dict(color='#444444', linewidth=0.8, linestyle='--')
        axes.axhline(0, **guide_kw)
        axes.axvline(0, **guide_kw)
        # 45° mono line  (L == R)
        axes.plot([-1, 1], [-1, 1], color='#333355', linewidth=0.8, linestyle=':')
        # −45° anti-phase line (L == −R)
        axes.plot([-1, 1], [1, -1], color='#333355', linewidth=0.8, linestyle=':')

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 300)
        axes.plot(
            np.cos(theta), np.sin(theta), color='#333333', linewidth=0.8, linestyle='-'
        )

    # Plot
    if color_mode == 'density':
        # 2-D histogram rendered as image
        counts, xedges, yedges = np.histogram2d(
            left, right, bins=512, range=[[-1.1, 1.1], [-1.1, 1.1]]
        )
        # Log scale + custom green-to-white colormap
        counts = np.log1p(counts.T)  # transpose: x→col, y→row
        axes.imshow(
            counts,
            extent=[-1.1, 1.1, -1.1, 1.1],
            origin='lower',
            aspect='equal',
            cmap='magma',
            interpolation='bilinear',
        )

    elif color_mode == 'time':
        # Colour points by time position
        t = np.linspace(0, 1, len(left))
        cmap = plt.cm.plasma
        axes.scatter(
            left,
            right,
            c=t,
            cmap=cmap,
            s=dot_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    else:  # solid
        axes.scatter(
            left,
            right,
            color=line_color,
            s=dot_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    # Labels
    label_kw = dict(color='#888888', fontsize=9)
    if show_axes:
        axes.text(1.05, 0.02, 'R+', ha='right', **label_kw)
        axes.text(-1.05, 0.02, 'R−', ha='left', **label_kw)
        axes.text(0.02, 1.07, 'L+', ha='center', **label_kw)
        axes.text(0.02, -1.07, 'L−', ha='center', **label_kw)

    axes.set_title(title, color='#cccccc', fontsize=12, pad=10)
    axes.tick_params(colors='#555555')
    for spine in axes.spines.values():
        spine.set_edgecolor('#333333')

    duration_s = n / sample_rate_hz
    axes.set_xlabel(
        f'Left channel   ({n:,} samples · {duration_s:.2f} s @ {sample_rate_hz} Hz)',
        color='#666666',
        fontsize=8,
    )
    axes.set_ylabel('Right channel', color='#666666', fontsize=8)

    fig.tight_layout()
    return fig


def _build_pseudo_hull(
    sample_angles_deg: np.ndarray,
    sample_radii: np.ndarray,
    num_bins: int = 360,
    percentile: float = 95.0,
    smoothing_sigma: float = 4.0,
    use_convex_hull: bool = True,
):
    """
    Build a polar envelope (pseudo convex hull) from sample polar coordinates.

    Steps:
      1. Bin the 180-degree angular range into num_bins slices.
      2. Take the given percentile of radii within each bin.
      3. Apply Gaussian smoothing around the ring.
      4. Optionally replace the smooth envelope with its geometric convex hull.

    Parameters
    ----------
    sample_angles_deg : np.ndarray
        Per-sample azimuth angles in degrees, range [-90, +90].
    sample_radii : np.ndarray
        Per-sample normalized radii, range [0, 1].
    num_bins : int
        Number of angular bins across the 180-degree range.
    percentile : float
        Radial percentile used as the envelope ceiling (0-100).
        95 gives a near-peak envelope; 50 gives the median shape.
    smoothing_sigma : float
        Standard deviation (in bin units) of the Gaussian smoothing kernel.
    use_convex_hull : bool
        If True, compute the geometric convex hull of the percentile envelope
        points and use that outline. If False, use the smoothed envelope directly.

    Returns
    -------
    hull_angles_deg : np.ndarray
        Angles of the hull vertices, closed (first == last).
    hull_radii : np.ndarray
        Radii of the hull vertices, closed (first == last).
    bin_center_angles : np.ndarray
        Center angle of every bin (useful for the inner RMS overlay).
    bin_percentile_radii : np.ndarray
        Percentile radius for every bin.
    """
    bin_edges = np.linspace(-90.0, 90.0, num_bins + 1)
    bin_center_angles = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_percentile_radii = np.zeros(num_bins)
    for bin_index in range(num_bins):
        mask = (sample_angles_deg >= bin_edges[bin_index]) & (
            sample_angles_deg < bin_edges[bin_index + 1]
        )
        if mask.any():
            bin_percentile_radii[bin_index] = np.percentile(
                sample_radii[mask], percentile
            )

    if smoothing_sigma > 0:
        bin_percentile_radii = gaussian_filter1d(
            bin_percentile_radii, sigma=smoothing_sigma, mode='wrap'
        )

    if use_convex_hull and np.count_nonzero(bin_percentile_radii) >= 3:
        # Convert binned envelope to Cartesian for convex hull computation.
        # Use the intermediate (pre-rotation) angle space where bins are
        # uniformly spaced from -90 to +90 degrees of standard math angle.
        envelope_x = bin_percentile_radii * np.cos(np.radians(bin_center_angles))
        envelope_y = bin_percentile_radii * np.sin(np.radians(bin_center_angles))

        # Include the origin so the hull is always grounded at the base.
        hull_input_points = np.vstack(
            [np.column_stack([envelope_x, envelope_y]), [0.0, 0.0]]
        )

        try:
            convex_hull = ConvexHull(hull_input_points)
            hull_vertices = hull_input_points[convex_hull.vertices]

            # Drop the origin vertex -- it was only needed for geometric
            # anchoring. Keeping it in the outline draws a line back through
            # the center of the plot.
            is_not_origin = ~(
                (np.abs(hull_vertices[:, 0]) < 1e-9)
                & (np.abs(hull_vertices[:, 1]) < 1e-9)
            )
            hull_vertices = hull_vertices[is_not_origin]

            # Sort by angle so the polygon outline draws correctly.
            vertex_angles = np.arctan2(hull_vertices[:, 1], hull_vertices[:, 0])
            sorted_order = np.argsort(vertex_angles)
            hull_vertices = hull_vertices[sorted_order]

            hull_angles_deg = np.degrees(
                np.arctan2(hull_vertices[:, 1], hull_vertices[:, 0])
            )
            hull_radii = np.hypot(hull_vertices[:, 0], hull_vertices[:, 1])

            # Close the loop
            hull_angles_deg = np.append(hull_angles_deg, hull_angles_deg[0])
            hull_radii = np.append(hull_radii, hull_radii[0])

        except Exception:
            # Fall back to the smooth envelope if the hull computation fails.
            hull_angles_deg = np.append(bin_center_angles, bin_center_angles[0])
            hull_radii = np.append(bin_percentile_radii, bin_percentile_radii[0])
    else:
        hull_angles_deg = np.append(bin_center_angles, bin_center_angles[0])
        hull_radii = np.append(bin_percentile_radii, bin_percentile_radii[0])

    return hull_angles_deg, hull_radii, bin_center_angles, bin_percentile_radii


def plot_polar_level(
    input_signal: NDArray,
    sample_rate_hz: int = 44100,
    title: str = 'Polar Vectorscope',
    figure_size: tuple = (8, 5.5),
    # Envelope options
    num_bins: int = 360,
    percentile: float = 95.0,
    smoothing_sigma: float = 5.0,
    use_convex_hull: bool = True,
    # Visual options
    hull_color: str = '#00e5ff',
    fill_color: str = '#003344',
    overlay_color: str = '#00e5ff',
    show_rms_overlay: bool = True,
    show_grid: bool = True,
    axes: plt.Axes = None,
) -> plt.Figure:
    """
    Plot a 180-degree polar level vectorscope for a stereo audio signal.

    The stereo image is summarized as a filled polar shape whose outline is
    the pseudo convex hull of the per-bin percentile envelope. An optional
    inner dashed overlay shows the median (50th percentile) shape.

    Parameters
    ----------
    input_signal : np.ndarray
        Stereo audio, shape (num_samples, 2) or (2, num_samples),
        with values in [-1, 1].
    sample_rate_hz : int
        Sample rate in Hz. Used for display annotations only.
    title : str
        Figure title.
    figure_size : tuple of (float, float)
        Width and height of the figure in inches.
    num_bins : int
        Number of angular bins used to compute the envelope.
    percentile : float
        Radial percentile that defines the outer envelope (0-100).
        95 gives a near-peak boundary; 50 gives the median shape.
    smoothing_sigma : float
        Standard deviation of the Gaussian smoothing applied to the
        binned radii before hull construction (in bin units).
    use_convex_hull : bool
        If True, replace the smooth envelope with its geometric convex hull
        for a cleaner, more angular outline.
    hull_color : str
        Color of the outer hull outline.
    fill_color : str
        Fill color inside the hull.
    overlay_color : str
        Color of the inner median overlay shape.
    show_rms_overlay : bool
        Draw a dashed inner shape at the 50th-percentile (median) envelope.
    show_grid : bool
        Draw angular spokes, radial arcs, and decibel labels.
    axes : matplotlib.axes.Axes, optional
        Existing axes to draw into. If None a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    check_stereo(input_signal)

    # ------------------------------------------------------------------
    # Polar conversion and hull construction
    # ------------------------------------------------------------------
    sample_radii, sample_angles_rad = polar_coordinates(
        input_signal[:, 0], input_signal[:, 1], compute_weights=False
    )
    sample_angles_deg = np.degrees(sample_angles_rad)

    hull_angles, hull_radii, _, _ = _build_pseudo_hull(
        sample_angles_deg,
        sample_radii,
        num_bins=num_bins,
        percentile=percentile,
        smoothing_sigma=smoothing_sigma,
        use_convex_hull=use_convex_hull,
    )

    if show_rms_overlay:
        _, _, median_bin_angles, median_bin_radii = _build_pseudo_hull(
            sample_angles_deg,
            sample_radii,
            num_bins=num_bins,
            percentile=50.0,
            smoothing_sigma=smoothing_sigma,
            use_convex_hull=False,
        )

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    background_color = '#0a0a0f'
    if axes is None:
        figure = plt.figure(figsize=figure_size, facecolor=background_color)
        axes = figure.add_subplot(111, facecolor=background_color)
    else:
        figure = axes.get_figure()

    axes.set_aspect('equal')
    axes.set_xlim(-1.25, 1.25)
    axes.set_ylim(-0.12, 1.25)
    axes.axis('off')

    _draw_scope_frame(axes, show_grid=show_grid)

    # ------------------------------------------------------------------
    # Hull fill and outline
    # ------------------------------------------------------------------
    hull_x, hull_y = polar_to_cartesian(hull_angles, hull_radii)

    # For the fill, anchor both ends at the origin to form a closed fan.
    hull_fill_x = np.concatenate([[0.0], hull_x, [0.0]])
    hull_fill_y = np.concatenate([[0.0], hull_y, [0.0]])
    axes.fill(hull_fill_x, hull_fill_y, color=fill_color, alpha=0.55, zorder=3)

    # Outline traces only the arc (not back through origin).
    axes.plot(hull_x, hull_y, color=hull_color, linewidth=1.8, alpha=0.9, zorder=4)

    # ------------------------------------------------------------------
    # Median overlay
    # ------------------------------------------------------------------
    if show_rms_overlay:
        median_x, median_y = polar_to_cartesian(median_bin_angles, median_bin_radii)
        median_fill_x = np.concatenate([[0.0], median_x, [0.0]])
        median_fill_y = np.concatenate([[0.0], median_y, [0.0]])
        axes.fill(
            median_fill_x, median_fill_y, color=overlay_color, alpha=0.12, zorder=5
        )
        axes.plot(
            median_x,
            median_y,
            color=overlay_color,
            linewidth=0.9,
            alpha=0.5,
            linestyle='--',
            zorder=6,
        )

    # ------------------------------------------------------------------
    # Stats annotation
    # ------------------------------------------------------------------
    center_angle = np.average(sample_angles_deg, weights=sample_radii)
    rms_level = np.sqrt(np.mean(sample_radii**2))
    rms_decibels = 20.0 * np.log10(rms_level + 1e-12)
    width_degrees = np.std(sample_angles_deg)

    stats_text = (
        f'Center: {center_angle:+.1f} deg   '
        f'RMS: {rms_decibels:.1f} dBFS   '
        f'Width sigma: {width_degrees:.1f} deg'
    )
    axes.text(
        0,
        -0.09,
        stats_text,
        ha='center',
        va='top',
        color='#3a7a7a',
        fontsize=8,
        zorder=10,
        fontfamily='monospace',
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    num_samples = len(input_signal)
    duration_sec = num_samples / sample_rate_hz
    axes.set_title(
        f'{title}\n'
        f'{num_samples:,} samples  {duration_sec:.2f} s  {sample_rate_hz} Hz'
        f'  |  envelope: p{percentile:.0f}',
        color='#66aaaa',
        fontsize=10,
        pad=8,
    )

    figure.tight_layout()
    return figure
