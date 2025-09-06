from matplotlib import pyplot as plt
from autograd import grad
import autograd.numpy as np               # drop-in replacement for numpy
import torch
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

FILETYPE = "pdf"
CMAP='coolwarm'
LINECOLOR='black'

# For conservative systems
def plot_phase_map(
        qs = np.linspace(-1.0, 1.0, 500),
        ps = np.linspace(-1.0, 1.0, 500),
        energy_func = None, 
        levels = None,
        countour_line_numbers = 60, 
        title="Phase Space Energy Level Sets of Two-Body Problem\n(Fixed $q_y=0$, $p_x=0$)",
        q_spec_label="(relative position)",
        p_spec_label="(momentum)",
        save_fig= False, 
        figure_path='visualizations/phase_map',
        cmap = CMAP,
        linecolor=LINECOLOR
        ):
    plt.figure(figsize=(8, 6))
    Q, P = np.meshgrid(qs, ps)
    H = energy_func(Q, P)
    if levels is None:
        levels = np.linspace(np.min(H), np.max(H), 30)
    _ = plt.contourf(Q, P, H, levels=levels, cmap=CMAP)
    plt.colorbar(label='Hamiltonian Energy $\\mathcal{H}$')
    plt.title(title)
    plt.xlabel('$q$ ' + q_spec_label)
    plt.ylabel('$p$ ' + p_spec_label)
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(figure_path + '.' + FILETYPE, dpi=300, format=FILETYPE)
        plt.close()
    else:
        plt.show()

# For dissipative systems
def plot_stream_lines(
        qs = np.linspace(-1.0, 1.0, 500),
        ps = np.linspace(-1.0, 1.0, 500),
        energy_func = None, 
        gamma = 0.0,
        levels = None,
        seed_points = None,
        title="Phase Space Energy Level Sets of Two-Body Problem\n(Fixed $q_y=0$, $p_x=0$)",
        q_spec_label="(relative position)",
        p_spec_label="(momentum)",
        save_fig= False, 
        figure_path='visualizations/phase_map',
        cmap=CMAP,
        linecolor=LINECOLOR
        ):
    plt.figure(figsize=(8, 6))
    Q, P = np.meshgrid(qs, ps)
    H = energy_func(Q, P)
    dHdq = grad(energy_func, argnum=0)
    dHdp = grad(energy_func, argnum=1)
    dHdq = np.vectorize(dHdq)(Q,P)
    dHdp = np.vectorize(dHdp)(Q,P)

    dQ =  dHdp
    dP = -dHdq - gamma * dHdp
    if levels is None:
        levels = np.linspace(np.min(H), np.max(H), 30)
    if seed_points is None:
        seed_points = np.array([[q, 0.0] for q in qs[::40]])
    _ = plt.contourf(Q, P, H, levels=levels, cmap=cmap)
    # plt.colorbar(label='Hamiltonian Energy $\\mathcal{H}$')
    s = plt.streamplot(Q, P, dQ, dP, color=linecolor, start_points=seed_points, linewidth=2, broken_streamlines=False)
    s.lines.set_label("Sampled path (noiseless)")
    plt.legend()
    plt.title(title)
    plt.xlabel('$q$ ' + q_spec_label)
    plt.ylabel('$p$ ' + p_spec_label)
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(figure_path + '.' + FILETYPE, dpi=300, format=FILETYPE)
        plt.close()
    else:
        plt.show()

# For Port Hamiltonian System, energy might be time-dependent, so need time integration lines.
def plot_port_Hamiltonian(
        qs = np.linspace(-1.0, 1.0, 500),
        ps = np.linspace(-1.0, 1.0, 500),
        energy_func = None, 
        levels = None,
        gamma = 0.0,
        T= 10.0,
        seed_points = None,
        title="Phase Space Energy Level Sets of Two-Body Problem\n(Fixed $q_y=0$, $p_x=0$)",
        q_spec_label="(relative position)",
        p_spec_label="(momentum)",
        quiver_scale=5.0,
        save_fig= False, 
        figure_path='visualizations/phase_map',
        cmap=CMAP,
        linecolor=LINECOLOR
    ):
    plt.figure(figsize=(8, 6))
    Q, P = np.meshgrid(qs, ps)
    H = energy_func(0, Q, P)
    # Assume H(t, q, p) to be time dependent and pack all (q,p) data
    def f(t, z):
        q, p = z
        dHdq = grad(energy_func, argnum=1)
        dHdp = grad(energy_func, argnum=2)
        dHdq = dHdq(t, q, p)
        dHdp = dHdp(t, q, p)
        return [dHdp, -dHdq - gamma * dHdp]
    if seed_points is None:
        seed_points = np.array([[q, 0.0] for q in qs[::40]])
    if levels is None:
        levels = np.linspace(np.min(H), np.max(H), 30)
    _ = plt.contourf(Q, P, H, levels=levels, cmap=cmap)
    # plt.colorbar(label='Hamiltonian Energy $\\mathcal{H}$')
    for points in seed_points:
        sol = solve_ivp(f, [0, T], points, t_eval=np.linspace(0, T, 101), max_step=T/100.0)
        i = len(sol.t) // 2
        plt.plot(sol.y[0], sol.y[1], linestyle='-', color=linecolor)
        dx, dy = f(sol.t[i], sol.y.T[i])
        plt.quiver(sol.y[0][i], sol.y[1][i], dx, dy, angles='xy', scale_units='inches', scale=quiver_scale,  color=linecolor, width=0.005)
    proxy = Line2D([0], [0], color=linecolor, lw=1.5, label="Sampled path (noiseless)")
    plt.legend(handles=[proxy],  loc='upper right')
    plt.title(title)
    plt.xlabel('$q$ ' + q_spec_label)
    plt.ylabel('$p$ ' + p_spec_label)
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(figure_path + '.' + FILETYPE, dpi=300, format=FILETYPE)
        plt.close()
    else:
        plt.show()

def plot_vector_field(
        qs = np.linspace(-1.0, 1.0, 500),
        ps = np.linspace(-1.0, 1.0, 500),
        energy_func = None, 
        gamma = 0.0,
        levels = None,
        seed_points = None,
        title="Phase Space Energy Level Sets of Two-Body Problem\n(Fixed $q_y=0$, $p_x=0$)",
        q_spec_label="(relative position)",
        p_spec_label="(momentum)",
        save_fig= False, 
        figure_path='visualizations/phase_map',
        cmap=CMAP,
        linecolor=LINECOLOR
        ):
    plt.figure(figsize=(8, 6))
    Q, P = np.meshgrid(qs, ps)
    H = energy_func(Q, P)
    dHdq = grad(energy_func, argnum=0)
    dHdp = grad(energy_func, argnum=1)
    dHdq = np.vectorize(dHdq)(Q,P)
    dHdp = np.vectorize(dHdp)(Q,P)

    dQ =  dHdp
    dP = -dHdq - gamma * dHdp
    if levels is None:
        levels = np.linspace(np.min(H), np.max(H), 30)
    if seed_points is None:
        seed_points = np.array([[q, 0.0] for q in qs[::40]])
    # plt.colorbar(label='Hamiltonian Energy $\\mathcal{H}$')
    s = plt.quiver(Q, P, dQ, dP, color=linecolor)
    plt.legend()
    plt.title(title)
    plt.xlabel('$q$ ' + q_spec_label)
    plt.ylabel('$p$ ' + p_spec_label)
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(figure_path + '.' + FILETYPE, dpi=300, format=FILETYPE)
        plt.close()
    else:
        plt.show()
