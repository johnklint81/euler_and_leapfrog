import numpy as np
import matplotlib.pyplot as plt

max_time = 4
k = 5
m = 0.1
r0 = 0.1
v0 = 0


def analytical_HO(_r0, _v0, _t, _k, _m):
    _omega = np.sqrt(_k / _m)
    _A = np.sqrt(_r0 ** 2 + (_v0 / _omega) ** 2)
    _phi = np.arccos(_r0 / _A)
    _r = _A * np.cos(_omega * _t + _phi)
    _v = - _omega * _A * np.sin(_omega * _t + _phi)
    return _r, _v, _omega, _A


def euler_HO(_r0, _v0, _t_step, _n_steps, _k, _m):
    _r = np.zeros(_n_steps)
    _r[0] = _r0
    _v = np.zeros_like(_r)
    _v[0] = _v0
    for i in range(_n_steps - 1):
        _r[i + 1] = _r[i] + _v[i] * _t_step
        _v[i + 1] = _v[i] - k * _r[i] / m * _t_step
    return _r, _v


def get_energy(_r, _v, _m, _k):
    _U = 0.5 * _k * _r ** 2
    _K = 0.5 * _m * _v ** 2
    return _U, _K


timestep_list = [0.0001, 0.002, 0.02]

fig1, axis = plt.subplots(2, 3)

for i, timestep in enumerate(timestep_list):
    number_of_steps = int(max_time / timestep)
    t = np.linspace(0, max_time, number_of_steps)
    r_analytic, v_analytic, omega, A = analytical_HO(r0, v0, t, k, m)
    T = 2 * np.pi / omega
    t_plot = t / T
    U_analytic, K_analytic = get_energy(r_analytic, v_analytic, m, k)
    r_euler, v_euler = euler_HO(r0, v0, timestep, number_of_steps, k, m)
    U_euler, K_euler = get_energy(r_euler, v_euler, m, k)
    E_euler = (U_euler + K_euler) / (U_euler[0] + K_euler[0])
    E_analytic = (U_analytic + K_analytic) / (U_analytic[0] + K_analytic[0])
    axis[0, i].plot(t_plot, r_euler / A, 'r', alpha=0.7, label='Euler position')
    # axis[0, i].plot(t_plot, v_euler, 'b', alpha=0.7, label='Euler velocity')
    axis[0, i].plot(t_plot, r_analytic / A, 'k--', label='Analytical position')
    axis[0, i].set_title(f'$\Delta t$ = {1000*timestep} ms')
    axis[1, i].plot(t_plot, E_euler, 'g', label='Total simulated energy')
    axis[1, i].plot(t_plot, E_analytic, 'k--', label='Total analytic energy')
    axis[1, i].set_xlabel('t/number_of_timesteps')
    axis[1, i].set_ylim([0, 5])
    axis[0, i].set_ylim([-5, 5])
    axis[0, i].set_xticks([])
    axis[0, i].set_xlim([0, 4])
    axis[1, i].set_xlim([0, 4])
    if i > 0:
        axis[0, i].set_yticks([])
        axis[1, i].set_yticks([])



fig1.suptitle(f'Euler algorithm')
axis[0, 0].set_ylabel('r/A')
axis[1, 0].set_ylabel('E/E$_0$')

plt.tight_layout()
plt.show()
