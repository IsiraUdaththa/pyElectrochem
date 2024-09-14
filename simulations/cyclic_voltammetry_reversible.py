import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants

F = constants.physical_constants['Faraday constant'][0]  # Faraday constant, 96485 C/mol
R = constants.R  # Gas constant, 8.314 J/(mol·K)
MOL_CONVERSION = 1000  # mol/L to mol/m³ conversion factor


def laplacian(f, delta_x):
    """
    Computes the Laplacian second derivative for given function

    Parameters:
    f : np.ndarray
        Array containing the values of the function.
    delta_x : float
        Step size for spatial derivative.

    Returns:
    np.ndarray
        Laplacian of the array.
    """
    return np.gradient(np.gradient(f, delta_x, edge_order=2), delta_x, edge_order=2)


def potential(Ei, Ef, nu, sampling_steps, nCycle):
    """
    Generate the potential profile as a function of time for cyclic simulations.

    Parameters:
        Ei (float): Initial potential (V)
        Ef (float): Final potential (V)
        nu (float): Scan rate (V/s)
        sampling_steps (int): Number of steps for one forward scan
        nCycle (float): Number of cycles, should be a multiple of 0.5

    Returns:
        Efull (list): Full potential profile for the entire scan
    """
    if nCycle % 0.5 != 0:
        raise ValueError("The number of cycles should be a multiple of 0.5")

    half_period_duration = abs(Ef - Ei) / nu
    t = np.linspace(0, half_period_duration, sampling_steps, endpoint=False)
    # Generate the forward and backward potential profiles based on Ei and Ef
    if Ei < Ef:
        Eforward = Ei + nu * t
        Ebackward = Ef - nu * t
    elif Ei > Ef:
        Eforward = Ei - nu * t
        Ebackward = Ef + nu * t
    else:
        raise ValueError("The starting and ending potential are the same.")
    single_cycle = np.concatenate((Eforward, Ebackward))
    num_full_cycles = int(nCycle)
    Efull = np.tile(single_cycle, num_full_cycles)

    # Handle the half-cycle if needed
    if nCycle % 1 != 0:
        Efull = np.concatenate((Efull, Eforward))  # Add forward scan for half-cycle
        Efull = np.append(Efull, Ef)  # End at Ef for the half cycle
    else:
        Efull = np.append(Efull, Ei)  # End at Ei for a full cycle
    return Efull


def calculate_intensity(n, A, D, C, delta_x):
    """
    Calculates the intensity at each time step. (Faraday's Law of Electrolysis)

    Parameters:
    C : np.ndarray
        Concentration array.
    n : int
        Number of electrons.
    F : float
        Faraday constant.
    D : np.ndarray
        Diffusion coefficients.
    delta_x : float
        Spatial step size.
    """

    gradCx = np.gradient(C[:, :, 0], delta_x, axis=0)

    return n * F * A * D * gradCx[0, :]


def solve_diffusion_equation(C, t, D, delta_t, delta_x):
    """
    Solve the diffusion equation (Fick's Second Law) for each species at time t.

    The laplacian at step t-1 is taken and
    the partial derivative equation is propagated for time t as C(t)=C(t-1)+D*delta_t*lapC(t-1)

    Parameters:
        C: np.ndarray
            Array containing the concentration of species at all points and times.
        t: int
            Index of the current time step.
        D: np.ndarray
            Diffusion coefficients for the species.
        delta_t: float
            Time step size.
        delta_x: float
            Spatial step size.

    Returns:
        newC: np.ndarray
            Updated concentration array after solving the diffusion equation.
    """
    newC = np.zeros_like(C[:, 0, :])

    for i in range(D.size):
        lapC = laplacian(C[:, t - 1, i], delta_x)
        newC[:, i] = C[:, t - 1, i] + D[i] * delta_t * lapC

    return newC


def apply_boundary_conditions(C, t, D, E, E0, n, T):
    """
    Apply the boundary conditions at the electrode surface using the Nernst equation.

    At the electrode surface, the concentration of oxidized and reduced species is related by the Nernst Equation

    C_red/C_ox = exp (-(E-E0)*nF/RT) = theta (Nernst Constant)
    C_red(0) - theta * C_ox(0) = 0

    D_red grad(C_red)+ D_ox grad (C_ox)_0 = 0 (conservation of matter)
    -D_red C_red(0) - D_ox C_ox(0) = -D_redC_red(1)-D_oxC_ox(1) (First order approximation for the gradient)

    Parameters:
        C: np.ndarray
            Array containing the concentration of species at all points and times.
        t: int
            Index of the current time step.
        D: np.ndarray
            Diffusion coefficients for the species.
        E: np.ndarray
            Array of voltages at each time step.
        E0: float
            Standard redox potential.
        n: int
            Number of electrons involved in the reaction.
        T: float
            Temperature in Kelvin.

    Returns:
        boundary_conditions: np.ndarray
            Updated concentrations at the boundary.
    """
    # Calculate the Nernst factor (theta)
    theta = np.exp(-(E[t] - E0) * n * F / (R * T))

    # Set up the matrix form AC = B
    A = np.array([[1, -theta], [-D[0], -D[1]]])
    B = np.array([0, -D[0] * C[1, t - 1, 0] - D[1] * C[1, t - 1, 1]])

    # Solve the system of linear equations
    sol = np.linalg.solve(A, B)

    return np.transpose(sol)


def calculate_concentration(C, t, D, Cini, E, E0, n, T, delta_t, delta_x):
    """
    Computes the concentration for the next time step based on the current concentration profile.

    Parameters:
        C: np.ndarray
            Array containing the concentration of species at all points and times.
        t: int
            Index of the current time step.
        D: np.ndarray
            Diffusion coefficients for the species.
        Cini: np.ndarray
            Initial concentration values.
        E: np.ndarray
            Array of voltages at each time step.
        E0: float
            Standard redox potential.
        n: int
            Number of electrons involved in the reaction.
        T: float
            Temperature in Kelvin.
        delta_t: float
            Time step size.
        delta_x: float
            Spatial step size.

    Returns:
        newC: np.ndarray
            Updated concentration array for the next time step.
    """
    # Solve the diffusion equation
    newC = solve_diffusion_equation(C, t, D, delta_t, delta_x)

    # Apply the boundary conditions
    newC[0] = apply_boundary_conditions(C, t, D, E, E0, n, T)

    return newC


def animate(time, C, intensity, E, t, x):
    """
    Animate the concentrations and dots on the corresponding graphs
    """
    Cox.set_data(x, C[:, time, 1] / MOL_CONVERSION)
    Cred.set_data(x, C[:, time, 0] / MOL_CONVERSION)
    Et.set_data([t[time]], [Efull[time]])
    it.set_data([t[time]], [intensity[time]])
    iE.set_data([E[time]], [intensity[time]])
    # ax2.set_ylim(np.min([-1,np.min(lapC[:,time])*1.05]),0.)

    return [Cox, Cred, Et, it, iE]


if __name__ == "__main__":
    Ei = 0.  # Initial potential in Volt
    Ef = 1.5  # Other end of the potential ramp in Volt
    E0 = 0.77  # Redox potential of the couple vs ESH

    n = 1  # Number of electrons exchanged
    nu = 50.e-3  # sweep rate in V/s

    D_ox = 6.04e-10  # Diffusion coefficient of the oxydant in m^2/s
    D_red = 6.04e-10  # Diffusion coefficient of the reductor in m^2/s

    C_ox = 0.0  # Initial concentration of the oxydant at the electrode in mol/L
    C_red = 0.05  # Initial concentration of the oxydant at the electrode in mol/L

    A = 1.e-4  # Area of the electrode in m^2
    l = 5.e-4  # Length of the simulation box

    nCycle = 1.  # number of cycles between Ei and Ef, it should be a multiple of 0.5

    sampling_t = 10000  # Sampling of a forward scan
    sampling_x = 100  # Sampling of the simulation box

    T = 298.15  # Temperature in Kelvin

    half_period_duration = abs(Ef - Ei) / nu

    # Creating the x values at which the concentration wil be computed each step.
    x, delta_x = np.linspace(0, l, sampling_x, retstep=True)

    # Splitting the time to make it correspond to the defined sampling and number of cycles
    t, delta_t = np.linspace(0, 2 * half_period_duration * nCycle, (int(2 * sampling_t * nCycle) + 1), retstep=True)

    # Creating the seesaw voltage
    Efull = potential(Ei, Ef, nu, sampling_t, nCycle)

    # Diffusion coefficients for all the species
    D = np.array([D_red, D_ox])
    Cini = MOL_CONVERSION * np.array([C_red, C_ox])

    C = np.zeros((sampling_x, len(t), 2))
    # Initial condition for Cred(x,0) : C(x,0,0) = C_red
    C[:, 0, 0] = C_red * MOL_CONVERSION * np.ones(sampling_x)
    # Initial condition for Cox(x,0) : C(x,0,1) = C_ox
    C[:, 0, 1] = C_ox * MOL_CONVERSION * np.ones(sampling_x)

    DM = np.minimum(D_ox, D_red) * delta_t / (delta_x ** 2)
    print('DM : {}'.format(DM))
    if DM > 0.5:
        print("the sampling is too scarce, choose more wisely : decrease t sampling or raise x sampling")

    # Frame interval to have an animation roughly at 50 fps
    frames_interval = int(len(t) / (50 * t[-1]))
    print(
        'Display every {} step, total length(s) {}, total simulation frames {} '.format(frames_interval, t[-1], len(t)))
    print('D*T {} , length^2 {}'.format(np.max(D) * t[-1], l ** 2))

    # Computation of the concentration at each position and time
    for time in range(1, len(t)):
        C[:, time, :] = calculate_concentration(C, time, D, Cini, Efull, E0, n, T, delta_t, delta_x)

    # Computation of the current from the concentration profiles
    i = calculate_intensity(n, A, D[0], C, delta_x)

    # Plotting of all the results
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # C=f(x)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Concentration')
    ax1.set_xlim(0., l)
    ax1.set_ylim(0., C_red * 1.05)

    # i, E = ft(t)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('intensity (A)')
    ax2.plot(t, i, label='i', color='C0')
    ax2_2 = ax2.twinx()
    ax2_2.plot(t, Efull, label='E', color='C1')
    ax2.legend(loc='upper left')
    ax2_2.legend(loc='upper right')

    # i = f(E)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('intensity (A)')
    ax3.plot(Efull, i, label='intensity', marker=None)

    # C(x=0) = f(t)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_xlabel('time (m)')
    ax4.set_ylabel('c at electrode')
    ax4.plot(t, C[0, :, 0] / MOL_CONVERSION, label='red', color='C0')
    ax4.plot(t, C[0, :, 1] / MOL_CONVERSION, label='ox', color='C1')
    ax4.legend(loc='upper right')

    # lines to animate
    Cox, = ax1.plot([], [], color='C2')
    Cred, = ax1.plot([], [], color='C3')
    Et, = ax2_2.plot([], [], marker='o', ms=2)
    it, = ax2.plot([], [], marker='o', ms=2)
    iE, = ax3.plot([], [], marker='o', ms=3, color='C0')
    plt.tight_layout()
    # animate all the lines as a function of time
    ani = animation.FuncAnimation(fig, animate, fargs=(C, i, Efull, t, x), blit=True,
                                  frames=range(0, len(t) + 1, frames_interval), interval=20)

    # ,save_count=int(sizet/frameselect))
    # filename = "simulations-r-sweep-{}-E0-{}-C_ox-{}-C_red-{}-Ei-{}-Ef-{}".format(nu, E0, C_ox, C_red, Ei, Ef)
    # save the animation as a mp4 file
    # writermp4 = animation.FFMpegWriter(fps=50)
    # ani.save(filename + '.mp4', writer=writermp4)
    # # save the i=f(E) curve as a csv file
    # np.savetxt(filename + '.csv', np.transpose([Efull, i]), delimiter=",")
    # #  Save all the produced data as a numpy file, (i, V, t, x, C=f(x,t)
    # with open(filename + '.npy', 'wb') as fileOutput:
    #     np.save(fileOutput, [Efull, i, t])
    #     np.save(fileOutput, x)
    #     np.save(fileOutput, C)

    plt.show()
