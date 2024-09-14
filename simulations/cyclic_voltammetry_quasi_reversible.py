import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants

from cyclic_voltammetry_reversible import calculate_intensity, potential, solve_diffusion_equation

F = constants.physical_constants['Faraday constant'][0]  # Faraday constant, 96485 C/mol
R = constants.R  # Gas constant, 8.314 J/(mol·K)
MOL_CONVERSION = 1000  # mol/L to mol/m³ conversion factor


def apply_boundary_conditions(C, t, D, E, E0, n, T):
    """
    Apply the boundary conditions at the electrode surface using the Butler Volmer equation.

    nFA D_red grad C_red(0) =
    i = nFA k0 (C_red exp (alpha n F /(RT) (E-E0)) -C_ox * exp( - (1-alpha) nF/(RT) (E-E0))
    which couple the gradient and the concentration at the electrode
    grad C \approx (C[1,t,i]-C[0,t,i])/delta x -> equation (5.23) of Britz & Strutwolf

    By conservation of matter,
    D_red grad(C_red)+ D_ox grad (C_ox)_0 = 0
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

    theta = n * F / (R * T) * (E[t] - E0)
    k_red = k0 * np.exp(alpha * theta)
    k_ox = k0 * np.exp(-(1 - alpha) * theta)

    # Set up the matrix form AC = B
    A = np.array([[1 + k_red * delta_x / D[0], -k_ox * delta_x / D[0]], [-D[0], -D[1]]])
    B = np.array([C[1, t - 1, 0], -D[0] * C[1, t - 1, 0] - D[1] * C[1, t - 1, 1]])

    # Solve the system of linear equations
    sol = np.linalg.solve(A, B)

    return np.transpose(sol)


def calculate_concentration(C, t, D, Cini, E, E0, n, T, alpha, k0, delta_t, delta_x):
    """
    Compute the concentration for the next time interval t+delta t from the concentration given at t,x
    First the laplacian at step t-1 is taken and the partial derivative equation is propagated for time t as
    C(t)=C(t-1)+D*delta_t*lapC(t-1)
        C array containing the concentration, first index position, second index : time, third index species
        t indix of the time to consider
        D array containing the diffusion coefficients of the species
        delta_t : time interval
        delta_x : space interval
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
    Et.set_data(t[time], Efull[time])
    it.set_data(t[time], intensity[time])
    iE.set_data(E[time], intensity[time])
    # ax2.set_ylim(np.min([-1,np.min(lapC[:,time])*1.05]),0.)

    return [Cox, Cred, Et, it, iE]


if __name__ == "__main__":
    alpha = 0.5  # asymmetry factor for Butler Volmer equation (alpha + beta = 1)
    k0 = 1e-6  # kinetic constant for the couple

    Ei = 0.  # Initial potential
    Ef = 1.5  # Potential for sweep
    E0 = 0.77  # Standard potential for the couple

    n = 1  # Number of electrons exchanged
    nu = 50.e-3  # sweep rate as V/s

    D_ox = 6.04e-10  # Diffusion coefficient of the oxydant in m^2/s
    D_red = 6.04e-10  # Diffusion coefficient of the reductor in m^2/s

    C_ox = 0.  # Initial concentration of the oxydant at the electrode in mol/L
    C_red = 0.05  # initial concentration of the oxydant at the electrode in mol/L

    A = 1e-4  # Area of the electrode in m^2
    l = 5e-4  # Length of the simulation box

    nCycle = 1.  # number of cycles between Ei and Ef, it should be a multiple of 0.5

    sampling_x = 100  # Sampling of the simulation box
    sampling_t = 10000  # sampling for a forward scan

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
    print('Psi {}'.format(k0 / np.sqrt(np.pi * D_red * n * F * nu / (R * T))))

    # computation of the concentration at each position and time
    for time in range(1, len(t)):
        C[:, time, :] = calculate_concentration(C, time, D, Cini, Efull, E0, n, T, alpha, k0, delta_t, delta_x)

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
    ax2_3 = plt.subplot(2, 2, 4)
    ax2_3.set_xlabel('time (m)')
    ax2_3.set_ylabel('c at electrode')
    ax2_3.plot(t, C[0, :, 0] / MOL_CONVERSION, label='red', color='C0')
    ax2_3.plot(t, C[0, :, 1] / MOL_CONVERSION, label='ox', color='C1')
    ax2_3.legend(loc='upper right')

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

    # ,save_count=int(len(t)/frameselect))
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
