import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants

from voltammetry import laplacian

F = constants.physical_constants['Faraday constant'][0]  # Faraday constant, 96485 C/mol
R = constants.R  # Gas constant, 8.314 J/(mol·K)
MOL_CONVERSION = 1000  # mol/L to mol/m³ conversion factor


def nextC(C, t, D_red, delta_t, delta_x):
    """
    Compute the concentration for the next time interval t+delta t from the concentration given at t,x
    First the laplacian at step t-1 is taken and the partial derivative equation is propagated for time t as
    C(t)=C(t-1)+D*delta_t*lapC(t-1)
    """
    lapC = laplacian(C[:, t - 1], delta_x)
    newC = C[:, t - 1] + D_red * delta_t * lapC
    newC[0] = 0.
    return newC


def intensity(n, A, D_red, C, delta_x):
    """
    compute the intensity from the concentration profile
    """
    gradCx, gradCt = np.gradient(C, delta_x)
    return n * F * A * D_red * gradCx[0, :]


def Cottrell(t, n, A, C_red, D_red):
    """
    Cottrell law for chronoamperometry i = n*F*A*C_red*sqrt(D_red/(Pi*t))
    """
    out = np.zeros_like(t)
    out[1:] = n * F * A * C_red * MOL_CONVERSION * np.sqrt(D_red / (np.pi * t[1:]))
    out[0] = np.inf
    return out


def animate(time, C, lapC, intensity, t, x):
    line1.set_data(x, C[:, time] / MOL_CONVERSION)
    line2.set_data(x, lapC[:, time])
    line3.set_data([t[time]], [intensity[time]])
    ax2.set_ylim(np.min([-1, np.min(lapC[:, time]) * 1.05]), 0.)
    return [line1, line2, line3]


if __name__ == "__main__":
    D_red = 7.19e-10  # Diffusion coefficient of the reductor in m^2/s
    C_red = 0.05  # Initial concentration of the oxydant at the electrode in mol/L

    A = 1e-4  # Area of the electrode in m^2
    l = 1e-3  # length in meter

    tfin = 5  # ending time in seconds
    sampling_x = 250
    sampling_t = 1000

    n = 1  # Number of electrons exchanged
    T = 298.15  # Temperature in Kelvin

    x, delta_x = np.linspace(0, l, sampling_x, retstep=True)
    t, delta_t = np.linspace(0, tfin, sampling_t, retstep=True)

    DM = D_red * delta_t / (delta_x ** 2)
    print('DM : {}'.format(DM))
    if DM > 0.5:
        print("the sampling is too scarce, choose more wisely")

    # Initializing the array for the concentrations
    C = np.zeros((sampling_x, sampling_t))
    lapC = np.zeros((sampling_x, sampling_t))

    # Initial condition for C(x,0) : C(x,0) = C_red
    C[:, 0] = C_red * MOL_CONVERSION * np.ones(sampling_x)
    lapC[:, 0] = laplacian(C[:, 0], delta_x)

    # just after step function for potential
    C[:, 1] = C_red * MOL_CONVERSION * np.ones(sampling_x)
    C[0, 1] = 0
    lapC[:, 1] = laplacian(C[:, 1], delta_x)

    # Plotting of all the results
    fig, axes = plt.subplots(2, 2)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2)

    for time in range(2, sampling_t):
        # shape = lapC.shape
        # print('lapC : {}'.format(shape))
        C[:, time] = nextC(C, time, D_red, delta_t, delta_x)
        lapC[:, time] = laplacian(C[:, time], delta_x)

    # lines to animate
    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])
    line3, = ax3.plot([], [], marker='o')
    lines = [line1, line2, line3]

    i_th = Cottrell(t, n, A, C_red, D_red)
    i = intensity(n, A, D_red, C, delta_x)

    # C = f(t)
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Concentration')
    ax1.set_xlim(0., l)
    ax1.set_ylim(0., C_red * 1.05)

    # laplacian = f(t)
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Laplacian')
    ax2.set_xlim(0., l)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Current')

    # i = f(t)
    ax3.plot(t, i, label='i_solve')
    ax3.plot(t, i_th, label='i_theo')
    ax3.legend(loc='upper right')
    ratio = i_th / i

    print('max I_th/I : {}'.format(ratio[1:].max()))
    plt.tight_layout()

    ani = animation.FuncAnimation(fig, animate, fargs=(C, lapC, i, t, x), frames=range(t.size),
                                  interval=delta_t / 1000)
    # writermp4 = animation.FFMpegWriter(fps=int(1 / delta_t))
    # ani.save("chronoamperometry.mp4", writer=writermp4)
    plt.show()
