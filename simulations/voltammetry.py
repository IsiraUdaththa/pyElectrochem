#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code provided as is, without any warranty or whatsoever
Made by Martin Vérot from the ENS de Lyon, France
The inputs are defined after the "main program" comment line (diffusion coefficients, temperature, etc...)

This code is under licence CC-BY-NC-SA
It means that you cannot make profit from it, you need to mention the original author and if you reuse it, you must take the same licencing
Otherwise, feel free to use it and enjoy the beauty of electrochemistry !


##########################################################################
Voltammetry for a rapid couple with the Nernst law as a boundary condition
#########################################################################

The numerical simulations come out of
- Britz & Strutwolf : Digital simulation in Electrochemistry (2016) p96-98
- Bard & Faulkner : Electrochemical Methods Fundamentals and Applications (2001) annex B
- Girault : Électrochimie Physique et Analytique (2007) chapter 10 (the book exists in english "ANALYTICAL AND PHYSICAL ELECTROCHEMISTRY"

I took variables with dimensions so as to stick as close as possible to the physical equations and keep them as straightforward as possible

Britz and Strutwolf propose some programs as fortran codes, it is helpful to see how things are done by more seasoned programmers, however, the code is less readable as quite a lot of 'for' are involved where numpy slicing make it much shorter
A program named ESP  (Electrochemical Simulations Package) also offers some simulations but the source code is not open and needs some compiling and it seems to be hard to run on recent computers


The sampling either being to tight or sparse can lead to big numerical instabilities, you can change the numbers to make the computations more precise or faster, but if you see things going to infinity or zero, the sampling is probably at fault
"""

# Importation of libraries 
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import matplotlib.animation as animation
import argparse

# Functions used by the main program 
def nextC(C,t,D,Cini,E,E0,n,T,deltat,deltax):
    """
    Compute the concentration for the next time interval t+delta t from the concentration given at t,x
    First the laplacian at step t-1 is taken and the partial derivative equation is propagated for time t as
    C(t)=C(t-1)+D*deltat*lapC(t-1)
        C array containing the concentration, first index position, second index : time, third index species
        t indix of the time to consider
        D array containing the diffusion coefficients of the species
        deltat : time interval
        deltax : space interval
    """
    F = constants.physical_constants['Faraday constant'][0]
    R = constants.R
    newC = np.zeros_like(C[:,0,:])
    #solving the diffusion equations for each species
    for i in range(0,D.size):
        lapC = laplacian(C[:,t-1,i],deltax) 
        newC[:,i] = C[:,t-1,i] + D[i]*deltat*lapC
        #Checking the concentrations at the end of the simulation
        #if newC[-1,i] != Cini[i]:
        #    print('the box considered is too short to be considered as a semi-infinite model')
    #now solving the boundary conditions : Nernst equation at the electrodes and conservation of matter
    #C_red/C_ox = exp (-(E-E0)*nF/RT) = theta (Nernst) -> C_red(0) - theta * C_ox(0) = 0
    #D_red grad(C_red)+ D_ox grad (C_ox)_0 = 0 (conservation of matter) -> -D_red C_red(0) - D_ox C_ox(0) = -D_redC_red(1)-D_oxC_ox(1) (First order approximation for the gradient)
    #In its matrix form AC=B 
    theta=np.exp(-(E[t]-E0)*n*F/(R*T))
    #print(alpha)
    A = np.array([[1,-theta],[-D[0],-D[1]]])
    B = np.array([0,-D[0]*C[1,t-1,0]-D[1]*C[1,t-1,1]])
    sol = np.linalg.solve(A, B)
    sol = sol.reshape(len(sol), 1)
    newC[0] = np.transpose(sol)
    #print(newC[0])
    return newC

def intensity(C,x,deltax,deltat,n,A,D):
    """
    compute the intensity from the concentration profile
    """
    gradCx = np.gradient(C[:,:,0],deltax,axis=0)
    
    F = constants.physical_constants['Faraday constant'][0]
    return n*F*A*D[0] * gradCx[0,:]#/deltax

def laplacian(f,deltax):
    """
    Computes the laplacian second derivative with central formula
        f : is the array containing the values of the function
        deltax : sampling interval
    """
    out = np.zeros_like(f)
    out[1:-1]=(f[2:]-2*f[1:-1]+f[0:-2])/deltax**2
    out[0]=(f[2]-2*f[1]+f[0])/deltax**2
    out[-1]=(f[-3]-2*f[-2]+f[-1])/deltax**2
    return out

def lap(C,deltax,t,species):
    """
    Compute the laplacian for species with index 'species' (0 : red, 1 : ox) at time t
        C : array containing the concentrations
        deltax : x sampling interval
        t : indix of the time in the array
        species : indix of the species for which the laplacian must be evaluated
    """
    return laplacian(C[:,t,species],deltax)


def halfPeriod(Ei,Ef,nu):
    """ 
    Compute the time of a half period
        Ei : starting potential (V)
        Ef : ending potential (V)
        nu : scan rate (V/s)
    """
    halfPeriod = np.abs((float(Ef)-float(Ei))/nu)
    return halfPeriod
         

def potential(Ei,Ef,nu,samplingt,nCycle):
    """
    Create the potential as a function of time
        Ei : starting potential (V)
        Ef : ending potential (V)
        nu : scan rate (V/s)
        sampling t : number of steps for a single forward scan (int)
        nCycle : number of cycles, it should be a multiple of 0.5
    """
    halfPer = halfPeriod(Ei,Ef,nu)
    t,deltat = np.linspace(0,halfPer,samplingt,endpoint=False,retstep=True)
    #Creating the scan in the correcti direction depending on the staring and ending potential
    if Ei<Ef:
        Eforward = Ei + nu * t 
        Ebackward = Ef - nu * t 
    elif Ei>Ef:
        Eforward = Ei - nu * t 
        Ebackward = Ef + nu * t 
    else:
        print('The starting and ending potential are the same')
    if nCycle % 0.5 != 0:
        print('The number of cycles should be a multiple of 0.5')
    Ecycle = [list(Eforward),list(Ebackward)]
    Efull = []
    for i in range(0,int(nCycle/0.5)):
        Efull.extend(Ecycle[i%2])
    #Adding the final point to make a full cycle ending either at Ei or Ef
    if nCycle % 1 == 0:
        Efull.append(Ei)
    elif nCycle % 0.5 ==0.:
        Efull.append(Ef)
    return np.array( Efull )

def animate(time,C,intensity,E,t,x,convertMoll):
    """
    Animate the concentrations and dots on the corresponding graphs
    """
    Cox.set_data(x, C[:,time,1]/convertMoll)
    Cred.set_data(x, C[:,time,0]/convertMoll)
    Et.set_data(t[time],Efull[time] )
    it.set_data(t[time], intensity[time])
    iE.set_data(E[time], intensity[time])
    #ax2.set_ylim(np.min([-1,np.min(lapC[:,time])*1.05]),0.)
    
    return [Cox,Cred,Et,it,iE]

     
#Main programm 
if __name__ == "__main__":
    #Input parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-Ei", "--initialPotential", type=float, default = 0., help="Initial potential in Volt",dest="Ei")
    parser.add_argument("-Ef", "--sweepPotential"  , type=float, default = 1.5, help="Other end of the potential ramp in Volt",dest="Ef")
    parser.add_argument("-E0", "--couplePotential"  , type=float, default = 0.77, help="Redox potential of the couple vs ESH",dest="E0")
    parser.add_argument("-n", "--nbElectrons"  , type=int, default = 1, help="Number of electrons exchanged during the oxydo reduction process",dest="n")
    parser.add_argument("-nu", "--sweepRate"  , type=float, default = 50.e-3, help="sweep rate in V/s",dest="nu")
    parser.add_argument("-D_ox", "--DiffusionCoefficientOxidant"  , type=float, default = 6.04e-10, help="Diffusion coefficient of the oxydant in m^2/s",dest="D_ox")
    parser.add_argument("-D_red", "--DiffusionCoefficientiReductor"  , type=float, default = 6.04e-10, help="Diffusion coefficient of the reductor in m^2/s",dest="D_red")
    parser.add_argument("-C_ox", "--ConcentrationOxidant"  , type=float, default = 0.0, help="Concentration of the oxidant in mol/L",dest="C_ox")
    parser.add_argument("-C_red", "--ConcentrationReductor"  , type=float, default = 0.05, help="Concentration of the reductor in mol/L",dest="C_red")
    parser.add_argument("-A", "--Area"  , type=float, default = 1.e-4, help="Area of the electrode in m^2",dest="A")
    parser.add_argument("-l", "--length"  , type=float, default = 5.e-4, help="Length of the simulation box",dest="l")
    parser.add_argument("-nCycle", "--nbCycle"  , type=float, default = 1., help="number of cycles between Ei and Ef, it should be a multiple of 0.5",dest="nCycle")
    parser.add_argument("-samplingx", "--samplingx"  , type=int, default = 100, help="Sampling of the simulation box",dest="samplingx")
    parser.add_argument("-samplingt", "--samplingt"  , type=int, default = 10000, help="Sampling of a forward scan",dest="samplingt")
    parser.add_argument("-T", "--Temperature"  , type=float, default = 298.15, help="Temperature in Kelvin",dest="T")
    parser.add_argument("-csv", "--saveCsv"    , type=bool, default = True, help="save the i=f(E) curve as a csv file",dest="saveCsv")
    parser.add_argument("-mp4", "--saveMovie"  , type=bool, default = False, help="save the animation as a mp4 file",dest="saveMovie")
    parser.add_argument("-movie", "--movieName"    , type=str, default = 'voltammmetry.mp4', help="name of the voltammetry file",dest="movie")
    parser.add_argument("-npy", "--saveNpy"    , type=bool, default = False, help="Save all the produced data as a numpy file, (i, V, t, x, C=f(x,t), the file produced can be really huge",dest="saveNpy")
    args = parser.parse_args()
    
    #########################
    #the program starts here
    #########################
    F = constants.physical_constants['Faraday constant'][0]
    R = constants.R
    convertMoll = 1000 #to convert mol/L to mol/m^3
    #length of a forward scan
    halfPer= halfPeriod(args.Ei,args.Ef,args.nu) 
    #Creating the x values at which the concentration wil be computed each step.    
    x,deltax = np.linspace(0,args.l,args.samplingx,retstep=True)
    #Splitting the time to make it correspond to the defined sampling and number of cycles
    sizet = int(args.samplingt*2*args.nCycle)+1
    t,deltat = np.linspace(0,2*args.nCycle*halfPer,sizet,retstep=True)
    #Creating the seesaw voltage    
    Efull = potential(args.Ei,args.Ef,args.nu,args.samplingt,args.nCycle)
    #Diffusion coefficients for all the species
    D = np.array([args.D_red,args.D_ox])
    Cini = convertMoll*np.array([args.C_red,args.C_ox])
    C = np.zeros((args.samplingx,sizet,2)) 
    #Initial condition for Cred(x,0) : C(x,0,0) = C_red 
    C[:,0,0] = args.C_red* convertMoll * np.ones(args.samplingx)
    #Initial condition for Cox(x,0) : C(x,0,1) = C_ox 
    C[:,0,1] = args.C_ox* convertMoll * np.ones(args.samplingx)

    DM = np.minimum(args.D_ox,args.D_red)*deltat/(deltax**2 )
    print('DM : {}'.format(DM))
    if DM > 0.5:
        print("the sampling is too scarce, choose more wisely : decrease t sampling or raise x sampling")
    #Frame interval to have an animation roughly at 50 fps
    frameselect = int(sizet/(50*t[-1]))
    print('display every {} step, length(s) {}, sizet {} '.format(frameselect,t[-1], sizet))
    print('D*T {} , length^2 {}'.format(np.max(D)*t[-1], args.l**2))
    

    #computation of the concentration at each position and time
    for time in range(1,sizet):
        C[:,time,:]=nextC(C,time,D,Cini,Efull,args.E0,args.n,args.T,deltat,deltax)

    #Computation of the current from the concentration profiles
    i = intensity(C,x,deltax,deltat,args.n,args.A,D)
   
    #Plotting of all the results
    fig, axes = plt.subplots(2,2, figsize =(10,6))
    #C=f(x)
    ax1 = plt.subplot(2,2,1)
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Concentration')
    ax1.set_xlim(0.,args.l)
    ax1.set_ylim(0.,args.C_red*1.05)
    #i, E = ft(t)
    ax2 = plt.subplot(2,2,2)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('intensity (A)')
    ax2.plot(t,i,label = 'i',color='C0') 
    ax2_2 = ax2.twinx() 
    ax2_2.plot(t,Efull,label = 'E',color='C1') 
    ax2.legend(loc='upper left')
    ax2_2.legend(loc='upper right')
    #i = f(E)
    ax3 = plt.subplot(2,2,3)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('intensity (A)')
    ax3.plot(Efull,i,label = 'intensity',marker=None) 
    #C(x=0) = f(t)
    ax2_3 =plt.subplot(2,2,4)  
    ax2_3.set_xlabel('time (m)')
    ax2_3.set_ylabel('c at electrode')
    ax2_3.plot(t,C[0,:,0]/convertMoll,label = 'red',color='C0') 
    ax2_3.plot(t,C[0,:,1]/convertMoll,label = 'ox',color='C1') 
    ax2_3.legend(loc='upper right')
    #lines to animate
    Cox, = ax1.plot([], [],color='C2' )
    Cred, = ax1.plot([], [],color='C3' )
    Et, = ax2_2.plot([], [] , marker='o',ms=2)
    it, = ax2.plot([], [] , marker='o',ms=2)
    iE, = ax3.plot([], [] , marker='o',ms=3,color='C0')
    plt.tight_layout()
    #animate all the lines as a function of time
    ani = animation.FuncAnimation(fig, animate, fargs =(C,i,Efull,t,x,convertMoll), blit=True , frames=range(0,sizet+1,frameselect),interval=20)#,save_count=int(sizet/frameselect))
    #write the file as a mp4
    writermp4 = animation.FFMpegWriter(fps=50) 
    if args.saveMovie == True:
        ani.save(args.movie, writer=writermp4)
    filename ="voltammetry-r-sweep-{}-E0-{}-C_ox-{}-C_red-{}-args.Ei-{}-Ef-{}".format(args.nu,args.E0,args.C_ox,args.C_red,args.Ei,args.Ef) 
    if args.saveCsv == True:
        np.savetxt(filename+'.csv', np.transpose([Efull,i]), delimiter=",")
    if args.saveNpy == True:
        with open(filename+'.npy','wb') as fileOutput:
            np.save(fileOutput, [Efull,i,t])
            np.save(fileOutput, x)
            np.save(fileOutput, C  )
    plt.show()
    pass
