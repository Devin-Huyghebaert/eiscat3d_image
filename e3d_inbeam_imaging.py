#!/usr/bin/env python

"""
Created by: Devin Huyghebaert, Björn Gustavsson, Ilkka Virtanen, Juha Vierinen in January, 2024 for EISCAT 3D Imaging Testing

Further explanations of the following assumptions can be found in Huyghebaert et al. (2024) - reference below

Current assumptions:    - scattering volume is stationary over sampling period (spectra for each sampling point does not change)
                        - plasma parameters are constant in the model input volume
                        - repeated baselines don't have significant variation in total distance of signal
                        - ionosphere is parallel to and centered on the main array at Skibotn
                        - The radar is observing along the geomagnetic field line
                        - transmit power all originates from the center of the main array
                        - no self-clutter (also known as self-interference) exists in the model
                        - no side-lobe interference/addition to the signal exists
                        - there are no differences in the power received at each radar panel in the main array due to distance
                         
---
Software is based on the following publications:

---
For the ISR Spectrum Generation:

https://github.com/jswoboda/ISRSpectrum

J. Swoboda, “SPACE-TIME SAMPLING STRATEGIES FOR ELECTRONICALLY STEERABLE INCOHERENT SCATTER RADAR,” Boston University, 2016.

J. Swoboda, J. Semeter, M. Zettergren, and P. J. Erickson, “Observability of ionospheric space-time structure with ISR: A simulation study,” Radio Science, vol. 52, no. 2, pp. 1–20, Feb. 2017, doi: 10.1002/2016RS006182.

---
For the Imaging:

Stamm, J., Vierinen, J., Urco, J. M., Gustavsson, B., and Chau, J. L.: Radar imaging with EISCAT 3D, Ann. Geophys., 39, 119–134, https://doi.org/10.5194/angeo-39-119-2021, 2021.

Huyghebaert, D., Gustavsson, B., Vierinen, J., Kvammen, A., Zettergren, M., Swoboda, J., Virtanen, I., Hatch, S., and Laundal, K. M.: Interferometric Imaging with EISCAT_3D for Fine-Scale In-Beam Incoherent Scatter Spectra Measurements, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-802, 2024. 

---
For the Error analysis:

Vallinkoski, M.: Statistics of incoherent scatter multiparameter fits, Journal of Atmospheric and Terrestrial Physics, 50, 839–851, https://doi.org/https://doi.org/10.1016/0021-9169(88)90106-7, 1988.

EISCAT_3D measurement methods handbook
http://urn.fi/urn:isbn:9789526205854

"""

import sys
import numpy as np
import time
import scipy.constants as spconst
from ISRSpectrum import Specinit
import matplotlib.pylab as plt
import h5py
import matplotlib as mpl
from mpl_toolkits.mplot3d.axis3d import Axis
import configparser
config = configparser.ConfigParser()

mpl.rcParams['figure.figsize'] = [30.0, 24.0]
mpl.rcParams.update({'font.size': 32})
np.set_printoptions(threshold=sys.maxsize)

#-----
#-----
#Read in data from the input configuration file
#-----
#-----

config.read('e3d_image_config.ini')
#File details
input_model_file = config.get('file_info','input_model_file')
input_data_field_ne = config.get('file_info','input_data_field_ne')
input_data_field_ti = config.get('file_info','input_data_field_ti')
input_data_field_vi = config.get('file_info','input_data_field_vi')
input_data_field_te = config.get('file_info','input_data_field_te')
output_data_location = config.get('file_info','output_data_location')
output_data_name = config.get('file_info','output_data_name')

#model details
spatial_res = float(config.get('model_input','spatial_res'))
x_num = int(config.get('model_input','x_num'))
y_num = int(config.get('model_input','y_num'))
altitude = float(config.get('model_input','altitude'))

#EISCAT 3D Radar Parameters
TX_beamwidth = float(config.get('radar_param','TX_beamwidth'))
radar_power = float(config.get('radar_param','radar_power'))
RX_noise = float(config.get('radar_param','RX_noise'))

#Radar Experiment parameters
range_resolution = float(config.get('exp_param','range_resolution'))
time_resolution = float(config.get('exp_param','time_resolution'))
pulse_length = float(config.get('exp_param','pulse_length'))
sampling_rate = float(config.get('exp_param','sampling_rate'))
interpulse_period = float(config.get('exp_param','interpulse_period'))
radar_freq = float(config.get('exp_param','radar_freq'))

#Image Processing Parameters
goal_SNR = float(config.get('image_param','SNR'))
regularization = float(config.get('image_param','regularization'))

#-----
#-----
#Data read finished
#-----
#-----

output_folder_name=output_data_location

time1=time.time()

#-----
#-----
#Radar Experiment Params
#-----
#-----

sample_rate = sampling_rate #in Hz - 93 kHz (tx baud rate)
sample_time = pulse_length #in seconds (pulse length)

inter_pulse_period = interpulse_period #in seconds (time between pulses, from beginning of pulse)

freq_points = int(sample_rate*sample_time) # number of points in the lag-profile / number of frequency bins

tau_value = 1/sample_rate # baud length

exp_integration_time = time_resolution #in seconds (how long of time is integrated for)

num_pulses_integrated = int(exp_integration_time / inter_pulse_period) #number of pulses per integration time

exp_range_resolution = range_resolution # in km

RX_noise_temperature = RX_noise #Kelvin

RX_noise_power = RX_noise_temperature * sample_rate * spconst.k # T * BW * kb for noise power

TX_power = radar_power #in MW

TX_beamwidth = TX_beamwidth #degrees FWHM with Gaussian Beam assumption

#-----
#-----
#End of Experiment Parameters
#-----
#-----

#------------------
#------------------
#------------------

if __name__== '__main__':

    #----
    #----
    #SYNTHETIC IONOSPHERE DATA SETUP
    #----
    #----

    #distance (or altitude) in m
    distance = altitude

    #number of points in x and y directions
    x_res = x_num
    y_res = y_num

    spatial_res = spatial_res #in meters

    altitude = altitude # in meters

    beam_plot = np.asarray([np.sin(np.arange(201)/200*2*np.pi),np.cos(np.arange(201)/200*2*np.pi)])*np.sin(TX_beamwidth*np.pi/180/2.0)*distance

    #read in test file for data from GEMINI model
    df = h5py.File(f'{input_model_file}','r')

    ne = df[f'{input_data_field_ne}'][...]
    te = df[f'{input_data_field_te}'][...]
    ti = df[f'{input_data_field_ti}'][...]
    vi = df[f'{input_data_field_vi}'][...]
    df.close()

    fig,ax=plt.subplots(2,2)
    ne_c=ax[0,0].imshow(ne,aspect='auto',interpolation='none',origin='lower',extent=[-x_res/2*spatial_res,x_res/2*spatial_res,-y_res/2*spatial_res,y_res/2*spatial_res])
    ax[0,0].plot(beam_plot[0,:],beam_plot[1,:],c='k',lw=4,label='Approx. EISCAT_3D FWHM Beam')
    ax[0,0].set_title('Ne')
    ax[0,0].set_xlabel('x (meters)')
    ax[0,0].set_ylabel('y (meters)')
    ax[0,0].legend(loc='upper left')
    te_c=ax[0,1].imshow(te,aspect='auto',interpolation='none',origin='lower',extent=[-x_res/2*spatial_res,x_res/2*spatial_res,-y_res/2*spatial_res,y_res/2*spatial_res])
    ax[0,1].plot(beam_plot[0,:],beam_plot[1,:],c='k',lw=4)
    ax[0,1].set_title('Te')
    ax[0,1].set_xlabel('x (meters)')
    ax[0,1].set_ylabel('y (meters)')
    ti_c=ax[1,0].imshow(ti,aspect='auto',interpolation='none',origin='lower',extent=[-x_res/2*spatial_res,x_res/2*spatial_res,-y_res/2*spatial_res,y_res/2*spatial_res])
    ax[1,0].plot(beam_plot[0,:],beam_plot[1,:],c='k',lw=4)
    ax[1,0].set_title('Ti')
    ax[1,0].set_xlabel('x (meters)')
    ax[1,0].set_ylabel('y (meters)')
    vi_c=ax[1,1].imshow(vi,aspect='auto',interpolation='none',origin='lower',extent=[-x_res/2*spatial_res,x_res/2*spatial_res,-y_res/2*spatial_res,y_res/2*spatial_res])
    ax[1,1].plot(beam_plot[0,:],beam_plot[1,:],c='k',lw=4)
    ax[1,1].set_title('Vi')
    ax[1,1].set_xlabel('x (meters)')
    ax[1,1].set_ylabel('y (meters)')
    plt.colorbar(ne_c,ax=ax[0,0],label=u'Plasma Density ($m^{-3}$)')
    plt.colorbar(te_c,ax=ax[0,1],label='Electron Temperature (K)')
    plt.colorbar(ti_c,ax=ax[1,0],label='Ion Temperature (K)')
    plt.colorbar(vi_c,ax=ax[1,1],label='Ion Velocity (m/s)')
    plt.tight_layout()
    plt.savefig(f'{output_data_location}/ionosphere_initial_plasma_params.png')
    plt.close()

    Ne_array = ne
    te_array = te
    ti_array = ti
    vi_array = vi

    print(Ne_array.shape)

    #----
    #----
    #END Synthetic ionosphere data setup
    #----
    #----

    #---------
    #--------- Here are the sampling points for the real EISCAT 3D Array
    #---------

    radar_freq = radar_freq
    speed_of_light = spconst.c

    #Determine location of EISCAT 3D antennas in 2D plane
    N=109+10 #number of panels (actually 109 - 3 extra per side are added to the E3D design of a 6 size ring (91+18), and should be added to this software at some point) - plus the outrigger locations should be added

    panel_distance = 7.07 #meters

    ant_x = np.zeros(N,dtype=np.float32)
    ant_y = np.zeros(N,dtype=np.float32)

    panel_num=11

    #double check these values at some point
    xvalues_outrigger=[-249.46,-543.21,-4.76,-113.11,-327.45,151.17,408.7,707.66,240.5,30.49]
    yvalues_outrigger=[165.38,291.98,173.29,324.07,531.9,103.19,153.01,127.15,362.64,734.76]

    for x in range(10):
        ant_x[x+1]=xvalues_outrigger[x]
        ant_y[x+1]=yvalues_outrigger[x]

    #create a hexagonal array of panels
    for side_num in range(6):
        for ring_num in range(1,6):
            for ring_panel in range(ring_num):
                ant_x[panel_num] = ring_num*panel_distance*np.sin((2*np.pi*((side_num)*60))/360)+(panel_distance*(ring_panel)*np.sin((2*np.pi*((side_num+2)*60))/360))
                ant_y[panel_num] = ring_num*panel_distance*np.cos((2*np.pi*((side_num)*60))/360)+(panel_distance*(ring_panel)*np.cos((2*np.pi*((side_num+2)*60))/360))
                panel_num+=1

    for side_num in range(6):
        for ring_panel in range(2,5):
                ant_x[panel_num] = 6*panel_distance*np.sin((2*np.pi*((side_num)*60))/360)+(panel_distance*(ring_panel)*np.sin((2*np.pi*((side_num+2)*60))/360))
                ant_y[panel_num] = 6*panel_distance*np.cos((2*np.pi*((side_num)*60))/360)+(panel_distance*(ring_panel)*np.cos((2*np.pi*((side_num+2)*60))/360))
                panel_num+=1
  
    #---------
    #--------- End of sampling points for the real EISCAT 3D Array
    #---------

    #create grid of points to evaluate (x,y space), require distance from radar as input

    TX_beam_angular_width = TX_beamwidth # FWHM estimate in degrees
    
    #how many meters per point
    meter_res = spatial_res

    #fwhm = 2.355 \sigma
    gauss_sigma = np.sin(TX_beam_angular_width*np.pi/180/2)*2*distance/2.355

    x_values = (np.arange(x_res)-(x_res/2))*meter_res
    y_values = (np.arange(y_res)-(y_res/2))*meter_res

    xx,yy = np.meshgrid(x_values,y_values)

    xx = xx.T
    yy = yy.T

    print(xx.shape)
    print(yy.shape)

    xx = xx.reshape(x_res*y_res)
    yy= yy.reshape(x_res*y_res)
    ti_array = ti_array.reshape(x_res*y_res)
    te_array = te_array.reshape(x_res*y_res)
    Ne_array = Ne_array.reshape(x_res*y_res)
    vi_array = vi_array.reshape(x_res*y_res)

    #At this point we have a 1-D array of the plasma parameters for the ionosphere 2D plane
    
    brightness_array = np.zeros((x_res,y_res,freq_points),dtype=np.complex64)

    brightness_array = brightness_array.reshape(x_res*y_res,freq_points)
    
    #----
    #----
    #HERE IS WHERE THE ISR SPECTRA ARE GENERATED FROM THE PLASMA PARAMETERS
    #----
    #----
    
    #calculate the spectra for each x,y point
    #this should be able to be parallelized realtively easily
    for x in range(len(xx)):
        ISS2 = Specinit(centerFrequency = radar_freq, bMag = 0.4e-4, nspec=freq_points, sampfreq=sample_rate,dFlag=False)

        ti = ti_array[x]
        te = te_array[x]
        Ne = Ne_array[x]
        if altitude>149000:
            mi = 16
        else:
            mi = 31
        Ce = np.sqrt(spconst.k*te/spconst.m_e)
        Ci = np.sqrt(spconst.k*ti/(spconst.m_p*mi))
        vi = vi_array[x]

        datablock = np.array([[Ne,ti,vi,1,mi,0],[Ne,te,0,1,1,0]])
        datablockn = datablock.copy()
        (omega,brightness_array[x,:]) = ISS2.getspec(datablock.copy())
    
        beam_offset_x=0
        beam_offset_y=0
    
        brightness_array[x,:] = brightness_array[x,:]*np.exp(-((xx[x]+beam_offset_x)**2+(yy[x]+beam_offset_y)**2)/(2*gauss_sigma**2))
    
    # axis in frequency
    omega_axis = omega
    
    brightness_array_plot_testing = brightness_array.reshape(x_res,y_res,freq_points)
    
    #----
    #----
    #WE NOW HAVE ISR SPECTRA FOR EACH POINT IN THE 2D PLANE
    #----
    #----
    
    #this is the array of B(l,m,f) that needs to be converted to I(u,v,tau)
    brightness_array = brightness_array.reshape(x_res*y_res,freq_points)
    
    xx_new=xx
    yy_new=yy
    
    #----
    #----
    #HERE WE CALCULATE THE NEAR FIELD DISTANCES FOR THE ARRAY AND THE UNIQUE BASELINES
    #----
    #----
    
    total_tx_rx_dist = np.zeros((len(xx),len(ant_x)),dtype=np.float32)
    
    #create array of propagation distance for each xx,yy pair to each antenna
    for num in range(len(ant_x)):
        total_tx_rx_dist[:,num] = np.sqrt(xx**2+yy**2+distance**2)+np.sqrt((xx-ant_x[num])**2+(yy-ant_y[num])**2+distance**2)
    
    #print(total_tx_rx_dist.shape)
    
    num_baselines = N*(N-1)/2

    uv_baselines = np.zeros((2,int(2*num_baselines+1)),dtype=np.float32)
    uv_baselines_dist = np.zeros((len(xx),int(2*num_baselines+1)),dtype=np.float32)

    baseline_num = 1

    #calculate the distance difference for each baseline
    for panel_num1 in range(N):
        for panel_num2 in range(panel_num1+1,N):
            uv_baselines[0,baseline_num] = ant_x[panel_num2]-ant_x[panel_num1]
            uv_baselines[1,baseline_num] = ant_y[panel_num2]-ant_y[panel_num1]
            uv_baselines_dist[:,baseline_num] = total_tx_rx_dist[:,panel_num2]-total_tx_rx_dist[:,panel_num1]
            uv_baselines[0,baseline_num+1] = ant_x[panel_num1]-ant_x[panel_num2]
            uv_baselines[1,baseline_num+1] = ant_y[panel_num1]-ant_y[panel_num2]
            uv_baselines_dist[:,baseline_num+1] = total_tx_rx_dist[:,panel_num1]-total_tx_rx_dist[:,panel_num2]
            
            baseline_num+=2
    
    uv_baselines[0,:] = np.round(uv_baselines[0,:],decimals=4)
    uv_baselines[1,:] = np.round(uv_baselines[1,:],decimals=4)
    
    uv_baselines_dist = uv_baselines_dist[:,np.lexsort((uv_baselines[0,:], uv_baselines[1,:]))]
    uv_baselines = uv_baselines[:,np.lexsort((uv_baselines[0,:], uv_baselines[1,:]))]
    
    #find unique baselines
    unique_baselines = []
    unique_baselines_dist = []
    unique_baseline_repeat = []
    std_deviation_baselines_num = []
    std_deviation_baselines = []

    baseline_num_t = 0
    for baseline_num in range(len(uv_baselines[0,:])):
        tmp_array=[]
        unique_baseline_repeat_num = 1
        unique_baselines.append([uv_baselines[0,baseline_num_t],uv_baselines[1,baseline_num_t]])
        unique_baselines_dist.append([uv_baselines_dist[:,baseline_num_t]])
        if uv_baselines[0,baseline_num_t]==0.0 and uv_baselines[1,baseline_num_t]==0.0:
            unique_baseline_repeat_num=N
        
        baseline_num_t+=1
        if baseline_num_t>(len(uv_baselines[0,:])-1):
            break
        tmp_array.append(uv_baselines_dist[:,baseline_num_t-1])
        while np.isclose(uv_baselines[0,baseline_num_t],uv_baselines[0,baseline_num_t-1],atol=1e-2) and np.isclose(uv_baselines[1,baseline_num_t],uv_baselines[1,baseline_num_t-1],atol=1e-2):
            unique_baseline_repeat_num+=1
            baseline_num_t+=1
            tmp_array.append(uv_baselines_dist[:,baseline_num_t-1])
        unique_baseline_repeat.append(unique_baseline_repeat_num)
        if unique_baseline_repeat_num>1:
            std_deviation_baselines_num.append(unique_baseline_repeat_num)
            std_deviation_baselines.append(np.mean(np.std(tmp_array,axis=0)))

    unique_baseline_repeat.append(unique_baseline_repeat_num)
    unique_baseline_repeat = np.asarray(unique_baseline_repeat)
    
    unique_baselines = np.asarray(unique_baselines).T
    unique_baselines_dist = np.asarray(unique_baselines_dist).T.reshape(len(xx),len(unique_baselines[0,:]))
    
    #----
    #----
    #NEAR FIELD DISTANCES AND UNIQUE BASELINES CALCULATED
    #----
    #----
    
    #----
    #----
    #CALCULATE THE IDEAL MEASUREMENTS AT THE BASELINES (VISIBILITIES)
    #----
    #----
    
    #Calculating Visibilities:
    #Visibilities = sum over xx,yy (B*np.exp(j*2pi*unique_baselines_dist/lambda)*del(x)*del(y))
    
    time2=time.time()
    
    #Calculating Original Brightness:
    #Brightness = sum over u,v (V*np.exp(j*2pi*unique_baselines_dist/lambda))
    
    visibilities = np.zeros((len(unique_baselines[0,:]),len(brightness_array[0,:])),dtype=np.complex64)
    
    for freq_num in range(len(brightness_array[0,:])):
        print(freq_num)
        visibilities[:,freq_num]=np.einsum('a,ab->b',brightness_array[:,freq_num],np.exp(2.0j*np.pi*unique_baselines_dist/(speed_of_light/radar_freq)))
    
    print(visibilities.shape)
    
    acf_points = int(freq_points/2)+1
    
    print(acf_points)
    
    visibilities_orig = np.fft.fft(visibilities,axis=1)[:,:acf_points]
    
    print('Visibility Generation: ', time.time()-time2)
    
    #----
    #----
    #WE NOW HAVE THE EXPECTED IDEALIZED MEASUREMENTS WITHOUT NOISE
    #----
    #----
    
    SNR_value = np.power(10.0,goal_SNR/10.0)*119
    
    alpha_value = regularization
    
    error_values = np.zeros((x_res,y_res,freq_points),dtype=np.complex64)
    
    new_brightness_values = np.zeros((x_res,y_res,freq_points),dtype=np.complex64)
        
    SNR = SNR_value
    
    #----
    #----
    #HERE IS WHERE NOISE IS ADDED TO THE IDEAL SIGNAL
    #noise is scaled by u=0,v=0,tau=0 lag
    #no scaling is incorporated into the number of averages that go into a lag/tau
    #----
    #----
    
    noise=(1/np.sqrt(2)*np.random.normal(size=len(visibilities_orig[:,0])*len(visibilities_orig[0,:]),scale=np.amax(np.abs(visibilities_orig[:,0])))+1.0j/np.sqrt(2)*np.random.normal(size=len(visibilities_orig[:,0])*len(visibilities_orig[0,:]),scale=np.amax(np.abs(visibilities_orig[:,0]))))*(1/SNR)*((N)/(np.tile(unique_baseline_repeat,acf_points)))
    
    print(np.amax(np.abs(visibilities_orig[:,0])))
    print(np.abs(np.std((noise))))
    print(np.median(np.abs(noise)))
    
    print(np.amax(np.abs(visibilities_orig[:,0]))/np.abs(np.std((noise))))
    print(np.amax(np.abs(visibilities_orig[:,0]))/np.median(np.abs(noise)))
    
    noise=noise.reshape(len(visibilities_orig[:,0]),acf_points)
    
    visibilities = visibilities_orig + noise
    
    #----
    #----
    #NOISE IS ADDED - THESE ARE EXPECTED MEASUREMENTS FROM EISCAT_3D AT THIS POINT
    #----
    #----
    
    #----
    #----
    #NOW WE USE IMAGING TECHNIQUES ON ARRAY u,v,tau DATA
    #----
    #----
    
    #implement SVD
    #Create Visibility->brightness array
    
    time3=time.time()
    
    uv_xy_matrix = (np.exp(-2.0j*np.pi*unique_baselines_dist/(speed_of_light/radar_freq))).T#*3*np.sqrt(3)/2*7.07/(speed_of_light/radar_freq)*7.07/(speed_of_light/radar_freq)).T
    
    u,s,vh = np.linalg.svd(uv_xy_matrix,full_matrices=True)
    
    print(u.shape,s.shape,vh.shape)
        
    alpha=alpha_value
    
    lambda_s=s
    
    s_regularization = lambda_s**2/(lambda_s**2+alpha**2)/lambda_s
    
    s_mat = np.zeros((len(unique_baselines_dist[:,0]),len(unique_baselines_dist[0,:])))
    s_mat[:len(unique_baselines_dist[0,:]),:len(unique_baselines_dist[0,:])] = np.diag(s_regularization)
    
    uv_xy_svd = np.matmul(np.matmul(vh.T,s_mat),u.T)
        
    print('SVD Matrix Creation: ', time.time()-time3)
        
    time4=time.time()
        
    imaged_brightness = np.zeros((len(xx),len(visibilities[0,:])),dtype=np.complex64)
        
    print(imaged_brightness.shape)
        
    for num_tau in range(len(visibilities[0,:])):
        print(num_tau)
        imaged_brightness[:,num_tau] = np.einsum('a,ba->b',visibilities[:,num_tau],uv_xy_svd)
        
    print(imaged_brightness.shape)
        
    print('Brightness Re-creation: ', time.time()-time4)
        
    imaged_brightness = np.fft.ifft(np.concatenate((imaged_brightness,np.conj(np.flip(imaged_brightness[:,1:],axis=1))),axis=1),axis=1)
    
    new_brightness_values[:,:,:] = imaged_brightness.reshape(x_res,y_res,freq_points)
    
    print('Total Analysis Time: ', time.time()-time1)
    
    mpl.rcParams['figure.figsize'] = [15.0, 24.0]
    
    fig,ax=plt.subplots(2)
    orig_imag=ax[0].imshow(np.abs(np.fft.fft(brightness_array_plot_testing,axis=2)[:,:,0]),origin='lower',aspect='auto',interpolation='none',extent=[-x_res/2*spatial_res,x_res/2*spatial_res,-y_res/2*spatial_res,y_res/2*spatial_res])
    ax[0].set_title('Original Image')
    ax[0].set_xlabel('x (meters)')
    ax[0].set_ylabel('y (meters)')
    plt.colorbar(orig_imag,ax=ax[0])
    svd_imag=ax[1].imshow(np.abs(np.fft.fft(imaged_brightness.reshape(x_res,y_res,freq_points),axis=2)[:,:,0]),origin='lower',aspect='auto',interpolation='none',extent=[-x_res/2*spatial_res,x_res/2*spatial_res,-y_res/2*spatial_res,y_res/2*spatial_res])
    ax[1].set_title('SVD Imaged')
    ax[1].set_xlabel('x (meters)')
    ax[1].set_ylabel('y (meters)')
    plt.colorbar(svd_imag,ax=ax[1])
    plt.tight_layout()
    plt.savefig(f'{output_data_location}/brightness_compare.png')    
    plt.close()
    
    mpl.rcParams['figure.figsize'] = [30.0, 24.0]
    
    #--------
    #--------
    # Estimate of Plasma Parameter Errors
    #--------
    #--------
    
    tau0 = pulse_length/2.0 # acf time scale in us
    dtau = tau_value
    ntau = tau0/dtau

    # parameter errors for gamma=0.01
    ParErr0 = np.array([0.035 , 0.053 , 0.072 , 8.110])
    gamma0 = 0.01

    speco = brightness_array_plot_testing
    speci = new_brightness_values

    noiselevel  = np.empty((x_num,y_num))
    ParErrs = np.empty((x_num,y_num,4))

    for k in range(x_num):
        for l in range(y_num):
            speco2 = np.concatenate( ( speco.real[k,l,12:] , speco.real[k,l,:12] ) )
            acfo = np.fft.ifft(speco2)
            speci2 = np.concatenate( ( speci[k,l,12:] , speci[k,l,:12] ) )
            acfi = np.fft.ifft(speci2)
            acfstd = np.std((acfi.real-acfo.real))
            noiselevel[k,l] = acfstd/acfo.real[0]/np.sqrt(ntau)
            ParErrs[k,l,:] = noiselevel[k,l]/gamma0 * ParErr0

    x_bins = (np.arange(x_num)-(x_num/2)+0.5)*spatial_res
    y_bins = (np.arange(y_num)-(y_num/2)+0.5)*spatial_res

    xx,yy=np.meshgrid(x_bins,y_bins)

    figPar = plt.figure(figsize=(30,24))
    axNe  = figPar.add_subplot(2,2,1)
    imNe=axNe.pcolor(xx,yy,ParErrs[:,:,0],clim=[0,1.0])
    axNe.set_xlabel('x (meters)')
    axNe.set_ylabel('y (meters)')
    plt.colorbar(imNe)
    plt.title('dNe/Ne')

    axTi  = figPar.add_subplot(2,2,2)
    imTi=axTi.pcolor(xx,yy,ParErrs[:,:,1],clim=[0,1.0])
    axTi.set_xlabel('x (meters)')
    axTi.set_ylabel('y (meters)')
    plt.colorbar(imTi)
    plt.title('dTi/Ti')

    axTr  = figPar.add_subplot(2,2,3)
    imTr=axTr.pcolor(xx,yy,ParErrs[:,:,2],clim=[0,1.0])
    axTr.set_xlabel('x (meters)')
    axTr.set_ylabel('y (meters)')
    plt.colorbar(imTr)
    plt.title('dTr/Tr')

    axVi  = figPar.add_subplot(2,2,4)
    imVi=axVi.pcolor(xx,yy,ParErrs[:,:,3],clim=[0,300])
    axVi.set_xlabel('x (meters)')
    axVi.set_ylabel('y (meters)')
    plt.colorbar(imVi)
    plt.title('dVi (m/s)')

    plt.tight_layout()
    plt.savefig(f'{output_data_location}/PlasmaParameterErrors.png')
    plt.close()
    
    #------
    #------
    # End of Estimated Plasma Errors Section
    #------
    #------
    
    #------
    #------
    # Determine Range and Time resolutions required to achieve these imaging results
    #------
    #------
    
    ne_avg = np.mean(Ne_array)
    te_avg = np.mean(te_array)
    ti_avg = np.mean(ti_array)
    r_e = 2.8179403262e-15
    
    variable_range_resolution = (np.arange(999)+1.0)*100
    
    #isr cross-section per meter cubed
    ne_sigma = ne_avg*4*np.pi*r_e**2*(1/(1+(te_avg/ti_avg)))
    
    #total electron volume scattering cross section
    ne_sigma_total = ne_sigma*variable_range_resolution*np.pi*(np.sin(TX_beamwidth*np.pi/180/2.0)*altitude)**2
    
    TX_efficiency = 0.5
    TX_gain = 16/(np.sin(TX_beamwidth*np.pi/180)*np.sin(TX_beamwidth*np.pi/180))*TX_efficiency
    
    #calculate the radar power budget
    radar_power_total = TX_power*TX_gain*np.power(10,23/10)*((speed_of_light/radar_freq)**2)*ne_sigma_total / (((4*np.pi)**3)*(altitude**4))
    
    #calculate the signal to noise ratio
    calculated_snr = radar_power_total/RX_noise_power
    
    plt.plot(variable_range_resolution/1000,10*np.log10(calculated_snr))
    plt.xscale('log')
    plt.xlabel('Range Resolution (km)')
    plt.ylabel('Signal to Noise Ratio (dB)')
    plt.savefig(f'{output_data_location}/snr_vs_range_res.png')
    plt.close()
    
    #calculate the number of measurement samples required for the signal to standard deviation ratio used in the imaging
    number_of_samples = ((radar_power_total+RX_noise_power)/((1/np.power(10.0,goal_SNR/10))*radar_power_total))**2
    
    #calculate the amount of integration time required for the snr request by the user
    variable_time_resolution = interpulse_period*number_of_samples/(pulse_length*sampling_rate)
    
    plt.plot(variable_range_resolution/1000,variable_time_resolution)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.title(f'Resolution Required for {goal_SNR:2.1f} dB signal to noise standard deviation ratio')
    plt.xlabel('Range Resolution (km)')
    plt.ylabel('Time Integration (s)')
    plt.savefig(f'{output_data_location}/range_res_vs_time_res.png')
    plt.close()
    
    #-----
    #-----
    # Output the Data array
    #-----
    #-----
    
    write_file  = h5py.File(f'{output_data_location}/{output_data_name}','w')
    write_file.create_dataset('original_spectra',data=brightness_array_plot_testing,dtype=np.complex64)
    write_file.create_dataset('imaged_spectra',data=new_brightness_values,dtype=np.complex64)
    write_file.create_dataset('imaged_ne_errors_ratio',data=ParErrs[:,:,0])
    write_file.create_dataset('imaged_ti_errors_ratio',data=ParErrs[:,:,1])
    write_file.create_dataset('imaged_tr_errors_ratio',data=ParErrs[:,:,2])
    write_file.create_dataset('imaged_vi_errors',data=ParErrs[:,:,3])
    write_file.close()
    
