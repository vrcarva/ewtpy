Empirical Wavelet Transform Python package

Original paper: 
Gilles, J., 2013. Empirical Wavelet Transform. IEEE Transactions on Signal Processing, 61(16), pp.3999–4010. 
Available at: http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6522142.
Original Matlab toolbox: https://www.mathworks.com/matlabcentral/fileexchange/42141-empirical-wavelet-transforms
 
@author: Vinícius Rezende Carvalho  
Programa de pós graduação em engenharia elétrica - PPGEE UFMG  
Universidade Federal de Minas Gerais - Belo Horizonte, Brazil  
Núcleo de Neurociências - NNC   

ewtpy performs the Empirical Wavelet Transform of a 1D signal over N scales. Main function is EWT1D:

ewt,  mfb ,boundaries = EWT1D(f, N = 5, log = 0,detect = "locmax", completion = 0, reg = 'average', lengthFilter = 10,sigmaFilter = 5)  
Other functions include:  
EWT_Boundaries_Detect  
EWT_Boundaries_Completion  
EWT_Meyer_FilterBank  
EWT_beta  
EWT_Meyer_Wavelet  
LocalMax  
LocalMaxMin  

Some functionalities from J.Gilles' MATLAB toolbox have not been implemented, such as EWT of 2D inputs, preprocessing, adaptive/ScaleSpace boundaries_detect.

The Example folder contains test signals and scripts

Any questions, comments, suggestions and/or corrections, please get in contact with vrcarva@ufmg.br


#%% Example script  
import numpy as np  
import matplotlib.pyplot as plt  
import ewtpy  

T = 1000  
t = np.arange(1,T+1)/T  
f = np.cos(2*np.pi*0.8*t) + 2*np.cos(2*np.pi*10*t)+0.8*np.cos(2*np.pi*100*t)  
ewt,  mfb ,boundaries = ewtpy.EWT1D(f, N = 3)  
plt.plot(f)  
plt.plot(ewt)  








