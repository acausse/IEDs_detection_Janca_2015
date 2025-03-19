# General import
import numpy as np
from scipy import signal as sig

# For get_IEDs
from scipy.stats import lognorm
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, cheby2, lfilter, iirnotch

def get_smooth(signal, N, sigma, kernel_type = 'gaussian', return_kernel = False):
    """
    Smooths a given signal using either a Gaussian or a rectangular (boxcar) kernel.
    
    Parameters:
    ----------
    signal : array-like
        The input signal to be smoothed.
    N : int
        The size of the smoothing kernel.
    sigma : float
        The standard deviation for the Gaussian kernel (not used for rectangular).
    kernel_type : str, optional
        The type of kernel to use for smoothing ('gaussian' or 'rectangular'). Default is 'gaussian'.
    return_kernel : bool, optional
        If True, returns both the smoothed signal and the kernel. Default is False.

    Returns:
    -------
    signal_smoothened : array-like
        The smoothed version of the input signal.
    kernel : array-like (only if return_kernel=True)
        The applied smoothing kernel.

    Notes:
    ------
    - Gaussian kernel is created using `sig.windows.gaussian(N, sigma)`.
    - Rectangular (boxcar) kernel is created using `sig.windows.boxcar(N)`.
    - The signal is smoothed via convolution with the kernel.
    - The resulting signal is adjusted to remove edge artifacts caused by convolution.
    """
    if kernel_type == 'gaussian':
        kernel = sig.windows.gaussian(N,sigma)
    elif kernel_type == 'rectangular':
        kernel = sig.windows.boxcar(N)
    
    signal_smoothened = np.convolve(signal, kernel)
    signal_smoothened = signal_smoothened[int(N/2):-(int(N/2))+1]
    
    if return_kernel:
        return signal_smoothened, kernel
    else:
        return signal_smoothened
      
  def get_timeWinsIntersect(itwA, itwB, lastPoint, firstPoint=0, verbose=False, lfp=None):
    """
    Computes the intersection of two sets of time windows and returns the non-overlapping portions.

    Parameters:
    ----------
    itwA : np.ndarray
        First set of time windows, shape (n, 2).
    itwB : np.ndarray
        Second set of time windows, shape (m, 2).
    lastPoint : int
        The last valid time point in the signal.
    firstPoint : int, optional
        The first valid time point in the signal (default is 0).
    ifDisplay : bool, optional
        If True, visualizes the overlapping windows and prints diagnostic messages (default is False).
    lfp : array-like, optional
        The signal data to be used for visualization (default is None).

    Returns:
    -------
    itwTot : np.ndarray
        The final set of non-overlapping time windows.
    
    Notes:
    ------
    - This function merges overlapping time windows.
    - If `ifDisplay=True`, it compares the intersection with the original windows and provides feedback.
    - Ensures the first time window starts at or after `firstPoint` to prevent invalid indices.
    """
    skull=np.ones(lastPoint, bool)
    
    if np.logical_or(itwA.shape[0]==0, itwA.shape[1]==0):
        itwTot=itwB
    elif np.logical_or(itwB.shape[0]==0, itwB.shape[1]==0):
        itwTot=itwA
    else:
        for itw in itwA:
            skull[itw[0]:itw[1]]=False
        for itw in itwB:
            skull[itw[0]:itw[1]]=False

        edgeInds=np.where(np.diff(np.where(skull)[0])>1)[0]
        trues=np.where(skull)[0]

        itwTot=[[trues[edgeInds][i]+1, trues[edgeInds+1][i]] for i in range(edgeInds.shape[0])]

        if skull[0]==False: # starts by False (not detected window)
            itwTot.insert(0, [firstPoint, trues[0]])

        if skull[-1]==False:
            itwTot.append([trues[-1], lastPoint])

        itwTot=np.array(itwTot)

        if verbose:
            lfpTpItwA=get_timeWinsTemplatedSignal(lfp, itwA)
            lfpTpItwB=get_timeWinsTemplatedSignal(lfp, itwB)
            lfpTpItwTot=get_timeWinsTemplatedSignal(lfp, itwTot)

            if np.sum(np.isnan(lfpTpItwA)==False)+np.sum(np.isnan(lfpTpItwB)==False) == np.sum(np.isnan(lfpTpItwTot)==False):
                print('NON-overlapping windows')
            else:
                print('Overlapping windows')

        if itwTot[0][0]>itwTot[0][1]:
            print('WARNING: firstPoint is greater than the first time window. Consider editing firstPoint.')
    return itwTot
    
def get_timeWins4TimeVect(timeVect, dtype = int):
    """
    Converts a time vector into a series of consecutive time windows.

    Parameters:
    ----------
    timeVect : np.ndarray
        A 1D array representing time points.
    dtype : type, optional
        The data type of the output array (default is int).

    Returns:
    -------
    timeWins : np.ndarray
        A (N-1, 2) array where each row represents a time window 
        defined by consecutive time points from `timeVect`.

    Notes:
    ------
    - The function constructs pairs of consecutive time points.
    - The output array has one fewer row than `timeVect` since it forms
      time intervals between adjacent elements.
    """
    timeWins = np.zeros((timeVect.shape[0], 2), dtype = dtype)
    for ind in range(timeVect.shape[0]-1):
        timeWins[ind, :] = [timeVect[0:-1][ind], timeVect[1:][ind]]
    timeWins = timeWins[0:-1]
    return timeWins


def get_timeWinsOverlapping(START, END, win_size, step):
    """
    Generates overlapping time windows within a specified range.

    Parameters:
    ----------
    START : int
        The starting point of the time range.
    END : int
        The ending point of the time range.
    win_size : int
        The size (duration) of each time window.
    step : int
        The step size between the start of consecutive windows.

    Returns:
    -------
    wins : np.ndarray
        A (N, 2) array where each row represents a time window 
        with [start, end] indices.

    Notes:
    ------
    - The function creates overlapping windows by shifting the start position by `step`.
    - Only windows that fully fit within the range (`END`) are retained.
    """
    starts=np.arange(START, END+step, step)
    ends=starts+win_size
    inds=np.logical_and(starts<END, ends<=END)
    wins=np.column_stack([starts, ends])[inds]
    return wins

def get_timeWinsTemplatedSignal(signal, timeWins, nTimePoints = None, filling = np.nan):
    """
    Extracts and templates signal segments based on provided time windows.

    Parameters:
    ----------
    signal : np.ndarray
        A 1D array representing the input signal.
    timeWins : np.ndarray
        A (N, 2) array where each row defines a time window with [start, end] indices.
    nTimePoints : int, optional
        The total number of time points in the output (default is the length of `signal`).
    filling : scalar, optional
        The value used to fill time points outside the defined windows (default is np.nan).

    Returns:
    -------
    templatedSignal : np.ndarray
        A 1D array of length `nTimePoints` where only the segments within `timeWins`
        are retained, and all other values are set to `filling`.

    Notes:
    ------
    - If `signal` contains nested lists of length 1, use `np.ravel(signal)` before passing it.
    - This function extracts portions of the signal based on `timeWins` and replaces
      undefined regions with the specified `filling` value.
    """
    if nTimePoints is None:
        nTimePoints = signal.shape[0]
    templatedSignal = np.full(nTimePoints, filling)
    for timeWin in timeWins:
        templatedSignal[timeWin[0]:timeWin[1]] = signal[timeWin[0]:timeWin[1]]
    return templatedSignal
  

def get_IEDs(trace, sr, k=3, extend_ms=120, win_s=5, step_s=1, verbose=True, verbose_win_mult=100):
    """
    Detects interictal epileptiform discharges (IEDs) from an intracranial EEG (iEEG) signal.

    Parameters:
    ----------
    trace : np.ndarray
        The 1D array representing the iEEG signal.
    sr : int
        The sampling rate of the signal (Hz).
    k : float, optional
        The detection threshold multiplier (default is 3).
    extend_ms : int, optional
        Time (in milliseconds) to extend detected events for merging (default is 120 ms).
    win_s : float, optional
        Window size (in seconds) for computing thresholds (default is 5 s).
    step_s : float, optional
        Step size (in seconds) for sliding window analysis (default is 1 s).
    verbose : bool, optional
        If True, prints progress updates (default is True).
    verbose_win_mult : int, optional
        Prints progress every X windows computed (default is 100).

    Returns:
    -------
    badWins : np.ndarray
        A (N, 2) array containing the detected time windows with high activity.
    IEDs : np.ndarray
        An array of time indices where IEDs are detected.

    Notes:
    ------
    - Uses a log-normal distribution to model signal envelope thresholds.
    - Applies bandpass and notch filtering before envelope extraction.
    - Extends detected IEDs to merge nearby events.
    - Recommended to downsample signal to 200 Hz for faster computation.
    - If downsampled to 200Hz, it runs on ~1h of data in 5 mins
    - Applies IED detection as in:
        Detection of interictal epileptiform discharges using signal envelope distribution modelling: application to epileptic and non-epileptic intracranial recordings
        Radek Janca, Petr Jezdik, Roman Cmejla, Martin Tomasek, Gregory A Worrell, Matt Stead, Joost Wagenaar, John G R Jefferys, Pavel Krsek, Vladimir Komarek, Premysl Jiruska, Petr Marusic
         Brain Topogr. 2015
    """    
    def apply_zero_phase_filter(data, sr):
        # Filter parameters
        highpass_freq = 10  # Hz
        highpass_order = 8
        lowpass_freq = 60  # Hz
        lowpass_order = 8
        notch_freq = 50  # or 60.0 Hz
        Q = 30.0  #

        # Apply filters
        hp_b, hp_a = cheby2(highpass_order, 30, highpass_freq / (sr / 2), btype='highpass')
        lp_b, lp_a = cheby2(lowpass_order, 30, lowpass_freq / (sr / 2), btype='lowpass')
        filtered_data = lfilter(hp_b, hp_a, data)
        filtered_data = lfilter(lp_b, lp_a, filtered_data)
        b, a = iirnotch(notch_freq, Q, sr)
        filt_data = lfilter(b, a, filtered_data)
        return filtered_data
    
    if verbose:
        print('Extracting envelope        ')
    trace_filt=apply_zero_phase_filter(trace, sr)
    analytic_signal = hilbert(trace_filt)
    trace_env = np.abs(analytic_signal)

    win_size=int(win_s*sr)
    step=int(step_s*sr)
    MAX=len(trace_env)

    wins=get_timeWinsOverlapping(0, MAX, win_size, step)

    if verbose:
        print('Lognorm modelling - computing thresholds        ')
    # Model lognorm distribution and find threshold value
    wThresh=np.zeros(len(wins))
    for wini, win in enumerate(wins):
        if verbose:
            if (wini+1)%verbose_win_mult==0:
                print('', wini+1, '/', len(wins))
        data=trace_env[win[0]:win[1]]

        # Fit a lognorm distribution
        mu=np.mean(np.log(data))
        initScale=np.exp(mu)
        sigma, loc, scale = lognorm.fit(data, method='MLE', scale=initScale)

        # Get the mode and median from the probability density function
        counts,ax_=np.histogram(data, bins=100)
        ax=np.mean(get_timeWins4TimeVect(ax_), 1)
        pdf=lognorm.pdf(ax, sigma, loc, scale)

        mode=ax[np.argmax(pdf)]

        pdfn=pdf/np.sum(pdf)
        argmedian=np.searchsorted(np.cumsum(pdfn), 0.5)
        median=ax[argmedian]

        thresh= k * (mode + median)
        wThresh[wini]=thresh

    if verbose:
        print('Last steps        ')
    x=np.linspace(0, len(trace_env), num=len(wins))
    xi=np.arange(len(trace_env))
    thresholds=CubicSpline(x, wThresh)(xi)
    thresholds_smooth=get_smooth(thresholds, kernel_type='rectangular', N=win_size, sigma=None)/win_size

    # Detect bad wins
    extend=int(extend_ms/1000*sr)

    events=trace_env>thresholds_smooth
    badPoints=np.where(events)[0]
    edges=np.where(np.diff(badPoints)>1)[0]
    
    DETECTED=True
    if edges.shape[0]>0:
        badWins_st=np.array([badPoints[0], badPoints[edges[0]]], ndmin=2)
        if edges.shape[0]>1:
            badWins_core=np.array([[badPoints[edges[ind-1]+1], badPoints[edges[ind]]] for ind in range(1, edges.shape[0])], ndmin=2)
            if edges.shape[0]>2:
                badWins_en=np.array([badPoints[edges[edges.shape[0]-1]+1], badPoints[-1]], ndmin=2)
                badWins=np.row_stack([badWins_st, badWins_core, badWins_en])
            else:
                badWins=np.row_stack([badWins_st, badWins_core])
        else:
            badWins=np.copy(badWins_st)
    else:
        if badPoints.shape[0]>0:
            badWins=np.array([badPoints[0], badPoints[-1]], ndmin=2)
        else:
            badWins=np.array([1,2], ndmin=2)
            DETECTED=False
    if DETECTED:
        # Extend bad Wins to merge events which are too close
        starts=badWins[:, 0]-extend
        starts[starts<0]=0
        badWins=np.column_stack([starts, badWins[:, 1]+extend])
        badWins=get_timeWinsIntersect(badWins, badWins, lastPoint=MAX)

        # Detect max values which indicates IEDs
        IEDs=np.array([np.argmax(trace_env[win[0]:win[1]])+win[0] for win in badWins])
    else:
        badWins=np.array([], ndmin=2)
        IEDs=np.array([])
        
    if verbose:
        print('Done        ')
    return badWins, IEDs
