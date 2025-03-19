# IED Detection and Time Window Management

This repository contains Python functions for detecting **interictal epileptiform discharges (IEDs)** from **intracranial EEG (iEEG) signals**. Additionally, it includes utility functions for **managing time windows**, which are essential for analyzing neural signals.  


## 📖 References
This implementation is based on:

Janca, R., Jezdik, P., Cmejla, R., Tomasek, M., Worrell, G. A., Stead, M., Wagenaar, J., Jefferys, J. G. R., Krsek, P., Komarek, V., Jiruska, P., & Marusic, P. (2015).
Detection of interictal epileptiform discharges using signal envelope distribution modelling: application to epileptic and non-epileptic intracranial recordings.
Brain Topography.
https://link.springer.com/article/10.1007/s10548-014-0379-1
         
## 📌 Features
- **`get_IEDs`**: Implements an IED detection method based on signal envelope distribution modeling.
- **`get_smooth`**: Smooths a given signal using either a Gaussian or a rectangular kernel.
- **Time Window Management Functions**:
  - `get_timeWinsOverlapping`: Generates overlapping time windows within a given range.
  - `get_timeWinsIntersect`: Merges and extracts non-overlapping portions of time windows.
  - `get_timeWins4TimeVect`: Converts a time vector into consecutive time windows.
  - `get_timeWinsTemplatedSignal`: Extracts and templates signal segments based on defined time windows.

## 📌 Important Note: Downsampling to 200 Hz
To speed up computation, it is highly recommended to downsample the signal to 200 Hz before applying get_IEDs.

Suggested Python code for downsampling:
from scipy.signal import resample

```python
srNew = 200  # Target sampling rate
n2resamp = int(srNew / sr * trace.shape[0])
trace_ds = resample(trace, n2resamp)
```

## 📦 Installation
To install the necessary dependencies, run:

```bash
pip install numpy scipy
```
