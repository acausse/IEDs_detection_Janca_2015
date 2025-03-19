# IED Detection and Time Window Management

This repository contains Python functions for detecting **interictal epileptiform discharges (IEDs)** from **intracranial EEG (iEEG) signals**. Additionally, it includes utility functions for **managing time windows**, which are essential for analyzing neural signals.  

## 📌 Features
- **`get_IEDs`**: Implements an IED detection method based on signal envelope distribution modeling.
- **`get_smooth`**: Smooths a given signal using either a Gaussian or a rectangular kernel.
- **Time Window Management Functions**:
  - `get_timeWinsOverlapping`: Generates overlapping time windows within a given range.
  - `get_timeWinsIntersect`: Merges and extracts non-overlapping portions of time windows.
  - `get_timeWins4TimeVect`: Converts a time vector into consecutive time windows.
  - `get_timeWinsTemplatedSignal`: Extracts and templates signal segments based on defined time windows.

## 📦 Installation
To install the necessary dependencies, run:

```bash
pip install numpy scipy
