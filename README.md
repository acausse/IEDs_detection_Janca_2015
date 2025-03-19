# IED Detection and Time Window Management

This repository contains Python functions for detecting **interictal epileptiform discharges (IEDs)** from **intracranial EEG (iEEG) signals**. Additionally, it includes utility functions for **managing time windows**, which are essential for analyzing neural signals.  

This is based on the following paper that you need to cite for reference:
https://link.springer.com/article/10.1007/s10548-014-0379-1
Detection of interictal epileptiform discharges using signal envelope distribution modelling: application to epileptic and non-epileptic intracranial recordings
        Radek Janca, Petr Jezdik, Roman Cmejla, Martin Tomasek, Gregory A Worrell, Matt Stead, Joost Wagenaar, John G R Jefferys, Pavel Krsek, Vladimir Komarek, Premysl Jiruska, Petr Marusic
         Brain Topogr. 2015
         

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
