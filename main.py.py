import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cmath
import math


# --- FROM-SCRATCH TRANSFORM FUNCTIONS ---

def _pad_to_power_of_2(x: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pads a 1D signal with zeros to the next power of 2.
    Returns the padded signal and the new length (N_padded).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N & (N - 1) == 0:
        # N is already a power of 2
        return x, N

    # Find the next power of 2
    target_N = 1 << (N - 1).bit_length()

    # Create a new array of zeros
    x_padded = np.zeros(target_N)
    # Copy the original signal into it
    x_padded[:N] = x

    return x_padded, target_N


def _scratch_fft_recursive(x: np.ndarray) -> np.ndarray:
    """
    Computes the FAST Fourier Transform (FFT) of a 1D array 'x'
    using the recursive Cooley-Tukey DIT algorithm.

    *** Assumes N is a power of 2. ***
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)

    # Base Case
    if N == 1:
        return x

    # Recursive Step (Divide)
    x_even = _scratch_fft_recursive(x[::2])
    x_odd = _scratch_fft_recursive(x[1::2])

    # Twiddle factors (W_N^k)
    k_array = np.arange(N // 2)
    twiddles = np.exp(-2j * np.pi * k_array / N)

    # Combine (Butterfly Operation)

    t_product = twiddles * x_odd

    X = np.zeros(N, dtype=complex)
    half_N = N // 2

    X[:half_N] = x_even + t_product
    X[half_N:] = x_even - t_product

    return X


def _scratch_ifft(X: np.ndarray) -> np.ndarray:
    """
    Computes the FAST Inverse FFT from scratch using the
    conjugate-compute-conjugate method.

    *** Assumes N is a power of 2. ***
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)

    if N == 0:
        return np.array([])

    # 1. Take conjugate of input
    X_conj = np.conjugate(X)

    # 2. Compute forward FFT of the conjugate
    fft_result = _scratch_fft_recursive(X_conj)

    # 3. Take conjugate of result
    ifft_result_conj = np.conjugate(fft_result)

    # 4. Divide by N
    return ifft_result_conj / N


# --- NEW: "From-Scratch" SLOW DFT / IDFT ---

def _scratch_dft(x: np.ndarray) -> np.ndarray:
    """
    Computes the 'slow' Discrete Fourier Transform (DFT)
    using the direct formula. Complexity is O(N^2).
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    #
    for k in range(N):  # For each frequency component
        for n in range(N):  # Sum over all time samples
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def _scratch_idft(X: np.ndarray) -> np.ndarray:
    """
    Computes the 'slow' Inverse Discrete Fourier Transform (IDFT)
    using the direct formula. Complexity is O(N^2).
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)

    #
    for n in range(N):  # For each time sample
        for k in range(N):  # Sum over all frequency components
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)

    return x / N  # Don't forget to normalize



# --- END OF NEW FUNCTIONS ---


def _load_amplitude_phase_from_path(filepath: str) -> tuple[int, np.ndarray, np.ndarray] | None:
    """
    Loads complex spectrum data (Amplitude and Phase) from a file.
    Handles the specific format with header lines and amplitude/phase pairs.
    """
    amplitudes, phases = [], []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

            data_lines = lines[3:]

            for line in data_lines:
                cleaned_line = line.strip().replace('f', '')
                parts = cleaned_line.split()

                if len(parts) >= 2:
                    try:
                        amp = float(parts[0])
                        phase = float(parts[1])
                        amplitudes.append(amp)
                        phases.append(phase)
                    except ValueError:
                        continue
                elif len(parts) == 1:
                    try:
                        amp = float(parts[0])
                        amplitudes.append(amp)
                        phases.append(0.0)
                    except ValueError:
                        continue

        if not amplitudes:
            messagebox.showwarning("Warning", f"No valid Amplitude/Phase data found in '{filepath}'.")
            return None

        N = len(amplitudes)
        return N, np.array(amplitudes), np.array(phases)

    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: '{filepath}'.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while reading '{filepath}': {e}")
    return None


def load_signal_from_path(filepath: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Loads time-domain signal data (index and amplitude) from a given file path,
    skipping header lines dynamically.
    """
    indices, amplitudes = [], []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

            data_started = False
            for line in lines:
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) == 2:
                    try:
                        index_val = float(parts[0])
                        amp_val = float(parts[1])
                        indices.append(index_val)
                        amplitudes.append(amp_val)
                        data_started = True
                    except ValueError:
                        if data_started:
                            break
                        continue
                elif data_started:
                    break

        if not indices:
            messagebox.showwarning("Warning", f"No valid signal data (Index, Amplitude) found in '{filepath}'.")
            return None

        indices = np.array(indices)
        amplitudes = np.array(amplitudes)

        min_length = min(len(indices), len(amplitudes))
        if min_length == 0:
            messagebox.showwarning("Warning", f"No valid data points found in '{filepath}'.")
            return None

        indices = indices[:min_length]
        amplitudes = amplitudes[:min_length]

        return indices, amplitudes

    except FileNotFoundError:
        messagebox.showerror("E  rror", f"File not found: '{filepath}'.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while reading '{filepath}': {e}")
    return None


# This function is now part of the FrequencyDomainOperations class
# def load_spectrum_and_plot(self, N, amps, phases, fs): ...

def _display_loaded_spectrum_output(self, amps, phases, N, fs):
    """Displays the loaded amplitude/phase spectrum data in a pop-up box."""
    display_limit = 256
    output_lines = [
        f"Loaded Spectrum Data (N={N}, Fs={fs}Hz)",
        "---------------------------------------",
        "Index (k) | Amplitude | Phase (rad)"
    ]

    for i in range(min(N, display_limit)):
        output_lines.append(f"{i:<9} | {amps[i]:<9.6f} | {phases[i]:.6f}")

    if N > display_limit:
        output_lines.append(f"...\n(Display limited to {display_limit} components)")

    messagebox.showinfo("Loaded Spectrum Output", "\n".join(output_lines))


def get_multiplication_description(factor: float) -> str:
    """Gets a user-friendly description of the multiplication operation."""
    if factor == -1:
        return "Inverted"
    if factor < 0:
        return "Inverted and Scaled"
    if 0 <= factor < 1:
        return "Reduced"
    return "Amplified"


def SignalsAreEqual(TaskName, given_output_filePath, Your_indices, Your_samples, tolerance=0.01):
    expected_indices = []
    expected_samples = []

    try:
        with open(given_output_filePath, 'r') as f:
            lines = f.readlines()

            # Parse file dynamically, skipping non-numeric headers
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        # Try to parse valid index and sample
                        idx = int(float(parts[0]))  # Handle "0.0" as 0
                        val = float(parts[1])
                        expected_indices.append(idx)
                        expected_samples.append(val)
                    except ValueError:
                        continue  # Skip header lines

        if not expected_samples:
            print(f"{TaskName} Test failed: No valid data found in test file.")
            return

    except FileNotFoundError:
        print(f"Error: Test file not found at {given_output_filePath}")
        return
    except Exception as e:
        print(f"Error reading test file {given_output_filePath}: {e}")
        return

    # 1. Check Length
    if len(expected_samples) != len(Your_samples):
        print(f"{TaskName} Test case failed, your signal have different length from the expected one")
        print(f"  Expected Length: {len(expected_samples)}, Got: {len(Your_samples)}")
        return



    # 2. Check Indices
    # We compare indices ensuring they are integers
    mismatched_indices = False
    for i in range(len(Your_indices)):
        if int(Your_indices[i]) != int(expected_indices[i]):
            print(f"{TaskName} Test case failed, your signal have different indicies from the expected one")
            print(f"  Index {i}: Expected {expected_indices[i]}, Got {int(Your_indices[i])}")
            mismatched_indices = True
            break
    if mismatched_indices:
        return

    # 3. Check Values (Samples)
    mismatched_values = False
    for i in range(len(expected_samples)):
        diff = abs(Your_samples[i] - expected_samples[i])
        if diff > tolerance:
            print(f"{TaskName} Test case failed, your signal have different values from the expected one")
            print(f"  Index {i}: Expected {expected_samples[i]}, Got {Your_samples[i]} (Diff: {diff})")
            mismatched_values = True
            break

    if mismatched_values:
        return

    print(f"{TaskName} Test case passed successfully (Tolerance={tolerance})")


def QuantizationTest1(file_name, Your_EncodedValues, Your_QuantizedValues, tolerance=0.01):
    expectedEncodedValues = []
    expectedQuantizedValues = []
    try:
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()

            while line:
                L = line.strip()
                parts = L.split()

                if len(parts) == 2:
                    try:
                        V2 = str(parts[0])
                        V3 = float(parts[1])
                        expectedEncodedValues.append(V2)
                        expectedQuantizedValues.append(V3)
                    except ValueError:
                        print(f"Warning: Skipping non-numeric line in {file_name}: {L}")

                    line = f.readline()
                else:
                    if L:
                        print(f"Warning: Skipping line with unexpected format in {file_name}: {L}")
                    line = f.readline()

    except FileNotFoundError:
        print(f"Error: Test file not found at {file_name}")
        return
    except Exception as e:
        print(f"Error reading test file {file_name}: {e}")
        return

    if ((len(Your_EncodedValues) != len(expectedEncodedValues)) or (
            len(Your_QuantizedValues) != len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        print(f"  Expected Encoded Length: {len(expectedEncodedValues)}, Got: {len(Your_EncodedValues)}")
        print(f"  Expected Quantized Length: {len(expectedQuantizedValues)}, Got: {len(Your_QuantizedValues)}")
        return

    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                f"QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one at index {i}")
            print(f"  Expected: '{expectedEncodedValues[i]}', Got: '{Your_EncodedValues[i]}'")
            return

    for i in range(len(expectedQuantizedValues)):
        diff = abs(Your_QuantizedValues[i] - expectedQuantizedValues[i])
        if diff > tolerance:
            print(
                f"QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one at index {i}")
            print(f"  Expected: {expectedQuantizedValues[i]:.8f}, Got: {Your_QuantizedValues[i]:.8f}")
            print(f"  Difference: {diff:.8f} (which is > {tolerance})")
            return

    print(f"QuantizationTest1 Test case passed successfully (Tolerance={tolerance})")


def QuantizationTest2(file_name, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError,
                      tolerance=0.01):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    try:
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()

            while line:
                L = line.strip()
                parts = L.split()

                if len(parts) == 4:
                    try:
                        V1 = int(parts[0])
                        V2 = str(parts[1])
                        V3 = float(parts[2])
                        V4 = float(parts[3])
                        expectedIntervalIndices.append(V1)
                        expectedEncodedValues.append(V2)
                        expectedQuantizedValues.append(V3)
                        expectedSampledError.append(V4)
                    except ValueError:
                        print(f"Warning: Skipping non-numeric line in {file_name}: {L}")

                    line = f.readline()
                else:
                    if L:
                        print(f"Warning: Skipping line with unexpected format in {file_name}: {L}")
                    line = f.readline()

    except FileNotFoundError:
        print(f"Error: Test file not found at {file_name}")
        return
    except Exception as e:
        print(f"Error reading test file {file_name}: {e}")
        return

    if (len(Your_IntervalIndices) != len(expectedIntervalIndices)
            or len(Your_EncodedValues) != len(expectedEncodedValues)
            or len(Your_QuantizedValues) != len(expectedQuantizedValues)
            or len(Your_SampledError) != len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")

        print(f"  IntervalIndices Length: Expected={len(expectedIntervalIndices)}, Got={len(Your_IntervalIndices)}")
        print(f"  EncodedValues Length: Expected={len(expectedEncodedValues)}, Got={len(Your_EncodedValues)}")
        print(f"  QuantizedValues Length: Expected={len(expectedQuantizedValues)}, Got={len(Your_QuantizedValues)}")
        print(f"  SampledError Length: Expected={len(expectedSampledError)}, Got={len(Your_SampledError)}")
        return

    for i in range(len(Your_IntervalIndices)):
        if (Your_IntervalIndices[i] != expectedIntervalIndices[i]):
            print(
                f"QuantizationTest2 Test case failed, your signal have different indicies from the expected one at index {i}")
            print(f"  Expected: {expectedIntervalIndices[i]}, Got: {Your_IntervalIndices[i]}")
            return

    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                f"QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one at index {i}")
            print(f"  Expected: '{expectedEncodedValues[i]}', Got: '{Your_EncodedValues[i]}'")
            return

    for i in range(len(expectedQuantizedValues)):
        diff = abs(Your_QuantizedValues[i] - expectedQuantizedValues[i])
        if diff > tolerance:
            print(f"QuantizationTest2 Test case failed (QuantizedValues) at index {i}")
            print(f"  Expected: {expectedQuantizedValues[i]:.8f}, Got: {Your_QuantizedValues[i]:.8f}")
            print(f"  Difference: {diff:.8f} (which is > {tolerance})")
            return

    for i in range(len(expectedSampledError)):
        diff = abs(Your_SampledError[i] - expectedSampledError[i])
        if diff > tolerance:
            print(f"QuantizationTest2 Test case failed (SampledError) at index {i}")
            print(f"  Expected: {expectedSampledError[i]:.8f}, Got: {Your_SampledError[i]:.8f}")
            print(f"  Difference: {diff:.8f} (which is > {tolerance})")
            return

    print(f"QuantizationTest2 Test case passed successfully (Tolerance={tolerance})")


def SignalComapreAmplitude(SignalInput=[], SignalOutput=[], tolerance=0.001):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):

            if abs(SignalInput[i] - SignalOutput[i]) > tolerance:
                return False
        return True


def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))


def SignalComaprePhaseShift(SignalInput=[], SignalOutput=[], tolerance=0.0001):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A = round(SignalInput[i])
            B = round(SignalOutput[i])
            if abs(A - B) > tolerance:
                return False
        return True


def Compare_Signals(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")

# --- TASK 6: TIME DOMAIN OPERATIONS ---
class TimeDomainOperations:
    """Handles all Time Domain operations for Task 6."""

    def __init__(self, gui_instance):
        self.gui = gui_instance

    def smoothing(self):
        """1) Smoothing: Compute moving average y(n) for signal x(n)."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices, amplitudes = signal


        window_size = simpledialog.askinteger("Smoothing","Enter number of points for moving average:",initialvalue=3, minvalue=1, maxvalue=len(amplitudes))
        if not window_size:
            return

        if window_size > len(amplitudes):
            messagebox.showerror("Error", "Window size cannot be larger than signal length.")
            return


        smoothed = np.convolve(amplitudes, np.ones(window_size) / window_size, mode='valid')

        new_indices = indices[window_size - 1:]

        self.gui.loaded_signals['main'] = (new_indices, smoothed)
        self.gui._plot_data([(new_indices, smoothed, f"Smoothed Signal (Window={window_size})")],"Signal Smoothing (Moving Average)")

        # Test functionality
        if messagebox.askyesno("Run Test", "Do you want to run Smoothing test?"):
            test_file_path = filedialog.askopenfilename(title="Select Smoothing Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("Smoothing", test_file_path, new_indices, smoothed, tolerance)

    def sharpening_first_derivative(self):
        """2) Sharpening: First derivative y(n) = x(n) - x(n-1)."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices, amplitudes = signal

        diff_vals = np.diff(amplitudes)
        first_deriv = np.insert(diff_vals, 0, amplitudes[0])


        new_indices = indices

        self.gui.loaded_signals['main'] = (new_indices, first_deriv)
        self.gui._plot_data([(new_indices, first_deriv, "First Derivative")],
                            "Signal Sharpening - First Derivative")

        # Test functionality
        if messagebox.askyesno("Run Test", "Do you want to run First Derivative test?"):
            test_file_path = filedialog.askopenfilename(title="Select First Derivative Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("First Derivative", test_file_path, new_indices, first_deriv, tolerance)

    def sharpening_second_derivative(self):
        """2) Sharpening: Second derivative y(n) = x(n+1) - 2x(n) + x(n-1)."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices, amplitudes = signal

        if len(amplitudes) < 3:
            messagebox.showerror("Error", "Signal must have at least 3 samples for second derivative calculation.")
            return

        # Second derivative: y(n) = x(n+1) - 2x(n) + x(n-1)
        second_deriv = amplitudes[2:] - 2 * amplitudes[1:-1] + amplitudes[:-2]
        new_indices = indices[1:-1]

        self.gui.loaded_signals['main'] = (new_indices, second_deriv)
        self.gui._plot_data([(new_indices, second_deriv, "Second Derivative")],
                            "Signal Sharpening - Second Derivative")

        # Test functionality
        if messagebox.askyesno("Run Test", "Do you want to run Second Derivative test?"):
            test_file_path = filedialog.askopenfilename(title="Select Second Derivative Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("Second Derivative", test_file_path, new_indices, second_deriv, tolerance)

    def delay_advance_signal(self):
        """3) Delaying or advancing a signal by k steps."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices, amplitudes = signal

        # Ask for k (positive for delay, negative for advance)
        k = simpledialog.askinteger("Delay/Advance",
                                    "Enter k steps (positive for delay, negative for advance):",
                                    initialvalue=1)
        if k is None:
            return

        new_indices = indices + k
        self.gui.loaded_signals['main'] = (new_indices, amplitudes)

        operation = "Delayed" if k > 0 else "Advanced" if k < 0 else "Same"
        self.gui._plot_data([(new_indices, amplitudes, f"{operation} Signal (k={k})")],f"Signal {operation}")

        # Test functionality
        if messagebox.askyesno("Run Test", f"Do you want to run {operation} test?"):
            test_file_path = filedialog.askopenfilename(title=f"Select {operation} Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual(operation, test_file_path, new_indices, amplitudes, tolerance)

    def fold_signal(self):
        """4) Folding a signal: y(n) = x(-n)."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices, amplitudes = signal

        # Fold the signal: reverse both indices and amplitudes
        folded_indices = -indices[::-1]
        folded_amplitudes = amplitudes[::-1]

        self.gui.loaded_signals['main'] = (folded_indices, folded_amplitudes)
        self.gui._plot_data([(folded_indices, folded_amplitudes, "Folded Signal")],
                            "Signal Folding")

        # Test functionality
        if messagebox.askyesno("Run Test", "Do you want to run Folding test?"):
            test_file_path = filedialog.askopenfilename(title="Select Folding Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("Folding", test_file_path, folded_indices, folded_amplitudes, tolerance)

    def delay_advance_folded_signal(self):
        """5) Delaying or advancing a folded signal by k steps."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices, amplitudes = signal

        k = simpledialog.askinteger("Delay/Advance Folded","Enter k steps for folded signal (positive for delay, negative for advance):",initialvalue=1)
        if k is None:
            return

        new_indices = indices + k

        self.gui.loaded_signals['main'] = (new_indices, amplitudes)

        operation = "Delayed" if k > 0 else "Advanced" if k < 0 else "Same"
        self.gui._plot_data([(new_indices, amplitudes, f"{operation} Folded Signal (k={k})")],
                            f"Folded Signal {operation}")

        # Test functionality
        if messagebox.askyesno("Run Test", f"Do you want to run {operation} Folded test?"):
            test_file_path = filedialog.askopenfilename(title=f"Select {operation} Folded Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    # Pass 'amplitudes' directly (assuming they are already in the correct folded state)
                    SignalsAreEqual(f"{operation} Folded", test_file_path, new_indices, amplitudes, tolerance)
    def remove_dc_component_time_domain(self):
        """6) Remove DC component in time domain."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices, amplitudes = signal

        # Remove DC component (subtract mean)
        dc_component = np.mean(amplitudes)
        signal_no_dc = amplitudes - dc_component

        self.gui.loaded_signals['main'] = (indices, signal_no_dc)
        self.gui._plot_data([
            (indices, amplitudes, "Original Signal"),
            (indices, signal_no_dc, "Signal without DC")
        ], "DC Component Removal (Time Domain)")

        # Test functionality
        if messagebox.askyesno("Run Test", "Do you want to run DC Removal test?"):
            test_file_path = filedialog.askopenfilename(title="Select DC Removal Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("DC Removal", test_file_path, indices, signal_no_dc, tolerance)

    def convolve_signals(self):
        """7) Convolve two signals."""

        file1 = filedialog.askopenfilename(title="Select First Signal for Convolution")
        if not file1:
            return
        signal1 = load_signal_from_path(file1)
        if not signal1:
            return


        file2 = filedialog.askopenfilename(title="Select Second Signal for Convolution")
        if not file2:
            return
        signal2 = load_signal_from_path(file2)
        if not signal2:
            return

        indices1, amplitudes1 = signal1
        indices2, amplitudes2 = signal2

        # Perform convolution
        convolved_amplitudes = np.convolve(amplitudes1, amplitudes2, mode='full')

        # Calculate new indices for convolved signal
        start_index = indices1[0] + indices2[0]
        end_index = indices1[-1] + indices2[-1]
        convolved_indices = np.arange(start_index, end_index + 1)

        # Ensure same length
        min_len = min(len(convolved_indices), len(convolved_amplitudes))
        convolved_indices = convolved_indices[:min_len]
        convolved_amplitudes = convolved_amplitudes[:min_len]

        self.gui.loaded_signals['main'] = (convolved_indices, convolved_amplitudes)
        self.gui._plot_data([(convolved_indices, convolved_amplitudes, "Convolved Signal")],
                            "Signal Convolution")

        # Test functionality using ConvTest
        if messagebox.askyesno("Run Test", "Do you want to run Convolution test?"):
            test_file_path = filedialog.askopenfilename(title="Select Convolution Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("Convolution", test_file_path, convolved_indices, convolved_amplitudes, tolerance)

    def cross_correlation(self):
        """8) Compute normalized cross-correlation of two signals."""

        file1 = filedialog.askopenfilename(title="Select First Signal for Cross-Correlation")
        if not file1:
            return
        signal1 = load_signal_from_path(file1)
        if not signal1:
            return


        file2 = filedialog.askopenfilename(title="Select Second Signal for Cross-Correlation")
        if not file2:
            return
        signal2 = load_signal_from_path(file2)
        if not signal2:
            return


        _, amplitudes1 = signal1
        _, amplitudes2 = signal2


        x = np.array(amplitudes1)
        y = np.array(amplitudes2)

        energy_x = np.sum(x ** 2)
        energy_y = np.sum(y ** 2)
        normalization_factor = np.sqrt(energy_x * energy_y)

        N = len(x)
        M = len(y)
        total_length = N + M - 1

        x_padded = np.pad(x, (0, total_length - N))
        y_padded = np.pad(y, (0, total_length - M))

        correlation = []
        for k in range(total_length):

            y_shifted = np.roll(y_padded, -k)

            corr_value = np.sum(x_padded * y_shifted)
            correlation.append(corr_value)

        correlation = np.array(correlation)

        # 4. Apply Normalization
        if normalization_factor != 0:
            correlation = correlation / normalization_factor
        else:
            correlation = np.zeros_like(correlation)

        # 5. Generate Indices (0 to total_length - 1)
        correlation_indices = np.arange(total_length)


        self.gui.loaded_signals['main'] = (correlation_indices, correlation)
        self.gui._plot_data([(correlation_indices, correlation, "Cross-Correlation")],
                            "Normalized Cross-Correlation")

        # Test functionality
        if messagebox.askyesno("Run Test", "Do you want to run Cross-Correlation test?"):
            test_file_path = filedialog.askopenfilename(title="Select Cross-Correlation Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("Cross-Correlation", test_file_path, correlation_indices, correlation, tolerance)

    def auto_correlation(self):
        """8) Compute normalized auto-correlation of a signal."""
        signal = self.gui._get_primary_signal()
        if not signal:
            return

        indices1, amplitudes = signal
        x = np.array(amplitudes)
        N = len(x)

        # Energy for normalization
        normalization_factor = np.sum(x ** 2)

        total_length_corr = 2 * N - 1
        x_padded = np.pad(x, (0, total_length_corr - N))

        correlation = []
        for k in range(total_length_corr):
            # Shift circular/padded
            x_shifted = np.roll(x_padded, -k)  # Shift left
            corr_value = np.sum(x_padded * x_shifted)
            correlation.append(corr_value)

        correlation = np.array(correlation)

        # Apply Normalization
        if normalization_factor != 0:
            correlation = correlation / normalization_factor
        else:
            correlation = np.zeros_like(correlation)

        autocorrelation_indices = np.arange(total_length_corr)

        self.gui.loaded_signals['main'] = (autocorrelation_indices, correlation)
        self.gui._plot_data([(autocorrelation_indices, correlation, "Auto-Correlation")],
                            "Normalized Auto-Correlation")

        if messagebox.askyesno("Run Test", "Do you want to run Auto-Correlation test?"):
            test_file_path = filedialog.askopenfilename(title="Select Auto-Correlation Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("Auto-Correlation", test_file_path, autocorrelation_indices, correlation, tolerance)
    def cross_correlation_periodic(self):
        """9) Compute normalized cross-correlation of periodic signals."""
        # Load first periodic signal
        file1 = filedialog.askopenfilename(title="Select First Periodic Signal")
        if not file1:
            return
        signal1 = load_signal_from_path(file1)
        if not signal1:
            return

        # Load second periodic signal
        file2 = filedialog.askopenfilename(title="Select Second Periodic Signal")
        if not file2:
            return
        signal2 = load_signal_from_path(file2)
        if not signal2:
            return

        indices1, amplitudes1 = signal1
        indices2, amplitudes2 = signal2

        # For periodic signals, we can use circular correlation
        # Pad to the same length
        max_len = max(len(amplitudes1), len(amplitudes2))
        x1_padded = np.pad(amplitudes1, (0, max_len - len(amplitudes1)), 'constant')
        x2_padded = np.pad(amplitudes2, (0, max_len - len(amplitudes2)), 'constant')

        # Normalize
        x1_norm = (x1_padded - np.mean(x1_padded)) / (np.std(x1_padded) * len(x1_padded))
        x2_norm = (x2_padded - np.mean(x2_padded)) / np.std(x2_padded)

        # Compute circular cross-correlation using FFT
        X1 = np.fft.fft(x1_norm)
        X2 = np.fft.fft(x2_norm)
        correlation_freq = X1 * np.conj(X2)
        correlation = np.fft.ifft(correlation_freq).real

        correlation_indices = np.arange(len(correlation))

        self.gui.loaded_signals['main'] = (correlation_indices, correlation)
        self.gui._plot_data([(correlation_indices, correlation, "Periodic Cross-Correlation")],
                            "Normalized Cross-Correlation (Periodic Signals)")

        # Test functionality
        if messagebox.askyesno("Run Test", "Do you want to run Periodic Cross-Correlation test?"):
            test_file_path = filedialog.askopenfilename(title="Select Periodic Cross-Correlation Test File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                if tolerance is not None:
                    SignalsAreEqual("Periodic Cross-Correlation", test_file_path, correlation_indices, correlation,
                                    tolerance)

    def time_delay_analysis(self):
        """10) Time delay analysis between two periodic signals."""
        # Load first signal
        file1 = filedialog.askopenfilename(title="Select First Periodic Signal")
        if not file1:
            return
        signal1 = load_signal_from_path(file1)
        if not signal1:
            return

        # Load second signal
        file2 = filedialog.askopenfilename(title="Select Second Periodic Signal")
        if not file2:
            return
        signal2 = load_signal_from_path(file2)
        if not signal2:
            return

        indices1, amplitudes1 = signal1
        indices2, amplitudes2 = signal2

        # Ask for sampling period
        Ts = simpledialog.askfloat("Sampling Period",
                                   "Enter sampling period (Ts) in seconds:",
                                   initialvalue=1.0)
        if Ts is None:
            return

        # Compute cross-correlation to find delay
        correlation = np.correlate(amplitudes1, amplitudes2, mode='full')
        lags = np.arange(-len(amplitudes2) + 1, len(amplitudes1))

        # Find the lag with maximum correlation
        max_corr_idx = np.argmax(correlation)
        delay_lags = lags[max_corr_idx]
        delay_time = delay_lags * Ts

        # Display results
        result_text = f"Time Delay Analysis Results:\n"
        result_text += f"Maximum correlation at lag: {delay_lags} samples\n"
        result_text += f"Time delay: {delay_time:.4f} seconds\n"
        result_text += f"Signal 2 is {'delayed' if delay_time > 0 else 'advanced'} by {abs(delay_time):.4f} seconds"

        messagebox.showinfo("Time Delay Analysis", result_text)

        # Plot the correlation with marked peak
        self.gui._plot_data([(lags, correlation, "Cross-Correlation")],
                            f"Time Delay Analysis (Delay: {delay_time:.4f}s)")

        # Mark the peak
        self.gui.ax[0].axvline(x=delay_lags, color='r', linestyle='--',
                               label=f'Peak at lag {delay_lags}')
        self.gui.ax[0].legend()
        self.gui.canvas.draw()


class FIROperations:
    """Handles Task 7: FIR Filtering and Resampling operations."""

    def __init__(self, gui_instance):
        self.gui = gui_instance

    def _get_window_parameters(self, stop_attenuation):
        """
        Determines window type and N calculation factor based on stop attenuation.
        Returns: (window_name, factor_numerator)
        """
        # Based on standard FIR design table
        if stop_attenuation <= 21:
            return "rectangular", 0.9
        elif stop_attenuation <= 44:
            return "hanning", 3.1
        elif stop_attenuation <= 53:
            return "hamming", 3.3
        else:
            return "blackman", 5.5

    def _calculate_window_function(self, window_name, N):
        """Computes window coefficients w(n) manually."""
        n = np.arange(N)
        if window_name == "rectangular":
            return np.ones(N)
        elif window_name == "hanning":
            return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        elif window_name == "hamming":
            return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
        elif window_name == "blackman":
            return 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
        return np.ones(N)

    def _calculate_ideal_response(self, filter_type, fc1, fc2, N):
        """
        Computes ideal impulse response hd(n).
        fc1, fc2 are normalized frequencies (0 to 0.5, or similar, here we assume f/Fs).
        """
        alpha = (N - 1) / 2
        n = np.arange(N)

        # Helper for sinc: sin(x)/x. Handle division by zero manually.
        def manual_sinc(x):
            result = np.zeros_like(x)
            mask = (x == 0)
            result[mask] = 1.0
            result[~mask] = np.sin(x[~mask]) / x[~mask]
            return result

        # Adjust indices to be centered around 0 for calculation
        m = n - alpha

        # Convert normalized freq (0 to 1 range relative to Fs) to radians
        # Note: If fc is f/Fs, then omega = 2 * pi * fc
        w1 = 2 * np.pi * fc1

        hd = np.zeros(N)

        if filter_type == "Low Pass":
            # 2 * fc * sinc(2 * fc * m) -> standard formula
            # using normalized f where 0.5 is nyquist
            hd = 2 * fc1 * manual_sinc(2 * np.pi * fc1 * m)

        elif filter_type == "High Pass":
            # delta(n) - LowPass
            # Low pass part
            hlp = 2 * fc1 * manual_sinc(2 * np.pi * fc1 * m)
            # Delta part (only at n = alpha)
            delta = np.zeros(N)
            delta[int(alpha)] = 1.0 if N % 2 != 0 else 0  # Technically alpha is integer only if N is odd
            hd = delta - hlp

        elif filter_type == "Band Pass":
            # LowPass(f2) - LowPass(f1)
            w2 = 2 * np.pi * fc2
            h2 = 2 * fc2 * manual_sinc(w2 * m)
            h1 = 2 * fc1 * manual_sinc(w1 * m)
            hd = h2 - h1

        elif filter_type == "Band Stop":
            # delta(n) - BandPass
            w2 = 2 * np.pi * fc2
            h2 = 2 * fc2 * manual_sinc(w2 * m)
            h1 = 2 * fc1 * manual_sinc(w1 * m)
            hbp = h2 - h1

            delta = np.zeros(N)
            if N % 2 != 0:  # Check if center exists
                delta[int(alpha)] = 1.0
            hd = delta - hbp

        return hd

    def _manual_convolution(self, x, h):
        """Performs convolution from scratch."""
        N = len(x)
        M = len(h)
        total_len = N + M - 1
        y = np.zeros(total_len)

        for n in range(total_len):
            # Sum(x[k] * h[n-k])
            val = 0
            for k in range(N):
                if 0 <= n - k < M:
                    val += x[k] * h[n - k]
            y[n] = val
        return y

    def _save_coefficients(self, coefficients):
        """Saves filter coefficients to a text file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Filter Coefficients",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if not filepath:
            messagebox.showinfo("Save Info", "Saving coefficients was cancelled.")
            return

        try:
            with open(filepath, 'w') as f:
                f.write("0\n")
                f.write("0\n")
                f.write(f"{len(coefficients)}\n")
                f.write("0\n")

                for i, val in enumerate(coefficients):
                    f.write(f"{i} {val:.10f}\n")

            messagebox.showinfo("Success", f"Coefficients saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def fir_filtering_gui(self, external_signal=None, external_fs=None):
        """
        GUI entry point for FIR Filter design.
        If external_signal is provided (from Resampling), it returns the filtered signal
        instead of plotting immediately.
        """
        dialog = tk.Toplevel(self.gui.root)
        dialog.title("FIR Filter Specifications")
        dialog.geometry("400x450")

        # Variables
        filter_type_var = tk.StringVar(value="Low Pass")
        fs_var = tk.StringVar(value="8000")
        if external_fs: fs_var.set(str(external_fs))

        stop_att_var = tk.StringVar(value="50")
        trans_band_var = tk.StringVar(value="500")
        f1_var = tk.StringVar(value="1500")  # Cutoff or Start
        f2_var = tk.StringVar(value="2000")  # End (for BP/BS)

        # UI Layout
        ttk.Label(dialog, text="Filter Type:").pack(pady=5)
        ttk.Combobox(dialog, textvariable=filter_type_var,
                     values=["Low Pass", "High Pass", "Band Pass", "Band Stop"],
                     state="readonly").pack()

        ttk.Label(dialog, text="Sampling Freq (Fs):").pack(pady=2)
        ttk.Entry(dialog, textvariable=fs_var).pack()

        ttk.Label(dialog, text="Stop Attenuation (dB):").pack(pady=2)
        ttk.Entry(dialog, textvariable=stop_att_var).pack()

        ttk.Label(dialog, text="Transition Band (Hz):").pack(pady=2)
        ttk.Entry(dialog, textvariable=trans_band_var).pack()

        ttk.Label(dialog, text="Cutoff F1/Start (Hz):").pack(pady=2)
        ttk.Entry(dialog, textvariable=f1_var).pack()

        f2_label = ttk.Label(dialog, text="Cutoff F2/End (Hz) (for BP/BS):")
        f2_entry = ttk.Entry(dialog, textvariable=f2_var)

        def update_f2_visibility(*args):
            current_type = filter_type_var.get()
            if current_type in ["Band Pass", "Band Stop"]:
                f2_label.pack(pady=2)
                f2_entry.pack()
            else:
                f2_label.pack_forget()
                f2_entry.pack_forget()

        filter_type_var.trace_add("write", update_f2_visibility)
        update_f2_visibility()

        result_container = {}

        def apply_filter():
            try:
                ftype = filter_type_var.get()
                fs = float(fs_var.get())
                stop_att = float(stop_att_var.get())
                trans_band = float(trans_band_var.get())
                f1_raw = float(f1_var.get())
                f2_raw = float(f2_var.get()) if ftype in ["Band Pass", "Band Stop"] else 0

                # 1. Determine Window & N
                win_type, factor = self._get_window_parameters(stop_attenuation=stop_att)

                # Normalized transition width
                norm_trans_width = trans_band / fs

                # Calculate N
                N = int(np.ceil(factor / norm_trans_width))
                if N % 2 == 0:
                    N += 1  # Force odd

                print(f"Filter Design: Type={ftype}, Win={win_type}, N={N}")

                # 2. Adjust Frequencies (Half transition shift)
                # Normalize frequencies by dividing by Fs
                f1_norm = f1_raw / fs
                f2_norm = f2_raw / fs
                half_trans_norm = 0.5 * norm_trans_width

                c1 = 0
                c2 = 0

                if ftype == "Low Pass":
                    # f1 is Pass Edge -> Center is f1 + half_width
                    c1 = f1_norm + half_trans_norm

                elif ftype == "High Pass":
                    # f1 is Pass Edge -> Center is f1 - half_width
                    c1 = f1_norm - half_trans_norm

                elif ftype == "Band Pass":
                    # f1, f2 are Pass Edges (Inner) -> Expand Outward
                    c1 = f1_norm - half_trans_norm
                    c2 = f2_norm + half_trans_norm

                elif ftype == "Band Stop":
                    # f1, f2 are Stop Edges (Inner) -> Expand Outward to Pass Bands
                    # ERROR FIX: Original code had + and - which narrowed the band.
                    c1 = f1_norm - half_trans_norm
                    c2 = f2_norm + half_trans_norm

                # 3. Compute Coefficients
                window = self._calculate_window_function(win_type, N)
                hd = self._calculate_ideal_response(ftype, c1, c2, N)
                h = hd * window

                # 4. Get Signal
                if external_signal is not None:
                    indices, amplitudes = external_signal
                else:
                    signal = self.gui._get_primary_signal()
                    if not signal:
                        messagebox.showerror("Error", "Load a signal first.")
                        return
                    indices, amplitudes = signal

                # 5. Convolve
                # Use manual convolution
                filtered_amplitudes = self._manual_convolution(amplitudes, h)

                # Adjust indices to account for filter delay (Zero Phase alignment)
                # Group delay for symmetric FIR is (N-1)/2
                start_shift = int(-(N - 1) / 2)
                start_idx = indices[0] + start_shift

                new_len = len(filtered_amplitudes)
                filtered_indices = np.arange(start_idx, start_idx + new_len)

                result_container['h'] = h
                result_container['signal'] = (filtered_indices, filtered_amplitudes)

                dialog.destroy()

            except ValueError as e:
                messagebox.showerror("Input Error", f"Check numeric values: {e}")
        ttk.Button(dialog, text="Apply Filter", command=apply_filter).pack(pady=15)

        self.gui.root.wait_window(dialog)

        # After dialog closes
        if 'signal' in result_container:
            h = result_container['h']
            res_sig = result_container['signal']

            if external_signal is None:
                # Normal operation
                if messagebox.askyesno("Save Coefficients", "Do you want to save the filter coefficients (h(n))?"):
                    self._save_coefficients(h)

                self.gui.loaded_signals['main'] = res_sig
                self.gui._plot_data([
                    (res_sig[0], res_sig[1], "Filtered Signal")
                ], f"FIR Filtered Signal ({filter_type_var.get()})")

                # Test logic for FIR
                if messagebox.askyesno("Run Test", "Do you want to run FIR Filter test?"):
                    test_file_path = filedialog.askopenfilename(title="Select FIR Test File")
                    if test_file_path:
                        tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                        if tolerance is not None:
                            SignalsAreEqual("FIR Filter", test_file_path, res_sig[0], res_sig[1], tolerance)

            else:
                # Resampling return
                return res_sig
        return None

    def resampling_gui(self):

        signal = self.gui._get_primary_signal()
        if not signal: return

        dialog = tk.Toplevel(self.gui.root)
        dialog.title("Resampling Specifications")
        dialog.geometry("300x250")

        m_var = tk.StringVar(value="0")
        l_var = tk.StringVar(value="0")

        ttk.Label(dialog, text="Decimation Factor (M):").pack(pady=5)
        ttk.Entry(dialog, textvariable=m_var).pack()

        ttk.Label(dialog, text="Interpolation Factor (L):").pack(pady=5)
        ttk.Entry(dialog, textvariable=l_var).pack()

        def execute_resampling():
            try:
                M = int(m_var.get())
                L = int(l_var.get())

                indices, amplitudes = signal

                # Base case logic
                if M == 0 and L == 0:
                    messagebox.showerror("Error", "Both M and L cannot be 0.")
                    return

                # Get Fs for filter design
                fs_str = self.gui.gen_params["Sampling Freq (Hz)"].get()
                fs = float(fs_str) if fs_str else 1.0

                processed_signal = (indices, amplitudes)

                # 1. Upsampling (if L != 0)
                if L != 0:
                    # Insert L-1 zeros
                    x = processed_signal[1]
                    N_orig = len(x)
                    x_up = np.zeros(N_orig * L)
                    x_up[::L] = x

                    # Indices expand
                    ind_up = np.arange(processed_signal[0][0], processed_signal[0][0] + len(x_up))

                    processed_signal = (ind_up, x_up)
                    print(f"Upsampled by {L}. New len: {len(x_up)}")

                messagebox.showinfo("Filter Step", "Please configure the Low Pass Filter for resampling.")

                filtered_res = self.fir_filtering_gui(external_signal=processed_signal,
                                                      external_fs=fs * L if L != 0 else fs)

                if not filtered_res:
                    if L != 0:
                     # filtered_res is tuple (indices, amplitudes)
                     indices_filt, amp_filt = filtered_res
                     amp_filt = amp_filt * L
                     filtered_res = (indices_filt, amp_filt)


                processed_signal = filtered_res

                # 3. Downsampling (if M != 0)
                if M != 0:
                    x_filt = processed_signal[1]
                    ind_filt = processed_signal[0]

                    # Decimate
                    x_down = x_filt[::M]
                    ind_down = ind_filt[::M]  # Roughly adjust indices

                    processed_signal = (ind_down, x_down)
                    print(f"Downsampled by {M}. New len: {len(x_down)}")

                # Final Display
                self.gui.loaded_signals['main'] = processed_signal
                self.gui._plot_data([
                    (processed_signal[0], processed_signal[1], "Resampled Signal")
                ], f"Resampling Result (M={M}, L={L})")

                # Test
                if messagebox.askyesno("Run Test", "Do you want to run Resampling test?"):
                    test_file_path = filedialog.askopenfilename(title="Select Resampling Test File")
                    if test_file_path:
                        tolerance = simpledialog.askfloat("Test Tolerance", "Enter tolerance:", initialvalue=0.01)
                        if tolerance is not None:
                            SignalsAreEqual("Resampling", test_file_path, processed_signal[0], processed_signal[1],
                                            tolerance)

                dialog.destroy()

            except ValueError:
                messagebox.showerror("Error", "M and L must be integers.")

        ttk.Button(dialog, text="Process Resampling", command=execute_resampling).pack(pady=20)

class FrequencyDomainOperations:
    """Handles DFT, IDFT, and spectrum manipulation for Task 3."""

    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.current_spectrum = None
        self.frequencies = None
        self.dominant_threshold = 0.5
        self.original_max_amplitude = 1.0

    def _calculate_frequencies(self, N, fs):
        """Calculates the frequency bins (in Hz) for a given N and Fs."""
        return np.arange(N) * (fs / N)

    # --- UPDATED: Renamed 'dft_or_idft' to '_run_transform' ---
    # This now selects between FAST (FFT) and SLOW (DFT) algorithms
    def _run_transform(self, signal_data: np.ndarray, inverse: bool, use_fast: bool) -> tuple[np.ndarray, int]:
        """
        Central dispatcher for transforms.
        Returns: (result_array, N_processed)
        """
        if inverse:
            # --- INVERSE ---
            X = np.asarray(signal_data, dtype=complex)
            N = len(X)
            if use_fast:
                # Fast O(N log N) inverse
                result = _scratch_ifft(X)
            else:
                # Slow O(N^2) inverse
                result = _scratch_idft(X)
            return result, N
        else:
            # --- FORWARD ---
            x = np.asarray(signal_data, dtype=float)
            N_original = len(x)
            if use_fast:
                # Pad to power of 2 for FFT
                x_padded, N_padded = _pad_to_power_of_2(x)
                result = _scratch_fft_recursive(x_padded)
                return result, N_padded
            else:
                # Slow O(N^2) forward, no padding needed
                result = _scratch_dft(x)
                return result, N_original

    # --- END UPDATE ---

    def _display_dft_output(self, transform_name="Transform"):
        """Displays the Amplitude and Phase of the current spectrum in a pop-up box."""
        if self.current_spectrum is None:
            return

        N, normalized_amplitudes, phases_rad, fs, complex_spectrum, original_amplitudes = self.current_spectrum

        display_limit = 256
        output_lines = [
            f"{transform_name} Spectrum Data (N={N}, Fs={fs}Hz)",
            "---------------------------------------",
            "Index (k) | Freq (Hz) | Norm. Amp | True Amp | Phase (rad)"
        ]

        for i in range(min(N, display_limit)):
            freq = self.frequencies[i] if self.frequencies is not None else i * (fs / N)
            output_lines.append(
                f"{i:<9} | {freq:<9.2f} | {normalized_amplitudes[i]:<10.6f} | {original_amplitudes[i]:<8.6f} | {phases_rad[i]:.6f}")

        if N > display_limit:
            output_lines.append(f"...\n(Display limited to {display_limit} components)")

        messagebox.showinfo(f"{transform_name} Spectrum Output", "\n".join(output_lines))

    # --- UPDATED: Renamed 'apply_dft' to 'apply_transform' ---
    def apply_transform(self, signal_data: tuple[np.ndarray, np.ndarray], use_fast: bool):
        """Applies FFT or DFT (from scratch) and sets up the spectrum."""
        indices, amplitudes = signal_data
        N = len(amplitudes)
        transform_name = "FFT" if use_fast else "DFT"

        if N == 0:
            messagebox.showerror("Error", f"Signal is empty. Cannot apply {transform_name}.")
            return

        fs_str = self.gui._ask_for_sampling_frequency()
        if fs_str is None: return
        try:
            fs = float(fs_str)
            if fs <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Sampling frequency must be a valid positive number.")
            return

        # Use the from-scratch transform
        complex_spectrum, N_processed = self._run_transform(amplitudes, inverse=False, use_fast=use_fast)

        # Show performance warning for slow DFT
        if not use_fast and N > 512:
            messagebox.showinfo("Performance", f"Slow DFT (O(N^2)) complete for N={N}.\n(This was slow, wasn't it?)")

        # Handle padding
        if N_processed != N:
            self.gui._update_status(f"Signal padded from {N} to {N_processed} for FFT.")
            N = N_processed  # Update N to be the new padded length

        amplitudes_freq = np.abs(complex_spectrum)
        phases_rad = np.angle(complex_spectrum)

        max_amplitude = np.max(amplitudes_freq)
        self.original_max_amplitude = max_amplitude if max_amplitude > 0 else 1.0
        normalized_amplitudes = amplitudes_freq / self.original_max_amplitude

        self.current_spectrum = (
            N, normalized_amplitudes, phases_rad, fs, complex_spectrum.copy(), amplitudes_freq.copy())
        self.frequencies = self._calculate_frequencies(N, fs)

        self._plot_spectrum(self.frequencies, normalized_amplitudes, phases_rad, N, fs,
                            f"{transform_name} Spectrum (from Scratch)")
        self._display_dft_output(transform_name)
        self.gui._update_status(
            f"{transform_name} applied. N={N}, Fs={fs}Hz. Max amplitude: {max_amplitude:.2f}. Spectrum normalized.")

        if messagebox.askyesno("Run Test", f"Do you want to run {transform_name} comparison tests?"):
            test_file_path = filedialog.askopenfilename(title=f"Select {transform_name} Test File (Amplitude/Phase)")
            if test_file_path:
                amp_tolerance = simpledialog.askfloat("Test Tolerance",
                                                      f"Enter tolerance for {transform_name} Amplitude test:",
                                                      initialvalue=0.001)
                if amp_tolerance is None:
                    print(f"--- {transform_name} Test Cancelled ---")
                    return

                phase_tolerance = simpledialog.askfloat("Test Tolerance",
                                                        f"Enter tolerance for {transform_name} Phase test:",
                                                        initialvalue=0.0001)
                if phase_tolerance is None:
                    print(f"--- {transform_name} Test Cancelled ---")
                    return

                print(f"\n--- Running {transform_name} Test ---")
                expected_data = _load_amplitude_phase_from_path(test_file_path)

                if expected_data:
                    N_exp, expected_amps, expected_phases = expected_data
                    N_calc, _, calc_phases, _, _, calc_amps = self.current_spectrum

                    # Handle padding difference
                    if N_exp < N_calc:
                        print(f"Test Note: Padding expected data from {N_exp} to {N_calc} to match FFT output.")
                        expected_amps = np.pad(expected_amps, (0, N_calc - N_exp), 'constant')
                        expected_phases = np.pad(expected_phases, (0, N_calc - N_exp), 'constant')
                        N_exp = N_calc
                    elif N_calc < N_exp and not use_fast:
                        print(f"Test Note: Truncating expected data from {N_exp} to {N_calc} to match DFT output.")
                        expected_amps = expected_amps[:N_calc]
                        expected_phases = expected_phases[:N_calc]
                        N_exp = N_calc

                    if N_exp != N_calc:
                        print(f"{transform_name} Test case failed: Mismatched N. Expected {N_exp}, Got {N_calc}")
                    else:
                        amp_test_passed = SignalComapreAmplitude(expected_amps, calc_amps, amp_tolerance)
                        print(
                            f"{transform_name} Amplitude Test: {'PASSED' if amp_test_passed else 'FAILED'} (Tolerance={amp_tolerance})")

                        rounded_expected_phases = [RoundPhaseShift(p) for p in expected_phases]
                        rounded_calc_phases = [RoundPhaseShift(p) for p in calc_phases]

                        phase_test_passed = SignalComaprePhaseShift(rounded_expected_phases, rounded_calc_phases,
                                                                    phase_tolerance)
                        print(
                            f"{transform_name} Phase Shift Test: {'PASSED' if phase_test_passed else 'FAILED'} (Tolerance={phase_tolerance})")
                else:
                    print(f"{transform_name} Test case failed: Could not load or parse test file.")

                print("------------------------\n")
                self.gui._update_status(f"{transform_name} Test complete. Check console for results.")

    def load_spectrum_and_plot(self, N, amps, phases, fs):
        """Loads and plots a spectrum directly from Amplitude/Phase data."""
        original_N = N

        # Pad to power of 2 for IFFT compatibility
        # IDFT (slow) doesn't need this, but it will work with the padded signal
        if N & (N - 1) != 0:
            target_N = 1 << (N - 1).bit_length()
            print(f"Padding loaded spectrum from {N} to {target_N} for from-scratch IFFT compatibility.")
            amps = np.pad(amps, (0, target_N - N), 'constant')
            phases = np.pad(phases, (0, target_N - N), 'constant')
            N = target_N

        complex_spectrum = amps * np.exp(1j * phases)

        max_amplitude_complex = np.max(np.abs(complex_spectrum))
        self.original_max_amplitude = max_amplitude_complex if max_amplitude_complex > 0 else 1.0

        normalized_amplitudes = np.zeros_like(amps)
        if self.original_max_amplitude > 1e-9:  # Avoid division by zero
            normalized_amplitudes = amps / self.original_max_amplitude

        self.current_spectrum = (N, normalized_amplitudes, phases, fs, complex_spectrum, amps.copy())
        self.frequencies = self._calculate_frequencies(N, fs)

        # Plot and display info using the *original* N for clarity
        self._plot_spectrum(self.frequencies[:original_N], normalized_amplitudes[:original_N], phases[:original_N],
                            original_N, fs, "Loaded Spectrum (Amplitude/Phase)")
        self.gui._update_status(
            f"Spectrum loaded (N={original_N}, Fs={fs}Hz). Padded to N={N}. Ready for Inverse Transform.")

        self._display_loaded_spectrum_output(amps[:original_N], phases[:original_N], original_N, fs)

    def _plot_spectrum(self, freqs, amplitudes, phases, N, fs, title):
        """Plots the Amplitude (normalized) and Phase vs Frequency."""
        if len(self.gui.fig.get_axes()) != 2:
            self.gui._reset_plot()
        else:
            self.gui.ax[0].clear()
            self.gui.ax[1].clear()

        # Plot the first half (0 to Nyquist)
        nyquist_idx = N // 2 + 1
        plot_freqs = freqs[:nyquist_idx]
        plot_amps = amplitudes[:nyquist_idx]
        plot_phases = phases[:nyquist_idx]

        min_length = min(len(plot_freqs), len(plot_amps), len(plot_phases))
        plot_freqs = plot_freqs[:min_length]
        plot_amps = plot_amps[:min_length]
        plot_phases = plot_phases[:min_length]

        self.gui.ax[0].stem(plot_freqs, plot_amps, linefmt='b-', markerfmt='bo', basefmt="k-")
        self.gui.ax[0].set_title(f"Frequency vs Normalized Amplitude [0, 1] (Fs={fs}Hz, N={N})")
        self.gui.ax[0].set_xlabel("Frequency (Hz)")
        self.gui.ax[0].set_ylabel("Normalized Amplitude")
        self.gui.ax[0].grid(True)

        self.gui.ax[1].stem(plot_freqs, plot_phases, linefmt='r-', markerfmt='ro', basefmt="k-")
        self.gui.ax[1].set_title("Frequency vs Phase")
        self.gui.ax[1].set_xlabel("Frequency (Hz)")
        self.gui.ax[1].set_ylabel("Phase (radians)")
        self.gui.ax[1].grid(True)

        self.gui.fig.suptitle(title, fontsize=16)
        self.gui.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.gui.canvas.draw()

    def display_dominant_frequencies(self):
        """Displays frequencies with normalized amplitudes > threshold (Requirement 2)."""
        if self.current_spectrum is None:
            messagebox.showinfo("Error", "Please apply FFT/DFT first to view the spectrum.")
            return

        threshold = self._ask_for_threshold()
        if threshold is None:
            return

        N, normalized_amplitudes, _, fs, complex_spectrum, original_amplitudes = self.current_spectrum
        frequencies = self.frequencies

        dominant_indices = np.where(normalized_amplitudes > threshold)[0]

        if len(dominant_indices) == 0:
            messagebox.showinfo("Dominant Frequencies",
                                f"No frequencies found with normalized amplitude > {threshold}.")
            return

        output_lines = [
            f"Dominant Frequencies (Normalized Amplitude > {threshold})",
            "---------------------------------------",
            "Index (k) | Frequency (Hz) | Norm. Amp | True Amp"
        ]

        for k in dominant_indices:
            output_lines.append(
                f"{k:<10} | {frequencies[k]:<14.4f} | {normalized_amplitudes[k]:<10.6f} | {original_amplitudes[k]:.6f}")

        messagebox.showinfo("Dominant Frequencies", "\n".join(output_lines))
        self.gui._update_status(f"Displayed {len(dominant_indices)} dominant frequencies (threshold={threshold}).")

    def _ask_for_threshold(self):
        """Opens a dialog to ask the user for the dominant frequency threshold."""
        threshold_dialog = tk.Toplevel(self.gui.root)
        threshold_dialog.title("Dominant Frequency Threshold")

        threshold_var = tk.StringVar(value=str(self.dominant_threshold))

        def on_ok():
            try:
                threshold = float(threshold_var.get())
                if not (0 <= threshold <= 1):
                    raise ValueError("Threshold must be between 0 and 1")
                threshold_dialog.result = threshold
                threshold_dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid threshold: {e}")

        ttk.Label(threshold_dialog, text="Enter threshold for dominant frequencies (0-1):").pack(padx=10, pady=10)
        ttk.Entry(threshold_dialog, textvariable=threshold_var).pack(padx=10, pady=5)
        ttk.Button(threshold_dialog, text="OK", command=on_ok).pack(pady=10)

        threshold_dialog.transient(self.gui.root)
        self.gui.root.wait_window(threshold_dialog)

        return getattr(threshold_dialog, 'result', None)

    def modify_spectrum_dialog(self):
        """Opens a dialog to allow modification of amplitude and phase of a signal component (Requirement 3)."""
        if self.current_spectrum is None:
            messagebox.showinfo("Error", "Please apply FFT/DFT first to modify the spectrum.")
            return

        N, norm_amplitudes, phases, fs, complex_spectrum, original_amplitudes = self.current_spectrum

        dialog = tk.Toplevel(self.gui.root)
        dialog.title("3) Modify F(k) Component")
        dialog.geometry("400x200")

        k_var = tk.StringVar(value="0")
        amp_var = tk.StringVar(value="0.0")
        phase_var = tk.StringVar(value="0.0")

        def set_k_data():
            try:
                k = int(k_var.get())
                if not (0 <= k < N):
                    messagebox.showerror("Invalid Index", f"Index k must be between 0 and {N - 1}.")
                    return

                original_amplitude = np.abs(complex_spectrum[k])
                original_phase = np.angle(complex_spectrum[k])

                amp_var.set(f"{original_amplitude:.6f}")
                phase_var.set(f"{original_phase:.6f}")

            except ValueError:
                messagebox.showerror("Input Error", "Index k must be an integer.")

        def apply_modification():
            try:
                k = int(k_var.get())
                new_amplitude = float(amp_var.get())
                new_phase = float(phase_var.get())

                if not (0 <= k < N):
                    messagebox.showerror("Invalid Index", f"Index k must be between 0 and {N - 1}.")
                    return

                if new_amplitude < 0:
                    messagebox.showerror("Input Error", "Amplitude must be non-negative.")
                    return

                is_even = (N % 2 == 0)

                if k == 0 or (is_even and k == N // 2):
                    new_complex = new_amplitude * cmath.exp(1j * new_phase)
                    new_real_val = new_complex.real
                    if not np.isclose(new_complex.imag, 0):
                        messagebox.showwarning("Symmetry",
                                               f"F({k}) (DC or Nyquist) must be a real value. Storing real part ({new_real_val:.2f}) only.")
                    complex_spectrum[k] = new_real_val + 0j
                    status_msg = f"F({k}) modified: Stored real value {new_real_val:.2f}."
                else:
                    new_complex = new_amplitude * cmath.exp(1j * new_phase)
                    complex_spectrum[k] = new_complex
                    conjugate_k = (N - k) % N
                    complex_spectrum[conjugate_k] = np.conjugate(new_complex)
                    status_msg = f"F({k}) modified: Amp={new_amplitude:.2f}, Phase={new_phase:.2f} rad. Symmetrically updated F({conjugate_k})."

                current_max_amplitude = np.max(np.abs(complex_spectrum))
                self.original_max_amplitude = current_max_amplitude if current_max_amplitude > 0 else 1.0

                if self.original_max_amplitude > 1e-9:
                    normalized_amplitudes = np.abs(complex_spectrum) / self.original_max_amplitude
                else:
                    normalized_amplitudes = np.abs(complex_spectrum)  # all zero

                phases = np.angle(complex_spectrum)
                original_amplitudes = np.abs(complex_spectrum)

                self.current_spectrum = (N, normalized_amplitudes, phases, fs, complex_spectrum, original_amplitudes)

                self._plot_spectrum(self.frequencies, normalized_amplitudes, phases, N, fs, "Modified Spectrum")
                self.gui._update_status(status_msg)
                dialog.destroy()

            except ValueError:
                messagebox.showerror("Input Error", "Index, Amplitude, and Phase must be valid numbers.")

        ttk.Label(dialog, text="Component Index (k):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(dialog, textvariable=k_var).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(dialog, text="Get Current F(k)", command=set_k_data).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(dialog, text="New Amplitude:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(dialog, textvariable=amp_var).grid(row=1, column=1, padx=5, pady=5, sticky='ew', columnspan=2)

        ttk.Label(dialog, text="New Phase (rad):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(dialog, textvariable=phase_var).grid(row=2, column=1, padx=5, pady=5, sticky='ew', columnspan=2)

        ttk.Button(dialog, text="Apply Modification", command=apply_modification).grid(row=3, column=0, columnspan=3,
                                                                                       padx=5, pady=10, sticky='ew')

        dialog.transient(self.gui.root)
        self.gui.root.wait_window(dialog)

    def remove_dc_component(self):
        """Sets the DC component F(0) to zero, then updates the spectrum plot (Requirement 4)."""
        if self.current_spectrum is None:
            messagebox.showinfo("Error", "Please apply FFT/DFT first.")
            return

        N, norm_amplitudes, phases, fs, complex_spectrum, original_amplitudes = self.current_spectrum

        if N == 0:
            self.gui._update_status("Cannot remove DC component: signal length N is zero.")
            return

        if np.isclose(complex_spectrum[0], 0):
            self.gui._update_status("DC Component (F(0)) is already zero or negligible.")
            return

        original_dc_value = complex_spectrum[0]
        complex_spectrum[0] = 0 + 0j

        if self.original_max_amplitude > 1e-9:
            normalized_amplitudes = np.abs(complex_spectrum) / self.original_max_amplitude
        else:
            normalized_amplitudes = np.abs(complex_spectrum)

        phases = np.angle(complex_spectrum)
        original_amplitudes = np.abs(complex_spectrum)

        self.current_spectrum = (N, normalized_amplitudes, phases, fs, complex_spectrum, original_amplitudes)

        self._plot_spectrum(self.frequencies, normalized_amplitudes, phases, N, fs,
                            "Spectrum (DC Component Removed)")
        self.gui._update_status(f"DC Component F(0)={original_dc_value.real:.2f} removed. Spectrum updated.")

    def _display_idft_output(self, time_indices, reconstructed_amplitudes, transform_name="IDFT"):
        """Displays the reconstructed time signal data in a pop-up box."""
        N = len(reconstructed_amplitudes)
        display_limit = 256
        output_lines = [
            f"{transform_name} Reconstructed Signal Data (N={N})",
            "---------------------------------------",
            "Index (n) | Amplitude (Real Part)"
        ]

        for i in range(min(N, display_limit)):
            output_lines.append(f"{time_indices[i]:<9} | {reconstructed_amplitudes[i]:.6f}")

        if N > display_limit:
            output_lines.append(f"...\n(Display limited to {display_limit} samples)")

        messagebox.showinfo(f"{transform_name} Signal Output", "\n".join(output_lines))

    # --- UPDATED: 'reconstruct_signal' now takes 'use_fast' param ---
    def reconstruct_signal(self, use_fast: bool):
        """Performs IFFT or IDFT on the current spectrum to reconstruct the time domain signal."""
        if self.current_spectrum is None:
            messagebox.showinfo("Error", "Please apply FFT/DFT first (or load a spectrum).")
            return

        N, _, _, fs, complex_spectrum, _ = self.current_spectrum
        transform_name = "IFFT" if use_fast else "IDFT"

        if np.allclose(complex_spectrum, 0):
            messagebox.showwarning("Empty Spectrum",
                                   "The spectrum is all zeros. Reconstruction will yield a zero signal.")

        # Check for IFFT compatibility (requires power of 2)
        if use_fast and (N & (N - 1) != 0):
            messagebox.showerror("IFFT Error",
                                 f"Cannot run from-scratch IFFT.\nSpectrum length N={N} is not a power of 2.\n\n(This can happen if you load a non-padded spectrum or run DFT first). \nTry using the 'slow' IDFT instead.")
            return

        # Use the from-scratch inverse transform
        reconstructed_signal_complex, _ = self._run_transform(complex_spectrum, inverse=True, use_fast=use_fast)

        reconstructed_amplitudes = reconstructed_signal_complex.real

        max_imag = np.max(np.abs(reconstructed_signal_complex.imag))
        if max_imag > 1e-9:
            messagebox.showwarning("Complex Reconstruction",
                                   f"Reconstructed signal has significant imaginary components (max={max_imag:.2e}).\n"
                                   "This may indicate a non-Hermitian spectrum (due to manual edits?). Using real part only.")

        time_indices = np.arange(N)

        # Truncate signal if it was padded (e.g., from loading a file)
        # We can't know the original length, so we'll show the whole N
        # A better approach would be to store original_N in self.current_spectrum

        self.gui.loaded_signals['main'] = (time_indices, reconstructed_amplitudes)
        self.gui._plot_data([(time_indices, reconstructed_amplitudes, "Reconstructed Signal")],
                            f"Signal Reconstruction ({transform_name} from Scratch)")

        self._display_idft_output(time_indices, reconstructed_amplitudes, transform_name)
        self.gui._update_status(f"Signal successfully reconstructed using {transform_name}. Length N={N}.")

        if messagebox.askyesno("Run Test", f"Do you want to run {transform_name} (SignalsAreEqual) test?"):
            test_file_path = filedialog.askopenfilename(title=f"Select {transform_name} Test File (Time/Amplitude)")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance",
                                                  f"Enter tolerance for {transform_name} test:",
                                                  initialvalue=0.01)
                if tolerance is None:
                    print(f"--- {transform_name} Test Cancelled ---")
                    return

                print(f"\n--- Running {transform_name} Test (SignalsAreEqual) ---")
                time_indices_test, reconstructed_amplitudes_test = self.gui.loaded_signals['main']

                SignalsAreEqual(transform_name,
                                test_file_path,
                                time_indices_test,
                                reconstructed_amplitudes_test,
                                tolerance)

                print("-------------------------------------------\n")
                self.gui._update_status(f"{transform_name} Test complete. Check console for results.")
    # --- END UPDATE ---


class SignalProcessingGUI:
    """Main GUI application for signal processing."""

    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Framework")
        self.root.geometry("1200x800")

        self.loaded_signals = {}

        self.freq_ops = FrequencyDomainOperations(self)
        self.time_ops = TimeDomainOperations(self)  # NEW: Time Domain Operations

        self.gen_params = {
            "Amplitude (A)": tk.StringVar(value="1.0"),
            "Frequency (Hz)": tk.StringVar(value="5.0"),
            "Phase (rad)": tk.StringVar(value="0.0"),
            "Sampling Freq (Hz)": tk.StringVar(value="100.0")
        }
        self.gen_type = tk.StringVar(value="sine")
        self.mult_factor = tk.StringVar(value="2.0")
        self.norm_range = tk.StringVar(value="0_1")
        self.quant_value = tk.StringVar(value="16")
        self.quant_param = tk.StringVar(value="levels")

        self.freq_ops = FrequencyDomainOperations(self)
        self.time_ops = TimeDomainOperations(self)
        self.fir_ops = FIROperations(self)

        self._setup_ui()
        self._create_menus()

    def _setup_ui(self):
        """Initializes and places all the UI components."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_panel = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Welcome! to the Signal Processing Framework")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._reset_plot()

        self._create_file_ops_widgets(control_panel)
        self._create_generation_widgets(control_panel)
        self._create_arithmetic_ops_widgets(control_panel)
        self._create_quantization_widgets(control_panel)

    # --- UPDATED: Menu now has options for FFT and DFT and Time Domain ---
    def _create_menus(self):
        """Creates the main menu bar, including the new Frequency Domain and Time Domain menus."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reset Application", command=self._reset_application_state)
        file_menu.add_command(label="Exit", command=self.root.quit)
        file_menu.add_separator()
        file_menu.add_command(label="Load Spectrum for IDFT/IFFT",
                              command=self._load_spectrum_wrapper)

        freq_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Frequency Domain", menu=freq_menu)

        freq_menu.add_command(label="Apply FFT & Display ",
                              command=self._fft_wrapper_scratch)
        freq_menu.add_command(label="Apply DFT & Display ",
                              command=self._dft_wrapper_scratch)

        freq_menu.add_command(label="Display Dominant Frequencies",
                              command=self.freq_ops.display_dominant_frequencies)

        freq_menu.add_command(label="Modify Spectrum Component (F(k))",
                              command=self.freq_ops.modify_spectrum_dialog)

        freq_menu.add_command(label="Remove DC Component (F(0))",
                              command=self.freq_ops.remove_dc_component)

        freq_menu.add_command(label="Reconstruct Signal (IFFT)",
                              command=self._ifft_reconstruct_wrapper)
        freq_menu.add_command(label="Reconstruct Signal (IDFT)",
                              command=self._idft_reconstruct_wrapper)

        # NEW: Time Domain Menu
        time_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Time Domain", menu=time_menu)

        time_menu.add_command(label="Smoothing (Moving Average)",
                              command=self.time_ops.smoothing)
        time_menu.add_separator()

        sharpening_menu = tk.Menu(time_menu, tearoff=0)
        time_menu.add_cascade(label="Sharpening", menu=sharpening_menu)
        sharpening_menu.add_command(label="First Derivative",
                                    command=self.time_ops.sharpening_first_derivative)
        sharpening_menu.add_command(label="Second Derivative",
                                    command=self.time_ops.sharpening_second_derivative)

        time_menu.add_command(label="Delay/Advance Signal",
                              command=self.time_ops.delay_advance_signal)
        time_menu.add_command(label="Fold Signal",
                              command=self.time_ops.fold_signal)
        time_menu.add_command(label="Delay/Advance Folded Signal",
                              command=self.time_ops.delay_advance_folded_signal)
        time_menu.add_command(label="Remove DC Component (Time Domain)",
                              command=self.time_ops.remove_dc_component_time_domain)
        time_menu.add_command(label="Convolve Two Signals",
                              command=self.time_ops.convolve_signals)

        correlation_menu = tk.Menu(time_menu, tearoff=0)
        time_menu.add_cascade(label="Correlation", menu=correlation_menu)
        correlation_menu.add_command(label="Cross-Correlation",
                                     command=self.time_ops.cross_correlation)
        correlation_menu.add_command(label="Auto-Correlation",
                                     command=self.time_ops.auto_correlation)

        time_menu.add_command(label="Cross-Correlation (Periodic)",
                              command=self.time_ops.cross_correlation_periodic)
        time_menu.add_command(label="Time Delay Analysis",
                              command=self.time_ops.time_delay_analysis)

        time_menu.add_separator()  # Separator for clarity
        time_menu.add_command(label="FIR Filtering",
                              command=lambda: self.fir_ops.fir_filtering_gui())
        time_menu.add_command(label="Resampling",
                              command=self.fir_ops.resampling_gui)

    def _load_spectrum_wrapper(self):
        """Wrapper to handle loading an Amplitude/Phase file and setting the frequency domain state."""
        filepath = filedialog.askopenfilename(title="Select Amplitude/Phase Spectrum File (for IDFT/IFFT)",
                                              filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not filepath:
            return

        spectrum_data = _load_amplitude_phase_from_path(filepath)
        if spectrum_data:
            N, amps, phases = spectrum_data

            fs_str = self._ask_for_sampling_frequency()
            if fs_str is None: return
            try:
                fs = float(fs_str)
                if fs <= 0: raise ValueError
            except ValueError:
                messagebox.showerror("Input Error", "Sampling frequency must be a valid positive number.")
                return

            self.freq_ops.load_spectrum_and_plot(N, amps, phases, fs)

    # --- NEW AND RENAMED WRAPPERS ---

    def _fft_wrapper_scratch(self):
        """Wrapper for calling FAST FFT (from scratch)."""
        signal = self._get_primary_signal()
        if signal:
            self.freq_ops.apply_transform(signal, use_fast=True)

    def _dft_wrapper_scratch(self):
        """Wrapper for calling SLOW DFT (from scratch)."""
        signal = self._get_primary_signal()
        if signal:
            # Add a warning for large signals
            N = len(signal[1])
            if N > 512:  # O(N^2) gets very slow after this
                if not messagebox.askyesno("Performance Warning",
                                           f"You are about to run a slow O(N^2) DFT on a signal of length {N}.\n"
                                           "This could take a very long time (minutes).\n\n"
                                           "Are you sure you want to continue?"):
                    self._update_status("DFT cancelled by user.")
                    return

            self._update_status(f"Running slow O(N^2) DFT for N={N}... Please wait.")
            self.root.update_idletasks()  # Force GUI update to show status

            self.freq_ops.apply_transform(signal, use_fast=False)

    def _ifft_reconstruct_wrapper(self):
        """Wrapper for FAST IFFT reconstruction."""
        self.freq_ops.reconstruct_signal(use_fast=True)

    def _idft_reconstruct_wrapper(self):
        """Wrapper for SLOW IDFT reconstruction."""
        if self.freq_ops.current_spectrum:
            N = self.freq_ops.current_spectrum[0]
            if N > 512:
                if not messagebox.askyesno("Performance Warning",
                                           f"You are about to run a slow O(N^2) IDFT on a spectrum of length {N}.\n"
                                           "This could take a very long time (minutes).\n\n"
                                           "Are you sure you want to continue?"):
                    self._update_status("IDFT cancelled by user.")
                    return
            self._update_status(f"Running slow O(N^2) IDFT for N={N}... Please wait.")
            self.root.update_idletasks()  # Force GUI update

        self.freq_ops.reconstruct_signal(use_fast=False)

    # --- END NEW WRAPPERS ---

    def _ask_for_sampling_frequency(self):
        """Opens a simple dialog to ask the user for the sampling frequency."""
        fs_dialog = tk.Toplevel(self.root)
        fs_dialog.title("Sampling Frequency")

        initial_fs = self.gen_params["Sampling Freq (Hz)"].get()
        fs_var = tk.StringVar(value=initial_fs)

        def on_ok():
            try:
                val = float(fs_var.get())
                if val <= 0:
                    raise ValueError("Fs must be positive.")
                fs_dialog.result = fs_var.get()
                fs_dialog.destroy()
            except ValueError:
                messagebox.showerror("Input Error", "Sampling frequency must be a valid positive number.")

        ttk.Label(fs_dialog, text="Enter Sampling Frequency (Fs) in HZ:").pack(padx=10, pady=10)
        ttk.Entry(fs_dialog, textvariable=fs_var).pack(padx=10, pady=5)
        ttk.Button(fs_dialog, text="OK", command=on_ok).pack(pady=10)

        fs_dialog.transient(self.root)
        self.root.wait_window(fs_dialog)

        return getattr(fs_dialog, 'result', None)

    def _update_status(self, message):
        """Updates the status bar text."""
        self.status_var.set(message)

    def _reset_plot(self):
        """Clears and resets the plot to the default two-panel view."""
        if len(self.fig.get_axes()) != 2:
            plt.close(self.fig)
            self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))
            self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas.get_tk_widget().master)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax[0].clear()
        self.ax[1].clear()
        self.fig.suptitle("Signal Processing Framework", fontsize=16)

        self.ax[0].set_title("Continuous Representation")
        self.ax[0].set_xlabel("Time / Index")
        self.ax[0].set_ylabel("Amplitude")
        self.ax[0].grid(True)

        self.ax[1].set_title("Discrete Representation")
        self.ax[1].set_xlabel("Time / Index")
        self.ax[1].set_ylabel("Amplitude")
        self.ax[1].grid(True)

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()

    def _plot_data(self, signals: list, title: str):
        """Generic plotting function for standard 2-subplot view (Continuous and Discrete)."""
        self._reset_plot()
        self.fig.suptitle(title, fontsize=16)

        colors = ['b', 'g', 'r', 'c', 'm']

        for i, sig in enumerate(signals):
            indices, amplitudes, label = sig

            if len(indices) != len(amplitudes):
                messagebox.showerror("Data Error",
                                     f"Signal '{label}' has mismatched array lengths: "
                                     f"indices({len(indices)}) != amplitudes({len(amplitudes)})")
                continue

            color = colors[i % len(colors)]

            self.ax[0].plot(indices, amplitudes, label=label, color=color)

            num_points = len(indices)
            if num_points > 0:
                max_stem_points = 75
                step = max(1, num_points // max_stem_points)
                plot_indices = indices[::step]
                plot_amplitudes = amplitudes[::step]

                self.ax[1].stem(plot_indices, plot_amplitudes, linefmt=f"{color}-", markerfmt=f"{color}o", basefmt="k-",
                                label=label if i == 0 else "")
            else:
                self._update_status(f"Warning: Signal '{label}' has no data points.")

        if len(signals) > 1:
            self.ax[0].legend()
            self.ax[1].legend()

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()

    def _quantization_plot(self, original_sig, quantized_sig, error_sig, title, L, b):
        """Handles the specialized 3-subplot plot for quantization results."""
        plt.close(self.fig)
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas.get_tk_widget().master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        indices, original_amplitudes = original_sig
        _, quantized_amplitudes = quantized_sig
        _, error_amplitudes = error_sig

        num_points = len(indices)
        max_stem_points = 75
        step = max(1, num_points // max_stem_points)
        plot_indices = indices[::step]

        self.ax[0].plot(indices, original_amplitudes, label="Original Signal", color='b')
        self.ax[0].plot(indices, quantized_amplitudes, label=f"Quantized Signal ($L={L}$)", color='g', linestyle='--')
        self.ax[0].set_title("Original and Quantized Signal (Continuous)")
        self.ax[0].set_ylabel("Amplitude")
        self.ax[0].legend()
        self.ax[0].grid(True)

        self.ax[1].stem(plot_indices, original_amplitudes[::step], linefmt='b-', markerfmt='bo', basefmt="k-",
                        label="Original Signal")
        self.ax[1].stem(plot_indices, quantized_amplitudes[::step], linefmt='g:', markerfmt='gD', basefmt="k-",
                        label=f"Quantized Signal ($L={L}$)")
        self.ax[1].set_title("Original and Quantized Signal (Discrete)")
        self.ax[1].set_ylabel("Amplitude")
        self.ax[1].legend()
        self.ax[1].grid(True)

        self.ax[2].plot(indices, error_amplitudes, label="Quantization Error", color='r')
        self.ax[2].stem(plot_indices, error_amplitudes[::step], linefmt='r-', markerfmt='rx', basefmt="k-")
        self.ax[2].set_title("Quantization Error")
        self.ax[2].set_xlabel("Time / Index")
        self.ax[2].set_ylabel("Error")
        self.ax[2].grid(True)

        self.fig.suptitle(title, fontsize=16)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()

    def _create_file_ops_widgets(self, parent):
        """Widgets for loading, comparing, and resetting."""
        frame = ttk.LabelFrame(parent, text="File Operations", padding=10)
        frame.pack(fill=tk.X, pady=5)

        ttk.Button(frame, text="Load & Display Signal", command=self._load_single_signal).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Compare Two Signals", command=self._compare_two_signals).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Reset Plot & State", command=self._reset_application_state).pack(fill=tk.X,
                                                                                                 pady=(10, 2))

    def _create_generation_widgets(self, parent):
        """Widgets for signal generation."""
        frame = ttk.LabelFrame(parent, text="Signal Generation", padding=10)
        frame.pack(fill=tk.X, pady=5)

        for name, var in self.gen_params.items():
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=name, width=18).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var).pack(side=tk.RIGHT, expand=True, fill=tk.X)

        radio_frame = ttk.Frame(frame)
        radio_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(radio_frame, text="Sine", variable=self.gen_type, value="sine").pack(side=tk.LEFT, expand=True)
        ttk.Radiobutton(radio_frame, text="Cosine", variable=self.gen_type, value="cosine").pack(side=tk.RIGHT,
                                                                                                 expand=True)

        ttk.Button(frame, text="Generate Signal", command=self._generate_signal).pack(fill=tk.X, pady=5)

    def _create_arithmetic_ops_widgets(self, parent):
        """Widgets for arithmetic operations."""
        frame = ttk.LabelFrame(parent, text="Arithmetic Operations", padding=10)
        frame.pack(fill=tk.X, pady=5)

        mult_frame = ttk.Frame(frame)
        mult_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mult_frame, text="Factor:").pack(side=tk.LEFT)
        ttk.Entry(mult_frame, textvariable=self.mult_factor, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(mult_frame, text="Multiply", command=self._multiply_signal).pack(side=tk.LEFT, expand=True,
                                                                                    fill=tk.X)

        norm_frame = ttk.Frame(frame)
        norm_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(norm_frame, text="[0, 1]", variable=self.norm_range, value="0_1").pack(side=tk.LEFT)
        ttk.Radiobutton(norm_frame, text="[-1, 1]", variable=self.norm_range, value="-1_1").pack(side=tk.LEFT, padx=10)
        ttk.Button(norm_frame, text="Normalize", command=self._normalize_signal).pack(side=tk.RIGHT, expand=True,
                                                                                      fill=tk.X)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Add Signals", command=self._add_signals).grid(row=0, column=0, sticky="ew", padx=2,
                                                                                  pady=2)
        ttk.Button(btn_frame, text="Subtract Signals", command=self._subtract_signals).grid(row=0, column=1,
                                                                                            sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Square Signal", command=self._square_signal).grid(row=1, column=0, sticky="ew",
                                                                                      padx=2, pady=2)
        ttk.Button(btn_frame, text="Accumulate", command=self._accumulate_signal).grid(row=1, column=1, sticky="ew",
                                                                                       padx=2, pady=2)
        btn_frame.grid_columnconfigure((0, 1), weight=1)

    def _create_quantization_widgets(self, parent):
        """Widgets for quantization input (levels or bits)."""
        frame = ttk.LabelFrame(parent, text="Quantization", padding=10)
        frame.pack(fill=tk.X, pady=5)

        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, pady=2)

        self.quant_label = ttk.Label(input_frame, text="Levels (L):", width=12)
        self.quant_label.pack(side=tk.LEFT)
        ttk.Entry(input_frame, textvariable=self.quant_value).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        radio_frame = ttk.Frame(frame)
        radio_frame.pack(fill=tk.X, pady=5)

        def update_label():
            self.quant_label.config(text="Levels (L):" if self.quant_param.get() == "levels" else "Bits (N):")

        ttk.Radiobutton(radio_frame, text="Levels (L)", variable=self.quant_param,
                        value="levels", command=update_label).pack(side=tk.LEFT, expand=True)
        ttk.Radiobutton(radio_frame, text="Bits (N)", variable=self.quant_param,
                        value="bits", command=update_label).pack(side=tk.RIGHT, expand=True)

        ttk.Button(frame, text="Quantize Signal", command=self._quantize_signal).pack(fill=tk.X, pady=5)

    def _quantize_signal(self):
        """Performs uniform quantization on the primary signal, plots results, and displays index/amplitude."""
        original_signal = self._get_primary_signal()
        if not original_signal:
            return

        indices, amplitudes = original_signal

        try:
            value = self.quant_value.get()
            if not value:
                messagebox.showerror("Input Error", "Please enter a value for levels or bits.")
                return

            num_input = int(value)
            if num_input <= 0:
                raise ValueError("Value must be a positive integer.")

            L = num_input
            if self.quant_param.get() == "bits":
                b = num_input
                L = 2 ** b
            else:
                b = int(np.ceil(np.log2(L))) if L > 1 else 1

            if L < 2:
                messagebox.showerror("Invalid Input", "Quantization requires 2 or more effective levels.")
                return
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric input: {e}")
            return

        min_val, max_val = np.min(amplitudes), np.max(amplitudes)

        indices_k = np.array([])
        quantized = np.array([])
        error = np.array([])

        if np.isclose(min_val, max_val):
            quantized = amplitudes.copy()
            error = np.zeros_like(amplitudes)
            indices_k = np.zeros_like(amplitudes, dtype=int)
        else:
            q_step = (max_val - min_val) / L

            indices_k = np.floor((amplitudes - min_val) / q_step - 1e-9).astype(int)
            indices_k = np.clip(indices_k, 0,
                                L - 1)

            quantized = min_val + (
                    indices_k + 0.5) * q_step

            error = amplitudes - quantized

        try:
            encoded_values = [f'{k:0{b}b}' for k in indices_k]
        except ValueError:
            b_calc = int(np.ceil(np.log2(L))) if L > 1 else 1
            encoded_values = [f'{k:0{b_calc}b}' for k in indices_k]

        if messagebox.askyesno("Run Test", "Do you want to run QuantizationTest1?"):
            test_file_path = filedialog.askopenfilename(title="Select QuantizationTest1 File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance",
                                                  "Enter tolerance for QuantizationTest1:",
                                                  initialvalue=0.01)
                if tolerance is not None:
                    print("\n--- Running QuantizationTest1 ---")
                    QuantizationTest1(test_file_path,
                                      encoded_values,
                                      quantized,
                                      tolerance)
                    print("---------------------------------\n")
                    self._update_status("QuantizationTest1 complete. Check console for results.")

        if messagebox.askyesno("Run Test", "Do you want to run QuantizationTest2?"):
            test_file_path = filedialog.askopenfilename(title="Select QuantizationTest2 File")
            if test_file_path:
                tolerance = simpledialog.askfloat("Test Tolerance",
                                                  "Enter tolerance for QuantizationTest2:",
                                                  initialvalue=0.01)
                if tolerance is not None:
                    print("\n--- Running QuantizationTest2 ---")

                    QuantizationTest2(test_file_path,
                                      indices_k + 1,
                                      encoded_values,
                                      quantized,
                                      -error,
                                      tolerance)

                    print("---------------------------------\n")
                    self._update_status("QuantizationTest2 complete. Check console for results.")

        self._quantization_plot(
            original_sig=original_signal,
            quantized_sig=(indices, quantized),
            error_sig=(indices, error),
            title=f"Signal Quantization (L={L} levels, N={b} bits)",
            L=L,
            b=b
        )

        self._update_status(f"Quantization complete with {L} levels ({b} bits). Quantized values displayed below.")

        display_limit = 256

        output_lines = [
            f"Quantized Signal Data (L={L}, N={b} bits)",
            "---------------------------------------",
            "Index | Quantized Amplitude"
        ]

        for i in range(min(len(indices), display_limit)):
            output_lines.append(f"{indices[i]:<5.0f} | {quantized[i]:.6f}")

        if len(indices) > display_limit:
            output_lines.append(f"...\n(Display limited to {display_limit} samples)")

        messagebox.showinfo("Quantized Signal Output", "\n".join(output_lines))

    def _reset_application_state(self):
        """Resets the entire application to its initial state."""
        self._reset_plot()
        self.loaded_signals.clear()
        self.freq_ops.current_spectrum = None
        self.freq_ops.original_max_amplitude = 1.0

        self.gen_params["Amplitude (A)"].set("1.0")
        self.gen_params["Frequency (Hz)"].set("5.0")
        self.gen_params["Phase (rad)"].set("0.0")
        self.gen_params["Sampling Freq (Hz)"].set("100.0")
        self.gen_type.set("sine")
        self.mult_factor.set("2.0")
        self.norm_range.set("0_1")
        self.quant_value.set("16")
        self.quant_param.set("levels")

        self._update_status("Welcome! Load a signal or generate one to begin.")

    def _get_primary_signal(self):
        """Helper to get the main loaded signal, showing an error if none exists."""
        if 'main' not in self.loaded_signals:
            messagebox.showinfo("No Signal", "Please load or generate a signal first to perform this operation.")
            return None
        return self.loaded_signals['main']

    def _load_single_signal(self):
        filepath = filedialog.askopenfilename(title="Select a Signal File",
                                              filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not filepath:
            return

        signal_data = load_signal_from_path(filepath)
        if signal_data:
            indices, amplitudes = signal_data

            if len(indices) == 0 or len(amplitudes) == 0:
                messagebox.showerror("Data Error", "Loaded signal contains no data points.")
                return

            if len(indices) != len(amplitudes):
                messagebox.showerror("Data Error",
                                     f"Loaded signal has mismatched array lengths: "
                                     f"indices({len(indices)}) != amplitudes({len(amplitudes)})")
                return

            self.loaded_signals['main'] = signal_data
            self._plot_data([(indices, amplitudes, "Loaded Signal")], "Signal Viewer")
            self._update_status(f"Successfully loaded and displayed '{filepath.split('/')[-1]}'.")

    def _compare_two_signals(self):
        f1 = filedialog.askopenfilename(title="Select First Signal", filetypes=[("Text Files", "*.txt")])
        if not f1: return
        f2 = filedialog.askopenfilename(title="Select Second Signal", filetypes=[("Text Files", "*.txt")])
        if not f2: return
        s1_data = load_signal_from_path(f1)
        s2_data = load_signal_from_path(f2)

        if s1_data and s2_data:
            plots = [
                (s1_data[0], s1_data[1], f"Signal 1: {f1.split('/')[-1]}"),
                (s2_data[0], s2_data[1], f"Signal 2: {f2.split('/')[-1]}")
            ]
            self._plot_data(plots, "Signal Comparison")
            self._update_status("Successfully displayed two signals for comparison.")

            s1_amps = s1_data[1]
            s2_amps = s2_data[1]

            if len(s1_amps) != len(s2_amps):
                messagebox.showinfo("Comparison Result",
                                    "Signals have different lengths and cannot be compared sample-by-sample.")
                return

            tolerance = simpledialog.askfloat("Comparison Tolerance",
                                              "Enter the tolerance for comparison (e.g., 0.001):",
                                              initialvalue=0.001)

            if tolerance is None:
                return

            are_equal = True
            max_diff = 0
            max_diff_index = 0

            for i in range(len(s1_amps)):
                diff = abs(s1_amps[i] - s2_amps[i])
                if diff > tolerance:
                    are_equal = False
                    max_diff = diff
                    max_diff_index = i
                    break

            if are_equal:
                messagebox.showinfo("Comparison Result",
                                    f"Test PASSED: Signal amplitudes are equal (within {tolerance} tolerance).")
            else:
                messagebox.showerror("Comparison Result",
                                     f"Test FAILED: Signal amplitudes are NOT equal.\n\n"
                                     f"First failure at index {max_diff_index}:\n\n"
                                     f"Signal 1: {s1_amps[max_diff_index]:.6f}\n"
                                     f"Signal 2: {s2_amps[max_diff_index]:.6f}\n\n"
                                     f"Difference: {max_diff:.6f} (which is > {tolerance})")

    def _generate_signal(self):
        try:
            A = float(self.gen_params["Amplitude (A)"].get())
            f = float(self.gen_params["Frequency (Hz)"].get())
            theta = float(self.gen_params["Phase (rad)"].get())
            fs = float(self.gen_params["Sampling Freq (Hz)"].get())
        except ValueError:
            messagebox.showerror("Invalid Input", "All generation parameters must be valid numbers.")
            return

        if fs <= 0:
            messagebox.showerror("Invalid Input", "Sampling Frequency must be a positive number.")
            return
        if f < 0:
            messagebox.showerror("Invalid Input", "Frequency must be non-negative.")
            return

        if fs < 2 * f:
            messagebox.showwarning("Nyquist Warning",
                                   f"Sampling frequency ({fs}Hz) should be at least 2x the analog frequency ({f}Hz) to avoid aliasing.")

        # Generate 3 cycles or 1 second, whichever is longer
        duration = max(1.0, 3.0 / f if f > 0 else 1.0)
        t = np.arange(0, duration, 1 / fs)

        if len(t) == 0:
            messagebox.showerror("Invalid Input", "Resulting signal has 0 samples. Check parameters (e.g., Fs > 0).")
            return

        is_cosine = self.gen_type.get() == "cosine"
        wave_type = "Cosine" if is_cosine else "Sine"

        x = A * np.cos(2 * np.pi * f * t + theta) if is_cosine else A * np.sin(2 * np.pi * f * t + theta)

        self.loaded_signals['main'] = (t, x)
        self._plot_data([(t, x, f"Generated {wave_type} Wave")], f"Generated {wave_type} Wave")
        self._update_status(f"Generated and displayed a {wave_type} wave.")

    def _add_signals(self):
        filepaths = filedialog.askopenfilenames(title="Select 2 or more signals to add")
        if len(filepaths) < 2:
            self._update_status("Addition cancelled. Please select at least two signals.")
            return

        signals_data = [load_signal_from_path(fp) for fp in filepaths]
        if not all(signals_data):
            self._update_status("Could not load all signals. Aborting addition.")
            return

        amplitudes = [s[1] for s in signals_data]
        max_len = max(len(a) for a in amplitudes)
        padded_signals = [np.pad(a, (0, max_len - len(a)), 'constant') for a in amplitudes]

        resultant_signal = np.sum(padded_signals, axis=0)
        resultant_indices = np.arange(max_len)  # Use simple indices for mixed-length signals
        self.loaded_signals['main'] = (resultant_indices, resultant_signal)
        self._plot_data([
            (resultant_indices, resultant_signal, "Resultant Signal")
        ], "Result of Signal Addition")
        self._update_status(f"Successfully added {len(filepaths)} signals.")

    def _subtract_signals(self):
        f1 = filedialog.askopenfilename(title="Select First Signal (A)")
        if not f1: return
        f2 = filedialog.askopenfilename(title="Select Second Signal (B)")
        if not f2: return

        s1_data = load_signal_from_path(f1)
        s2_data = load_signal_from_path(f2)
        if not s1_data or not s2_data: return

        a1, a2 = s1_data[1], s2_data[1]
        max_len = max(len(a1), len(a2))
        a1 = np.pad(a1, (0, max_len - len(a1)), 'constant')
        a2 = np.pad(a2, (0, max_len - len(a2)), 'constant')

        result = a1 - a2
        result_indices = np.arange(max_len)  # Use simple indices
        self.loaded_signals['main'] = (result_indices, result)
        self._plot_data([(result_indices, result, "Result (A - B)")], "Result of Signal Subtraction")
        self._update_status("Subtraction complete.")

    def _multiply_signal(self):
        original_signal = self._get_primary_signal()
        if not original_signal: return

        try:
            factor = float(self.mult_factor.get())
        except ValueError:
            messagebox.showerror("Invalid Factor", "The multiplication factor must be a valid number.")
            return

        indices, amplitudes = original_signal
        modified_amplitudes = amplitudes * factor
        description = get_multiplication_description(factor)
        self.loaded_signals['main'] = (indices, modified_amplitudes)

        self._plot_data([
            (indices, amplitudes, "Original Signal"),
            (indices, modified_amplitudes, f"{description} Signal (x{factor})")
        ], "Signal Multiplication")
        self._update_status(f"Signal multiplied by {factor}.")

    def _square_signal(self):
        original_signal = self._get_primary_signal()
        if not original_signal: return

        indices, amplitudes = original_signal
        result = amplitudes ** 2
        self.loaded_signals['main'] = (indices, result)
        self._plot_data([
            (indices, amplitudes, "Original"),
            (indices, result, "Squared")
        ], "Signal Squaring")
        self._update_status("Signal squaring complete.")

    def _normalize_signal(self):
        original_signal = self._get_primary_signal()
        if not original_signal: return

        indices, amplitudes = original_signal
        min_amp, max_amp = np.min(amplitudes), np.max(amplitudes)

        if np.isclose(min_amp, max_amp):
            messagebox.showinfo("Normalization Info", "Cannot normalize a signal with constant amplitude.")
            return

        if self.norm_range.get() == "-1_1":
            result = 2 * (amplitudes - min_amp) / (max_amp - min_amp) - 1
            label = "Normalized to [-1, 1]"
        else:
            result = (amplitudes - min_amp) / (max_amp - min_amp)
            label = "Normalized to [0, 1]"

        self.loaded_signals['main'] = (indices, result)
        self._plot_data([
            (indices, amplitudes, "Original"),
            (indices, result, label)
        ], "Signal Normalization")
        self._update_status(f"Normalization to {label.split(' ')[-1]} complete.")

    def _accumulate_signal(self):
        original_signal = self._get_primary_signal()
        if not original_signal: return

        indices, amplitudes = original_signal
        result = np.cumsum(amplitudes)
        self.loaded_signals['main'] = (indices, result)
        self._plot_data([
            (indices, amplitudes, "Original"),
            (indices, result, "Accumulated")
        ], "Signal Accumulation (Cumulative Sum)")
        self._update_status("Signal accumulation complete.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessingGUI(root)
    root.mainloop()