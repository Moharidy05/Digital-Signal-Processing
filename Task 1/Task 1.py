import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Data Loading and Utility Functions
# ==============================================================================

def load_signal(filepath: str | None = None, prompt: str = "Enter the path of the signal file: ") -> tuple[np.ndarray, np.ndarray] | None:
    """Prompts for a signal file path (if not provided) and loads signal data."""
    if not filepath:
        filepath = input(prompt).strip()

    indices, amplitudes = [], []
    try:
        with open(filepath, 'r') as f:
            for _ in range(3):
                next(f, None)
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        indices.append(float(parts[0]))
                        amplitudes.append(float(parts[1]))
                    except ValueError:
                        continue

        if not indices:
            print(f"No valid signal data found in '{filepath}'.")
            return None

        return np.array(indices), np.array(amplitudes)

    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
    return None


def _get_multiplication_description(factor: float) -> str:
    """Gets a user-friendly description of the multiplication operation."""
    if factor == -1:
        return "Inverted"
    if factor < 0:
        return "Inverted and Scaled"
    if 0 <= factor < 1:
        return "Reduced"
    return "Amplified"


# ==============================================================================
# 2. Plotting Functions
# ==============================================================================

def display_signal(indices: np.ndarray, amplitudes: np.ndarray, title: str):
    """Displays a signal in both continuous and discrete representations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    ax1.plot(indices, amplitudes, 'b-')
    ax1.set_title("Continuous Representation")
    ax1.set_xlabel("Time / Index")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    ax2.stem(indices, amplitudes, basefmt="r-")
    ax2.set_title("Discrete Representation")
    ax2.set_xlabel("Time / Index")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def display_two_signals(
    signal1: tuple[np.ndarray, np.ndarray],
    signal2: tuple[np.ndarray, np.ndarray],
    label1: str,
    label2: str,
    title: str
):
    """Displays two signals on the same plot for comparison."""
    indices1, amplitudes1 = signal1
    indices2, amplitudes2 = signal2

    plt.figure(figsize=(12, 6))
    plt.plot(indices1, amplitudes1, 'b-', label=label1)
    plt.plot(indices2, amplitudes2, 'g--', label=label2)
    plt.title(title, fontsize=16)
    plt.xlabel("Time / Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==============================================================================
# 3. Signal Operation Functions
# ==============================================================================

def add_signals():
    """Prompts for multiple signals, adds them, and displays the result."""
    try:
        num_signals = int(input("How many signals do you want to add? "))
        if num_signals < 2:
            print("You need at least two signals to perform addition.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    signals: list[np.ndarray] = []
    for i in range(num_signals):
        signal_data = load_signal(prompt=f"Enter path for signal #{i + 1}: ")
        if signal_data:
            _, amplitudes = signal_data
            signals.append(amplitudes)

    if len(signals) != num_signals:
        print("Could not load all signals. Aborting addition.")
        return

    max_len = max(len(s) for s in signals)
    padded_signals = [np.pad(s, (0, max_len - len(s)), 'constant') for s in signals]

    resultant_signal = np.sum(padded_signals, axis=0)
    resultant_indices = np.arange(max_len)

    print("Signals added successfully!")
    display_signal(resultant_indices, resultant_signal, "Result of Signal Addition")


def multiply_signal():
    """Multiplies a signal by a constant factor and displays the result."""
    signal_data = load_signal()
    if not signal_data:
        return

    try:
        factor = float(input("Enter the constant value to multiply by: "))
    except ValueError:
        print("Invalid constant. Please enter a number.")
        return

    indices, original_amplitudes = signal_data
    modified_amplitudes = original_amplitudes * factor
    description = _get_multiplication_description(factor)
    print(f"Signal multiplied by {factor}.")

    display_two_signals(
        (indices, original_amplitudes),
        (indices, modified_amplitudes),
        "Original Signal",
        f"{description} Signal (x{factor})",
        "Signal Multiplication"
    )


# ==============================================================================
# 4. Menu Functions
# ==============================================================================

def _handle_load_and_display():
    """Action for loading and displaying a single signal."""
    signal_data = load_signal()
    if signal_data:
        indices, amplitudes = signal_data
        display_signal(indices, amplitudes, "Signal Data")


def _handle_compare_signals():
    """Action for comparing two signals."""
    print("--- Compare Two Signals ---")
    signal1 = load_signal(prompt="Enter path for the first signal: ")
    if not signal1:
        return
    signal2 = load_signal(prompt="Enter path for the second signal: ")
    if not signal2:
        return

    display_two_signals(signal1, signal2, "Signal 1", "Signal 2", "Signal Comparison")


def arithmetic_operations_menu():
    """Handles the arithmetic operations sub-menu."""
    menu_actions = {
        '1': add_signals,
        '2': multiply_signal,
    }
    while True:
        print("\n--- Arithmetic Operations ---")
        print("  1. Addition (Add multiple signals)")
        print("  2. Multiplication (Amplify/Reduce signal)")
        print("  3. Back to Main Menu")
        choice = input("Enter your choice: ")

        if choice == '3':
            break
        action = menu_actions.get(choice)
        if action:
            action()
        else:
            print("Invalid choice, please try again.")


def main():
    """Main function to run the framework menu."""
    menu_actions = {
        '1': _handle_load_and_display,
        '2': arithmetic_operations_menu,
        '3': _handle_compare_signals,
    }
    while True:
        print("\n======= Signal Processing Framework =======")
        print("1. Load and Display a Signal")
        print("2. Arithmetic Operations")
        print("3. Compare Two Signals")
        print("4. Exit")
        print("========================================")
        choice = input("Enter your choice: ")

        if choice == '4':
            print("Exiting framework. Goodbye!")
            break
        action = menu_actions.get(choice)
        if action:
            action()
        else:
            print("Invalid choice, please enter a number from 1 to 4.")


if __name__ == "__main__":
    main()