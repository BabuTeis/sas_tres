# This code was rewritten to avoid using numpy because Themis gave an error
# when trying to import numpy. Instead, it uses built-in Python functionality.

import cmath
from typing import List, Tuple


# -- Functions --

def read_signal() -> Tuple[List[int], int]:
    """
    Reads the input signal and its length.
    Input format: "length: [values]"
    :return: A tuple (signal, length), where signal is a list of integers and
    length is the number of elements.
    """
    input_line = input().strip()
    length = int(input_line.split(':')[0])
    signal = list(map(int, input_line.split(':')[1].strip()[1:-1].split(',')))
    return signal, length


def create_vandermonde_matrix(length: int) -> List[List[complex]]:
    """
    Creates a Vandermonde matrix for the given signal length.
    :param length: The size of the matrix (length x length).
    :return: A complex Vandermonde matrix as a list of lists.
    """
    # Initialize matrix
    vandermonde_matrix = [[0 for _ in range(length)] for _ in range(length)]
    omega = cmath.exp(-2j * cmath.pi / length)  # Compute root of unity
    for k in range(length):
        for n in range(length):
            vandermonde_matrix[k][n] = omega ** (k * n)  # Populate matrix
    return vandermonde_matrix


def compute_dft(vandermonde_matrix: List[List[complex]],
                signal: List[int]) -> List[complex]:
    """
    Computes the Discrete Fourier Transform (DFT) of the signal using
    the Vandermonde matrix.
    :param vandermonde_matrix: The Vandermonde matrix as a list of lists.
    :param signal: The input signal (list of integers).
    :return: The DFT as a list of complex numbers.
    """
    length = len(signal)
    dft_result = [0] * length  # Initialize result list
    for k in range(length):
        # Compute the dot product of row k of the Vandermonde
        # matrix and the signal
        dft_result[k] = sum(
            vandermonde_matrix[k][n] * signal[n] for n in range(length))
    return dft_result


def format_output(dft_result: List[complex]) -> List[str]:
    """
    Formats the DFT result to always show 2 decimal places for both real
    and imaginary parts.
    :param dft_result: The computed DFT (list of complex numbers).
    :return: A list of formatted strings for each frequency component.
    """
    formatted_output = []
    for value in dft_result:
        real_part = f"{value.real:.2f}"  # Format real part
        imag_part = f"{value.imag:.2f}"  # Format imaginary part
        if value.imag >= 0:
            # Positive imaginary part
            formatted_output.append(f"{real_part}+j{imag_part}")
        else:
            # Negative imaginary part
            formatted_output.append(f"{real_part}-j{-float(imag_part):.2f}")
    return formatted_output


def print_formatted_output(formatted_output: List[str]) -> None:
    """
    Prints the formatted DFT result.
    :param formatted_output: A list of formatted strings representing
    the DFT result.
    """
    for output in formatted_output:
        print(output)


# -- Main Execution --

# Step 1: Read the signal
signal, length = read_signal()

# Step 2: Create the Vandermonde matrix
vandermonde_matrix = create_vandermonde_matrix(length)

# Step 3: Compute the DFT
dft_result = compute_dft(vandermonde_matrix, signal)

# Step 4: Format the output
formatted_output = format_output(dft_result)

# Step 5: Print the formatted output
print_formatted_output(formatted_output)
