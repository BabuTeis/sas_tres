# This code was rewritten to avoid using numpy because Themis gave an error
# when trying to import numpy. Instead, it uses built-in Python functionality.

import cmath
from typing import List, Tuple


# -- Functions --

def read_signal() -> Tuple[List[complex], int]:
    """
    Reads the input signal and its length.
    Input format: "length: [values]"
    :return: A tuple (signal, length), where signal is a list of integers and
    length is the number of elements.
    """
    input_line = input().strip()
    length = int(input_line.split(':')[0])
    signal = list(map(complex,
                      input_line.split(':')[1].strip()[1:-1].split(',')))
    return signal, length


def create_vandermonde_matrix(length: int,
                              inverse: bool = False) -> List[List[complex]]:
    """
    Creates a Vandermonde matrix for the given signal length.
    :param length: The size of the matrix (length x length).
    :param inverse: If True, creates the matrix for the IDFT.
    :return: A complex Vandermonde matrix as a list of lists.
    """
    # Initialize matrix
    vandermonde_matrix = [[0 for _ in range(length)] for _ in range(length)]
    # Compute root of unity
    omega = cmath.exp((2j if inverse else -2j) * cmath.pi / length)
    for k in range(length):
        for n in range(length):
            vandermonde_matrix[k][n] = omega ** (k * n)  # Populate matrix
    return vandermonde_matrix


def compute_transform(vandermonde_matrix: List[List[complex]],
                      signal: List[complex],
                      inverse: bool = False) -> List[complex]:
    """
    Computes the Fourier Transform (DFT or IDFT) of the signal using
    the Vandermonde matrix.
    :param vandermonde_matrix: The Vandermonde matrix as a list of lists.
    :param signal: The input signal (list of complex numbers).
    :param inverse: If True, computes the IDFT. Otherwise, computes the DFT.
    :return: The Transform (DFT/IDFT) as a list of complex numbers.
    """
    length = len(signal)
    transform_result = [0] * length  # Initialize result list
    for k in range(length):
        # Compute the dot product of row k of the Vandermonde
        # matrix and the signal
        transform_result[k] = sum(
            vandermonde_matrix[k][n] * signal[n] for n in range(length))
        if inverse:
            transform_result[k] /= length  # Normalize for IDFT
    return transform_result


def format_output(transform_result: List[complex]) -> List[str]:
    """
    Formats the transform result to always show 2 decimal places for both real
    and imaginary parts.
    :param transform_result: The computed Transform (list of complex numbers).
    :return: A list of formatted strings for each frequency component.
    """
    formatted_output = []
    for value in transform_result:
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
    Prints the formatted Transform result.
    :param formatted_output: A list of formatted strings representing
    the Transform result.
    """
    for output in formatted_output:
        print(output)


# -- Main Execution --

# Step 1: Read the signal
signal, length = read_signal()

# Step 2: Create the Vandermonde matrix for the IDFT
vandermonde_matrix = create_vandermonde_matrix(length, inverse=True)

# Step 3: Compute the IDFT
idft_result = compute_transform(vandermonde_matrix, signal, inverse=True)

# Step 4: Format the output
formatted_output = format_output(idft_result)

# Step 5: Print the formatted output
print_formatted_output(formatted_output)
