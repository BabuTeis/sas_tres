import math
import cmath

def pad_to_power_of_two(sequence):
    """Pad the input sequence with zeros to the nearest power of 2."""
    n = len(sequence)
    next_power_of_two = 2**math.ceil(math.log2(n))
    return sequence + [0] * (next_power_of_two - n)

def fft_recursive(x):
    """Recursive implementation of the FFT."""
    n = len(x)
    if n == 1:
        return x

    # Split the input into even and odd indexed elements
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])

    # Compute the FFT combining step
    combined = [0] * n
    for k in range(n // 2):
        twiddle_factor = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        combined[k] = even[k] + twiddle_factor
        combined[k + n // 2] = even[k] - twiddle_factor

    return combined

def format_fft_output(fft_result):
    """Format the FFT result to have two decimal places."""
    formatted = [f"{z.real:.2f}+j{z.imag:.2f}" if z.imag >= 0 else f"{z.real:.2f}-j{-z.imag:.2f}" for z in fft_result]
    return formatted

def parse_input(input_str):
    """Parse input in the format 'N: [x1,x2,x3,...]'"""
    n, sequence = input_str.split(":")
    sequence = list(map(int, sequence.strip().strip('[]').split(',')))
    return sequence

def compute_fft_from_input(input_str):
    sequence = parse_input(input_str)
    sequence = pad_to_power_of_two(sequence)
    fft_result = fft_recursive(sequence)
    return format_fft_output(fft_result)

input_str = input()
output = compute_fft_from_input(input_str)
for value in output:
    print(value)
