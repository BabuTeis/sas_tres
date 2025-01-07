import math
from typing import List, Tuple


def read_signal() -> Tuple[List[int], int]:
    """
    Reads a signal in the format 'length: [values]' from input.
    :return: A tuple containing the signal (as a list of integers) and
    its length.
    """
    input_line = input().strip()
    length = int(input_line.split(':')[0])
    signal = list(map(int, input_line.split(':')[1].strip()[1:-1].split(',')))
    return signal, length


def print_signal(length: int, signal: List[float]) -> None:
    """
    Prints the signal in the format 'length: [values]', with values formatted
    to 5 decimal places.
    :param length: The length of the signal.
    :param signal: The signal values.
    """
    formatted_signal = [f"{value:.5f}" for value in signal]
    print(f"{length}: [{', '.join(formatted_signal)}]")


def compute_mean(signal: List[int]) -> float:
    """
    Computes the mean of a signal.
    :param signal: A list of integers representing the signal.
    :return: The mean of the signal.
    """
    return sum(signal) / len(signal)


def compute_deviations(signal: List[int], mean: float) -> List[float]:
    """
    Computes deviations from the mean for a signal.
    :param signal: A list of integers representing the signal.
    :param mean: The mean of the signal.
    :return: A list of deviations from the mean.
    """
    return [value - mean for value in signal]


def compute_variance(deviations: List[float]) -> float:
    """
    Computes the variance of a signal based on deviations from the mean.
    :param deviations: A list of deviations from the mean.
    :return: The variance of the signal.
    """
    return sum(dev ** 2 for dev in deviations)


def compute_standard_deviation(variance: float) -> float:
    """
    Computes the standard deviation of a signal.
    :param variance: The variance of the signal.
    :return: The standard deviation.
    """
    return math.sqrt(variance)


def compute_pearson_correlation(
    template_deviations: List[float],
    window_deviations: List[float],
    template_std_dev: float,
    window_std_dev: float
) -> float:
    """
    Computes the Pearson correlation between the template and a
    window in the signal.
    :param template_deviations: Deviations of the template from its mean.
    :param window_deviations: Deviations of the signal window from its mean.
    :param template_std_dev: Standard deviation of the template.
    :param window_std_dev: Standard deviation of the signal window.
    :return: The Pearson correlation coefficient.
    """
    if template_std_dev > 0 and window_std_dev > 0:
        cross_product_sum = sum(
            t_dev * w_dev for t_dev, w_dev in zip(template_deviations,
                                                  window_deviations)
        )
        return cross_product_sum / (template_std_dev * window_std_dev)
    return 0.0  # Default to 0 if standard deviation is zero


def process_sliding_window(
    template_signal: List[int],
    input_signal: List[int]
) -> Tuple[int, List[float]]:
    """
    Processes the input signal with the template signal using sliding window
    and Pearson correlation.
    :param template_signal: The template signal as a list of integers.
    :param input_signal: The input signal as a list of integers.
    :return: The output length and the list of
    Pearson correlation coefficients.
    """
    template_length = len(template_signal)
    input_length = len(input_signal)
    output_length = input_length - template_length + 1
    output_signal = []

    # Precompute template statistics
    template_mean = compute_mean(template_signal)
    template_deviations = compute_deviations(template_signal, template_mean)
    template_variance = compute_variance(template_deviations)
    template_std_dev = compute_standard_deviation(template_variance)

    # Sliding window
    for i in range(output_length):
        # Extract current window
        current_window = input_signal[i:i + template_length]

        # Compute window statistics
        window_mean = compute_mean(current_window)
        window_deviations = compute_deviations(current_window, window_mean)
        window_variance = compute_variance(window_deviations)
        window_std_dev = compute_standard_deviation(window_variance)

        # Compute Pearson correlation
        correlation = compute_pearson_correlation(
            template_deviations, window_deviations, template_std_dev,
            window_std_dev
        )
        output_signal.append(round(correlation, 5))
        # Round to 5 decimal places

    return output_length, output_signal


# --- Full Execution --- #

# Read template and input signals
template_signal, template_length = read_signal()
input_signal, input_length = read_signal()

# Process sliding window and compute output
output_length, output_signal = process_sliding_window(template_signal,
                                                      input_signal)

# Print the final output
print_signal(output_length, output_signal)
