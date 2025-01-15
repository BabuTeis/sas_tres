from math import cos, sin

def fir_filter_impulse_response_no_numpy(M, angles):
    # Convert angles to roots
    roots = [(cos(angle) + 1j * sin(angle)) for angle in angles]

    # Initialize coefficients as [1]
    coefficients = [1]

    # Multiply the factors (1 - z_k * z^-1) to find the polynomial
    for root in roots:
        new_coefficients = [0] * (len(coefficients) + 1)
        for i in range(len(coefficients)):
            # Multiply each coefficient by (1 - root*z^-1)
            new_coefficients[i] += coefficients[i]  # Current coefficient
            new_coefficients[i + 1] -= coefficients[i] * root  # Contribution from -root
        coefficients = new_coefficients

    # Convert coefficients to real values (imaginary parts should cancel)
    coefficients = [round(c.real, 2) for c in coefficients]
    formatted_coefficients = ",".join(f"{c:.2f}" for c in coefficients)
    return f"{len(coefficients)}: [{formatted_coefficients}]"

M = int(input())  # First line: Number of roots
angles = list(map(float, input().strip().split()))
# Generate outputs for each test case
output = fir_filter_impulse_response_no_numpy(M, angles)
print(output)