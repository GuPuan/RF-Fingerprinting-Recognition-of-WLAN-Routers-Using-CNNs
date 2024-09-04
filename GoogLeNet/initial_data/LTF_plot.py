import numpy as np
import matplotlib.pyplot as plt

# Provided complex vector
original_vector = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    (1.4719601443879746+1.4719601443879746j), 0.0, 0.0, 0.0,
    (-1.4719601443879746-1.4719601443879746j), 0.0, 0.0, 0.0,
    (1.4719601443879746+1.4719601443879746j), 0.0, 0.0, 0.0,
    (-1.4719601443879746-1.4719601443879746j), 0.0, 0.0, 0.0,
    (-1.4719601443879746-1.4719601443879746j), 0.0, 0.0, 0.0,
    (1.4719601443879746+1.4719601443879746j), 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, (-1.4719601443879746-1.4719601443879746j), 0.0, 0.0, 0.0,
    (-1.4719601443879746-1.4719601443879746j), 0.0, 0.0, 0.0,
    (1.4719601443879746+1.4719601443879746j), 0.0, 0.0, 0.0,
    (1.4719601443879746+1.4719601443879746j), 0.0, 0.0, 0.0,
    (1.4719601443879746+1.4719601443879746j), 0.0, 0.0, 0.0,
    (1.4719601443879746+1.4719601443879746j), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# Compute the 64-point inverse Fourier transform (IFFT)
ifft_result = np.fft.ifft(original_vector, 64)

plt.figure(figsize=(12, 8))

# Plot the real and imaginary parts of the original complex vector
plt.subplot(2, 1, 1)
plt.stem(np.real(original_vector), markerfmt='bo', linefmt='b-', basefmt='-', label='Real Part')
imag_stem1 = plt.stem(np.imag(original_vector), markerfmt='rx', linefmt='r-', basefmt='-', label='Imaginary Part')
plt.setp(imag_stem1.markerline, markersize=8)  # Set the size of the red crosses

plt.xlabel('Index', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)

# Plot the real and imaginary parts after IFFT
plt.subplot(2, 1, 2)
plt.stem(np.real(ifft_result), markerfmt='bo', linefmt='b-', basefmt='-', label='Real Part')
imag_stem2 = plt.stem(np.imag(ifft_result), markerfmt='rx', linefmt='r-', basefmt='-', label='Imaginary Part')
plt.setp(imag_stem2.markerline, markersize=8)  # Set the size of the red crosses

plt.xlabel('Sample Index', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
plt.tight_layout(rect=[0, 0, 0.4, 1])  # Adjust layout to make space for the legend
plt.show()
