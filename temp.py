import numpy as np
import matplotlib.pyplot as plt

# Define the function with k = 0.2
k = 0.05
def y_function(x):
    return -(1 - np.exp(-k * x))

# Generate x values
x_values = np.linspace(0, 50, 1000)  # From 0 to 20 with 1000 points

# Calculate corresponding y values
y_values = y_function(x_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, 'b-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('x', fontsize=12)
plt.ylabel('y = 1 - exp(-{}x)'.format(k), fontsize=12)
plt.title('Plot of y = 1 - exp(-{}x)'.format(k), fontsize=14)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)  # Horizontal asymptote at y=1
plt.xlim(0, 50)
plt.ylim(0, -1.1)

# Display the plot
plt.tight_layout()
plt.show()