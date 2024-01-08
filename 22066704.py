import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Read the Data
data = pd.read_csv('data4-2_1575043522.csv', header=None, names=['salary'])


# Step 2: Create a Probability Density Function (PDF) and Plot Histogram
mu, std = norm.fit(data['salary'], loc=0)
print("Mean Value Intially")
print(mu)
print("Standard Deviation")
print(std)

# Reduce the figure size
plt.figure(figsize=(10, 5))

# Set the number of bins for the histogram
num_bins = 30

# Plot the histogram
plt.hist(data['salary'], bins=num_bins, density=True, alpha=0.6, color='g', label='Actual Data')

# Plot the PDF
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fit Results (PDF)')

# Calculate the mean annual salary using the obtained PDF
mean_salary = np.trapz(x * p, x)

# Calculate the value of X for 25% of people below X
X = norm.ppf(0.25, mu, std)

# Add labels and title
plt.title('Salary Distribution')
plt.xlabel('Salary (Euros)')
plt.ylabel('Probability')

# Add a legend at the top right corner
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)

# Create a separate box to display values of W and X below the actual data box with colors
box_text_w = f'Mean Salary (\u0303W): {mean_salary:.2f} Euros'
box_text_x = f'Value of X (25% below): {X:.2f} Euros'

plt.text(0.98, 0.80, box_text_w, transform=plt.gca().transAxes,
         color='blue', verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))

plt.text(0.98, 0.70, box_text_x, transform=plt.gca().transAxes,
         color='green', verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))

# Display the plot in a separate window
plt.show()





