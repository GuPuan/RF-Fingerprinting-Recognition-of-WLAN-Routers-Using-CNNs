import matplotlib.pyplot as plt

# Data
x = [0, 1, 2, 3, 4]  # X-axis labels corresponding to different layers
n = 0.9833
b1a = 0.9847
p1a = 0.9815
b2a = 0.9861
p2a = 0.9810
y = [n, b1a, p1a, b2a, p2a]  # Average accuracy values for each layer

# Accuracy measurements for different runs or experiments
n = [0.9827272727272728, 0.9863636363636363, 0.9872727272727273, 0.98, 0.980909090909091,
      0.980909090909091, 0.9772727272727273, 0.97, 0.9790909090909091, 0.9827272727272728,
      0.9827272727272728, 0.9836363636363636, 0.9845454545454545, 0.99, 0.9890909090909091,
      0.9890909090909091, 0.98, 0.98, 0.9863636363636363, 0.9872727272727273]
b1 = [0.9890909090909091, 0.9854545454545455, 0.9790909090909091, 0.9818181818181818,
      0.9854545454545455, 0.99, 0.9863636363636363, 0.9872727272727273, 0.9772727272727273,
      0.9863636363636363, 0.9827272727272728, 0.990909090909091, 0.9772727272727273,
      0.9890909090909091, 0.980909090909091, 0.9881818181818182, 0.98, 0.9827272727272728,
      0.9863636363636363, 0.9872727272727273]
p1 = [0.9827272727272728, 0.9836363636363636, 0.9854545454545455, 0.9827272727272728,
      0.9754545454545455, 0.9863636363636363, 0.9845454545454545, 0.9827272727272728,
      0.9836363636363636, 0.9818181818181818, 0.9781818181818182, 0.9818181818181818,
      0.9781818181818182, 0.9836363636363636, 0.9845454545454545, 0.9790909090909091,
      0.9781818181818182, 0.9763636363636363, 0.9836363636363636, 0.9727272727272728]
b2 = [0.9836363636363636, 0.9918181818181818, 0.9872727272727273, 0.9927272727272727,
      0.99, 0.9827272727272728, 0.9881818181818182, 0.9881818181818182, 0.9818181818181818,
      0.9818181818181818, 0.990909090909091, 0.99, 0.9872727272727273, 0.980909090909091,
      0.9863636363636363, 0.980909090909091, 0.98, 0.9890909090909091, 0.9745454545454545,
      0.9890909090909091]
p2 = [0.98, 0.9763636363636363, 0.9818181818181818, 0.9736363636363636, 0.9790909090909091,
      0.9827272727272728, 0.9772727272727273, 0.9845454545454545, 0.9872727272727273,
      0.9790909090909091, 0.980909090909091, 0.9854545454545455, 0.9763636363636363,
      0.9818181818181818, 0.9872727272727273, 0.9818181818181818, 0.9818181818181818,
      0.9818181818181818, 0.9818181818181818, 0.9772727272727273]

# List of all data sets for easier iteration
data = [n, b1, p1, b2, p2]

# Set the figure size and adjust the margins
plt.figure(figsize=(6, 4))  # Reduced width for a more compact display
plt.subplots_adjust(left=0.2, right=0.5, top=0.8, bottom=0.3)  # Adjusted margins

# Plot scatter plots for each data set
for i, b in enumerate(data):
    max_value = max(b)
    min_value = min(b)
    for value in b:
        color = 'black' if value == max_value or value == min_value else 'red'  # Highlight min/max values
        plt.scatter(i, value, color=color, marker='x', s=100)  # Plot scatter with size 100

# Plot the line graph for the average values
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Average Value')

# Add a legend to describe the markers
plt.scatter([], [], color='red', marker='x', label='Measured Values')
plt.scatter([], [], color='black', marker='x', label='Measured Min/Max')

# Add title and axis labels
plt.xlabel('Layer', fontsize=24)  # X-axis label
plt.ylabel('Average Accuracy', fontsize=24)  # Y-axis label

# Custom labels for the x-axis
labels = ['N$_{L}$', 'C$_{L}^{1}$', 'P$_{L}^{1}$', 'C$_{L}^{2}$', 'P$_{L}^{2}$']
x = [0, 1, 2, 3, 4]
plt.xticks(x, labels, fontsize=24)  # Set x-axis ticks and labels

# Set y-axis tick format and fontsize
plt.yticks(fontsize=24)

# Limit the y-axis range for a more compact data display
plt.ylim(min(min(b) for b in data) - 0.001, max(max(b) for b in data) + 0.001)  # Tighten y-axis range

# Display the legend
plt.legend(fontsize=20)

# Show the plot
plt.show()
