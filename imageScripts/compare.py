import numpy as np  
import matplotlib.pyplot as plt 

# Define discrete classes
x = ['PC CPU', 'Raspberry Pi', 'Hailo-8L']
models = ['RN50', 'RN50x4', 'RN101', 'TinyCLIP-19M', 'TinyCLIP-30M']

# Update font sizes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,  # Default text size
    "axes.titlesize": 16,  # Title size
    "axes.labelsize": 14,  # X and Y label size
    "xtick.labelsize": 12,  # X tick label size
    "ytick.labelsize": 12,  # Y tick label size
    "legend.fontsize": 12,  # Legend font size
    "figure.titlesize": 16  # Figure title size
})

# Performance data for each model in different environments
performance_data_accuracy = {
    'RN50': [0.845, 0.845, 0.799],
    'RN50x4': [0.937, 0.937, 0.845],
    'RN101': [0.882, 0.882, 0.743],
    'TinyCLIP-19M': [0.882, 0.882, 0.141],
    'TinyCLIP-30M': [0.839, 0.839, 0.624]
}

# Create the first plot
plt.figure(figsize=(6,6))
for model in models:
    plt.plot(x, performance_data_accuracy[model], label=model)
plt.xlabel('Environment')
plt.ylabel('Accuracy')
plt.ylim(-0.1,1.1)
plt.legend()
plt.grid(True)
plt.savefig('AccuracyModels.png',dpi= 600)

performance_data_balanced_accuracy = {
    'RN50': [0.596, 0.596, 0.200],
    'RN50x4': [0.856, 0.856, 0.560],
    'RN101': [0.706, 0.706, 0.507],
    'TinyCLIP-19M': [0.830, 0.830, 0.211],
    'TinyCLIP-30M': [0.822, 0.822, 0.521]
}

# Create the second plot
plt.figure(figsize=(6,6))
for model in models:
    plt.plot(x, performance_data_balanced_accuracy[model], label=model)
plt.xlabel('Environment')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.ylim(-0.1,1.1)
plt.grid(True)
plt.savefig('BalancedAccuracyModels.png',dpi= 600)


performance_data_throughput = {
    'RN50': [12.46, 1.15, 20.63],
    'RN50x4':[5.79, 0.61, 10.13],
    'RN101': [9.38, 0.92, 15.61],
    'TinyCLIP-19M': [54.06, 2.27, 30.41],
    'TinyCLIP-30M': [37.53, 1.60, 22.96]
}

# Create the third plot
plt.figure(figsize=(6,6))
for model in models:
    plt.plot(x, performance_data_throughput[model], label=model)
plt.xlabel('Environment')
plt.ylabel('Throughput (it/s)')
plt.legend()
plt.grid(True)
plt.savefig('ThroughputModels.png',dpi= 600)

plt.show()


