import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read the data
df = pd.read_csv('out.csv')
df['Local Timestamp'] = pd.to_datetime(df['Local Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
print(df.info)
df = df.dropna(subset=['Local Timestamp'])
timestamp = df['Local Timestamp']
print(timestamp.info)
timestamp = timestamp.unique()
timestamp = np.sort(timestamp)
x_min = df['X'].min()
x_max = df['X'].max()
y_min = df['Y'].min()
y_max = df['Y'].max()

df = df.groupby('Local Timestamp')

print(df['Local Timestamp'])

# Plot the data of 1st frame
fig, ax = plt.subplots()
plt.xlabel("X")
plt.ylabel("Y")
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
plt.legend()
plt.title("Grouped Data by Timestamp")

def update(frame):
    frame_df = df.get_group(timestamp[frame])
    ax.plot(frame_df['X'], frame_df['Y'], marker='o', linestyle='', ms=12)
    plt.xlabel("X")
    plt.ylabel("Y")
    return ax

ani = animation.FuncAnimation(fig, update, frames=len(timestamp), repeat=False)
plt.show()
# Plot the data of 1st frame in 3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for name, group in df:
#     ax.scatter(group['X'], group['Y'], group['Z'], marker='o')
    
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()
