import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def gcd_extended(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = gcd_extended(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def find_min_x(num, rem):
    prod = 1
    for n in num:
        prod *= n

    result = 0
    for i in range(len(num)):
        prod_i = prod // num[i]
        _, inv_i, _ = gcd_extended(prod_i, num[i])
        result += rem[i] * prod_i * inv_i

    return result % prod

# Example Usage
num1 = [5, 7]
rem1 = [1, 3]
print("x is", find_min_x(num1, rem1)) 

num2 = [3, 4, 5]
rem2 = [2, 2, 2]
print("x is", find_min_x(num2, rem2)) 



VM1 = 25
VM2 = 30

x1 = np.arange(-VM1, VM1+1)
x2 = np.arange(-VM2, VM2+1)


true_speed = 0
speed1 = lambda speed : (speed+VM1) %(2*VM1) - VM1
speed2 = lambda speed : (speed+VM2) %(2*VM2) - VM2
print(speed1, speed2)

num1 = [2*VM1, 2*VM2]
rem1 = [speed1(true_speed) + VM1, speed2(true_speed) + VM2 -5]
print(num1)
print(rem1)
res = find_min_x(num1, rem1)/10 - VM1
print("x is", res)

exit()
y1 = np.zeros_like(x1)
y2 = np.zeros_like(x2)
y1[speed1(true_speed)+VM1] = 1
y2[speed2(true_speed)+VM2] = 1
fig, ax = plt.subplots()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def frame(i):
    # y1 = np.roll(y1, 2*VM1)
    # print(y1)
    x1_ = x1 + i*2*VM1
    x2_ = x2 + i*2*VM2
    print(x1_.shape, x2_.shape)
    print(y1.shape, y2.shape)
    
    line1, = ax.plot(x1_, y1, color=colors[i])
    line2, = ax.plot(x2_, y2, color=colors[i])
    dotSpeed, = ax.plot([true_speed], [1], 'ro')
    plt.ylim(0, 1)
    plt.xlim(-12*VM2, 12*VM2)
    plt.xlabel("Speed")
    plt.ylabel("Probability")
    plt.title("Speed Probability Distribution")
    plt.legend(["VM1", "VM2", "True Speed"])
    plt.grid()
    plt.show()
    
    print(f"Rolling {i} times")
    return line1, line2, dotSpeed

ani = animation.FuncAnimation(fig, frame, frames=range(-4,4), interval=1000, repeat=False)
plt.show()