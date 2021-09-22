import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

# params
s = 10
p = 28
b = 8/3
v_0 = [1.0001, -1, 10]
t = np.linspace(0, 80, 30000)  # (start t, finish t, steps)

# right side ODE
def f(v, t):
    v1, v2, v3 = v
    r = [s * (v2 - v1), v1 * (p - v3) - v2, v1 * v2 - b * v3]
    return r


# solution
[x, y, z] = odeint(f, v_0, t).T

# visualising Lorenz system
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz System")
plt.show()

# making dataset
input_data = []
output_data = []
for n in range(len(x)-20):
    input_scrap = []
    output_scrap = []
    for i in range(20):
        input_scrap.append(x[n+i])
    output_scrap.append(x[n + 20])
    input_data.append(input_scrap)
    output_data.append(output_scrap)
input_data = np.array(input_data)
output_data = np.array(output_data)

#  splitting dataset to training data and testing data (50/50)
spl_id = np.vsplit(input_data, 2)
spl_od = np.vsplit(output_data, 2)

training_in = spl_id[0]
testing_in = spl_id[1]
training_out = spl_od[0]
testing_out = spl_od[1]






