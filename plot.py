import matplotlib.pyplot as plt

import numpy as np

# 237000 x1 474000 x2 711000 x3 948000 x4 1185000 x5
PV_x = np.array([0, 1, 2, 3])
PV_y = np.array([0, 237000, 711000, 1422000])
AC_x = np.array([0, 5])
AC_y = np.array([0, 236800])
EV_x = np.array([0, 5])
EV_y = np.array([0, 252300])
pointsX = [5, 4.7]
pointsY = [1422000, 1335211]


fig, ax = plt.subplots()
ax.annotate("BAC", (pointsX[0], pointsY[0]))
ax.annotate("EAC", (pointsX[1], pointsY[1]))
plt.title("EVM Visualization")
plt.xlabel("Month")
plt.ylabel("EGP")
ax.plot(PV_x, PV_y, '-b', label='PV')
ax.plot(EV_x, EV_y, '-b', label='EV', linestyle='--')
ax.plot(AC_x, AC_y, '-b', linewidth=2, color='green', label='AC', linestyle=':')
ax.scatter(pointsX, pointsY)

# ax.axis('equal')
leg = ax.legend();
# plt.plot(xpoints, ypoints, linestyle='dotted')
plt.show()