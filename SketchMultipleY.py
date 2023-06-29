import matplotlib.pyplot as plt

# Create some mock data
t = [1,2,3,4,5]
gridTime = [4,3,4,3,3]
gridSpace = [0.08928014 , 0.08599463, 0.085799751, 0.095314767, 0.084644667]

randomTime = [5,4,7,6,3]
randomSpace = [0.09 , 0.10, 0.09]

degreeTime = [15,10,84]
degreeSpace = [0.2 , 0.13, 0.14]

opFTime = [7,8,43]
opFSpace = [0.14 , 0.10, 0.13]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('number of iteration')
ax1.set_ylabel('Time Complexity', color=color)
ax1.plot(t, gridTime, color=color)


ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Space Complexity', color=color)  # we already handled the x-label with ax1
ax2.plot(t, gridSpace, color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.xaxis.set_tick_params(labelbottom=False)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()