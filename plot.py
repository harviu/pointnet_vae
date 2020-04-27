from matplotlib import pyplot as plt 
import numpy as np 

# l = [6.935143327948405, 8.965171335258427, 11.597996565346108, 12.50755607132556, 15.392655079909616, 14.416854041017714, 57.40821761236677, 54.92846249955617, 244.29971434797758]
l = [8.196247170114317, 8.29600201718756, 14.212776436742644, 12.173923156579729, 95.20948612643826, 206.03231317423544, 334.57933833989546, 250.27858259517765, 475.67669753951003]
# a = [4.1928573, 4.766005, 6.1226754, 8.710853, 9.684097, 15.542747, 354.77896, 362.6682, 863.3907]
a = [4.668906, 3.7032626, 85.96202, 103.05894, 222.57246, 338.42798, 415.3846, 485.58203, 507.8628]

plt.grid(True)
plt.plot(l,c="c",marker="o",label='latent')
plt.plot(a,c="c",marker="^",label='no latent')
# plt.plot(l[1],c="m",marker="o",label='Halo 2, latent')
# plt.plot(a[1],c="m",marker="^",label='Halo 2, no latent')
# plt.plot(l[2],c="green",marker="o",label='Halo 3, latent')
# plt.plot(a[2],c="green",marker="^",label='Halo 3, no latent')
plt.xticks(ticks=range(9),labels=range(20,30))
plt.xlabel("Time")
plt.ylabel("Chamfer Distance")
plt.legend()
plt.show()