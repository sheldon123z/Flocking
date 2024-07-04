# %%
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.proj3d import proj_transform
# from mpl_toolkits.mplot3d.axes3d import Axes3D

# from matplotlib.text import Annotation



# class Annotation3D(Annotation):

#     def __init__(self, text, xyz, *args, **kwargs):
#         super().__init__(text, xy=(0, 0), *args, **kwargs)
#         self._xyz = xyz

#     def draw(self, renderer):
#         x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
#         self.xy = (x2, y2)
#         super().draw(renderer)
        
# def _annotate3D(ax, text, xyz, *args, **kwargs):
#     '''Add anotation `text` to an `Axes3d` instance.'''

#     annotation = Annotation3D(text, xyz, *args, **kwargs)
#     ax.add_artist(annotation)

# setattr(Axes3D, 'annotate3D', _annotate3D)




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # adding a few 3d points to annotate
# ax.scatter([0, 0, 0], [0, 0, 1], [0, 1, 0], s=30, marker='o', color='green')
# ax.annotate3D('point 1', (0, 0, 0), xytext=(3, 3), textcoords='offset points')
# ax.annotate3D('point 2', (0, 1, 0),
#               xytext=(-30, -30),
#               textcoords='offset points',
#               arrowprops=dict(ec='black', fc='white', shrink=2.5))
# ax.annotate3D('123',xyz=(0, 0, 1),
#               xytext=(-30, -30),
#               textcoords='offset points',
#               arrowprops=dict(arrowstyle="->", ec='black', fc='white', lw=1))
# ax.set_title('3D Annotation Demo')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# fig.tight_layout()

# plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Make the grid
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

# Make the direction data for the arrows
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()



#%%
rom mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')

# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="r")

# draw a point
ax.scatter([0], [0], [0], color="g", s=100)

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

a = Arrow3D([0, 1], [0, 1], [0, 1], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
ax.add_artist(a)
plt.show()


#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 # Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 5 * np.outer(np.cos(u), np.sin(v)) +10
y = 5 * np.outer(np.sin(u), np.sin(v)) +10
z = 5 * np.outer(np.ones(np.size(u)), np.cos(v))
 # Plot the surface
ax.plot_surface(x, y, z, color='b')
plt.show()


# %%


for i in range (1,5):
    print(i)
# %%
