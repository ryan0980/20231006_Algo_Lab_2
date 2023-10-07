from scipy.spatial import ConvexHull
import numpy as np
import time
import matplotlib.pyplot as plt
import math


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        elapsed_time = (time2 - time1) * 1000.0
        return ret, elapsed_time

    return wrap


def turn(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])


def distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


@timing
def graham_scan(points):
    def sorting_key(p):
        angle = math.atan2(p[1] - pivot[1], p[0] - pivot[0])  # by angle and distance
        dist = distance(p, pivot)
        return (angle, dist)

    pivot = points[0]
    for point in points[1:]:  # update to rightmost and lowest point
        if point[1] < pivot[1] or (point[1] == pivot[1] and point[0] > pivot[0]):
            pivot = point
    sorted_pts = sorted(points, key=sorting_key)
    hull = [pivot, sorted_pts[0]]
    for p in sorted_pts[1:]:
        while len(hull) > 1 and turn(hull[-2], hull[-1], p) <= 0:
            hull.pop()  # pop if turn right
        hull.append(p)

    return hull  # complete


"""
@timing
def compute_convex_hull(points):
    return ConvexHull(points)
"""


def generate_sizes(initial_size=1):
    sizes = [initial_size]
    for _ in range(10):
        sizes.append(sizes[-1] * 2)
    return sizes


sizes = generate_sizes(5000)
rng = np.random.default_rng()
sizes_recorded = []
times_recorded = []


for size in sizes:
    points = rng.random((size, 2))
    hull, elapsed_time = graham_scan(points)
    print(f"For {size} points, computation took {elapsed_time:.3f} ms")
    sizes_recorded.append(size)
    times_recorded.append(elapsed_time)

scaling_factor = np.mean(times_recorded[1:]) / np.mean(
    [size * np.log(size) for size in sizes_recorded[1:]]
)
scaling_factor = np.mean(times_recorded[2:]) / np.mean(
    [size * np.log(size) for size in sizes_recorded[2:]]
)
nlogn_values = [size * np.log(size) * scaling_factor for size in sizes_recorded[2:]]

plt.figure(figsize=(10, 6))
plt.plot(sizes_recorded[2:], times_recorded[2:], marker="o", linestyle="-")
plt.plot(sizes_recorded[2:], nlogn_values, linestyle="--", color="red", label="nlogn")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Points")
plt.ylabel("ms")
plt.legend()

print(sizes_recorded[2:])
print(times_recorded[2:])
print(nlogn_values)

plt.show()
