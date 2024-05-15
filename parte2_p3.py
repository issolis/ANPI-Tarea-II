from parte2_p2 import newton

print(newton([1.8,1.8], ['exp(x**2) - exp ((2**(1/2)) * x)', 'x-y'], ['x', 'y'],  0.00001, 1000))
print(newton([1.5,2], ['x + exp(y) - cos(y)', '3*x - y -sin(y)'], ['x', 'y'],  0.00001, 1000))
print(newton([3,2], ['x**2 - 2*x - y + 0.5 ', 'x**2 + 4*y**2 -4'], ['x', 'y'],  0.00001, 1000))
print(newton([0.7,1.2], ['x**2 + y**2 -1', 'x**2 - y**2 +0.5'], ['x', 'y'],  0.00001, 1000))
print(newton([1.2,-1.5], ['sin(x) + y*cos(x)', 'x - y'], ['x', 'y'],  0.00001, 1000))
print(newton([-1, -1, -1, -1], ['y*z + w*(y + z) - x', 'x*z  + w * (x + z) - y', 'x*y + w*(x + y) - z', 'x*y + x*z + y*z - 1 -w'], ['x', 'y', 'z', 'w'],  0.00001, 1000))

