"""
Mapper for hadoop

Mapper firstly reads all input by line and creats according float numbers, then use length of this array
to creat a Numpy matrix. Power all the value, then send the mean value and the mean value of powered matrix.
These values will be used to calculate mean and variance of Whole input
"""

import sys
from numpy import mat, mean, power

def read_input(file):
	for line in file:
		yield line.rstrip()

input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
sqInput = power(input, 2)

print "%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput))
"""
This is a good habbit to send error report
"""
print >> sys.stderr, "report: still alive"