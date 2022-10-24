import pylab
import numpy as np
fig = pylab.figure()
figlegend = pylab.figure()
ax = fig.add_subplot(111)
vals = np.load("/home/josyula/Documents/Assignments/Deep Learning/A3/something2.npy")
lines  = []
length = range(1,21)
lengths = np.tile(length, (len(vals),1))
lines = ax.plot(np.transpose(lengths), np.transpose(vals))
s = [3, 6, 9]
leg = [ str(i) for i in s]
figlegend.legend(lines, leg, 'center')
fig.show()
figlegend.show()
# figlegend.savefig('legend.png')
