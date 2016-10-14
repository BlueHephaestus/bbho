import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

def graph_output(plot_2d_results, plot_3d_results, bbf_evaluation_i, bbf_evaluation_n, domain_x, domain_y, detail_n, test_means, bbf_inputs, bbf_evaluations, val1, val2):

  #Set the filename
  fname = "results/%02d" % bbf_evaluation_i

  #Plot our updates
  if plot_2d_results:
      plt.plot(domain_x, test_means)
      #plt.plot(domain_x, test_variances, 'r')
      #plt.plot(bbf_inputs, bbf_evaluations, 'bo')
      plt.scatter(bbf_inputs, bbf_evaluations, marker='o', c='b', s=100.0, label="Function Evaluations")
      plt.plot(domain_x, val1, 'r')
      plt.plot(domain_x, val2, 'r')
      #plt.plot(domain_x, bbf(domain_x), 'y')
      plt.savefig("%s.jpg" % fname, dpi=None, facecolor='w', edgecolor='w',
          orientation='portrait', papertype=None, format=None,
          transparent=False, bbox_inches='tight', pad_inches=0.1,
          frameon=None)
      plt.xlabel("X-Axis")
      plt.ylabel("Y-Axis")

      plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
      plt.axis([0, 10, 0, 2])
      #plt.show()
      plt.gcf().clear()

  elif plot_3d_results:
      #So we only render on the last one(just erase this if you want all of them)
      if bbf_evaluation_i == bbf_evaluation_n-1:
          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')
          #X & Y have to be matrices of all vertices
          #Z has to be matrix of outputs
          #Convert our vectors to compatible matrix counterparts
          Y = np.array([[i] for i in domain_y])

          X = np.tile(domain_x, (detail_n, 1))
          Y = np.tile(Y, (1, detail_n))

          #This ones easy, just reshape
          Z1 = test_means.reshape(detail_n, detail_n)
          #Z2 = test_variances.reshape(detail_n, detail_n)
          Z3 = (val1).reshape(detail_n, detail_n)
          Z4 = (val2).reshape(detail_n, detail_n)


          ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.coolwarm)
          #ax.plot_wireframe(X, Y, Z2, rstride=1, cstride=1)
          ax.plot_wireframe(X, Y, Z3, rstride=1, cstride=1)
          ax.plot_wireframe(X, Y, Z4, rstride=1, cstride=1)
          plt.savefig("%s.jpg" % fname, dpi=None, facecolor='w', edgecolor='w',
              orientation='portrait', papertype=None, format=None,
              transparent=False, bbox_inches='tight', pad_inches=0.1,
              frameon=None)

          plt.gcf().clear()
          #plt.show()
