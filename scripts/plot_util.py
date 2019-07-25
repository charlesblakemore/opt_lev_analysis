import matplotlib.pyplot as plt
import numpy as np

def plot_line(xdata, ydata, xlabel='', ylabel=''):
	plt.plot(xdata,ydata)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
def plot_scatter(xdata, ydata, xlabel='', ylabel='', alpha=0.25):
	plt.scatter(xdata,ydata)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)	
