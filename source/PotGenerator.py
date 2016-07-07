import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as p

'''
	Viewer window: (general) accepts a list of points and displays. (Specific) accepts a list of parametric function curves and plots.
	SQLLite module: accepts the list of functions and stores it. RECORD and FUNCTION tables. Possibly also subtables  of the FUNCTION table.
	Generator module: actually generates random functions. 
		Pics a function type, picks parameters. 
		A simple starter is to generate random polynomials on their side then flip 90 
		deg for the display window. Extend the list class for function type!! 
		Viewer displays based on that type!
	Main: generates functions, displays functions, allows user to like or not, stores in db. 
'''
'''
	Notes 5/10/2016: Not everything needs to be an object. Here, we don't really see any advantages over just 
	writing an algorithm that generates the polynomial on the spot and then just outputs the points and the 
	constants used in the generation.
	Simple(ish) nested for loop. This would have taken substantially less time than the monstrosity I built below.
'''

class PolyPotGenerator(list):
	def __init__(self, polyLimits):
		super(PolyPotGenerator,self).__init__()
		self.polyLimits = polyLimits # limits for each poly
		# self.xLimits = sorted(xLimits)
		self.numCurves = random.randint(2,6)
		self.xLimits = sorted([random.randint(1, 100) for i in range(self.numCurves-1)])
		self.xLimits.append(100)
		self.offset = random.randint(10, 40)
		
		self.Build(polyLimits)
	
	def Build(self, polyLimits):
		for i in range(self.numCurves):
			super(PolyPotGenerator,self).append( Poly3Curve(polyLimits) )
	
	def getY(self, x):
		if x > max(self.xLimits): #[len(self.xLimits)-1]:
			return None
		y = 0
		# i = 0
		limsum = 0
		# while self.xLimits[i] < x:
			# y = y + self[i].getY(self.xLimits[i]-limsum)
			# limsum = limsum + self.xLimits[i]
			# i = i + 1
			# # print y
		# y = y + self[i].getY(x)
		# return self.offset + y
		for i in range(self.numCurves):
			# sum each y-val
			if self.xLimits[i] < x:
				y = y + self[i].getY(self.xLimits[i]-limsum)
				limsum = self.xLimits[i]
			# sum the last poly's x-val
			else:
				y = y + self[i].getY(x-limsum)
				break
		return self.offset + y
		
	def verify(self,x,y):
		if min(y) < 10:
			m = -min(y)+10
			return (x,[yi+m for yi in y])
		else:
			return (x,y)
		
	def getPoints(self):
		x = range(max(self.xLimits))
		y = [self.getY(i) for i in x]
		# print x,y
		return self.verify(x,y)
			
	def plot(self,save):
		X,Y = self.getPoints()
		# print self.numCurves
		p.cla()
		p.plot(Y,X,'bo-',[-y for y in Y],X,'bo-')
		p.grid()
		p.axis([-100,100,0,100])
		# p.axis('equal')
		if save:
			p.savefig('fig.jpg')
		else:
			p.show(True)
	
class Poly3Curve(list):
	''' Represents a polynomial. Does not have an understanding of x- or y-offset. '''
	def __init__(self,polyLimits):
		super(Poly3Curve,self).__init__()
		self.polyLimits = polyLimits
		self.getPoly3()
	
	def getPoly3(self):
		''' returns 3 random polynomial constants '''
		for i in range(0,6,2):
			super(Poly3Curve,self).append(
				random.uniform(self.polyLimits[i], self.polyLimits[i+1]))
				
	def getY(self, x):
		''' returns the y output for a given x output '''
		# print 'getY', x, sum([x*i for i in self])
		# range(1,len(self))
		# return self[0]*x
		return self[0]*pow(x,1) + self[1]*pow(x,2) # + self[2]*pow(x,2)
		# return sum([i*pow(x,n) for (i,n) in zip(self,[1, 2, 1])])
		

if __name__ == "__main__":
	#Testing only.
	polyLimits = (-.1,.1,-.03,.03,-.0001,.0001)
	g = PolyPotGenerator(polyLimits)
	print "test"
	print str(g.numCurves)+': '+str([round(c,2) for poly in g for c in poly])
	g.plot(False)
	while raw_input() != 'q':
		g = PolyPotGenerator(polyLimits)
		print str(g.numCurves)+': '+str([round(c,2) for poly in g for c in poly])
		g.plot(False)
		
	
	
	
		
