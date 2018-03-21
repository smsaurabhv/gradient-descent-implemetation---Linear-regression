from numpy import *


def error(b,m,points):
	computerror = 0
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		computerror+=(y-(m*x+b))**2
	return computerror/float(len(points))	

def stepgradient(thisb,thism,points,lrate):
	#start here
	b_grad = 0
	m_grad = 0
	n = float(len(points))
	for i in range(0,len(points)):
		x =points[i,0]
		y =points[i,1]
		b_grad+=-(2/n)*(y-((thism*x)+thisb))
		m_grad+=-(2/n)*x*(y-((thism*x)+thisb))
	newb = thisb - (lrate*b_grad)
	newm = thism - (lrate*m_grad)
	return [newb,newm]	

def gradient_descent(points,initialmvalue,initialbvalue,lrate,num_iteration):
	b = initialbvalue
	m = initialmvalue

	for i in range(num_iteration):
		b,m = stepgradient(b,m,array(points),lrate)
	return [b,m]	



def run():
	points =genfromtxt('C:/Users/smsaurabhv/Desktop/deeplearning/data.csv',delimiter=',') 
	lrate = 0.0001
	print ("datsets")
	print(points)
	# y = mx+b linear requations
	initialbvalue =0
	initialmvalue =0
	num_iteration = 1000
	[b,m]  = gradient_descent(points,initialmvalue,initialbvalue,lrate,num_iteration)
	print(b)
	print(m)

run()	
