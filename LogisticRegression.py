import sys
import numpy as np
import math
from LSE_newton_method import get_LU,get_inv_column
import matplotlib.pyplot as plt

def transpose(m):
	trans = []
	for i in range(len(m[0])):
		temp = []
		for j in range(len(m)):
			temp.append(m[j][i])
		trans.append(temp)
	return trans

def matrix_mul(m1,m2):
	if len(m1[0]) != len(m2):
		print("matrix_mul dim error!")
		sys.exit(1)

	result = []
	for i in range(len(m1)):
		temp = []
		for j in range(len(m2[0])):
	 		sum = 0
		 	for k in range(len(m2)):
		 		sum = sum + m1[i][k] * m2[k][j]
	 		
		 	temp.append(sum)
		result.append(temp)

	return result

def Univariate_gaussian_data_generator(mean,variance):
	S = 0.0
	while S >= 1.0 or S == 0.0:
		U = np.random.random_sample() * 2 - 1
		V = np.random.random_sample() * 2 - 1
		S = U**2 + V**2

	multiply = math.sqrt( (-2.0) * math.log(S) / S )
	U = U * multiply
	V = V * multiply

	return mean + math.sqrt(variance) * U

def Create_data(n,mx,vx,my,vy):
	list = []
	for i in range(n):
		list.append([Univariate_gaussian_data_generator(mx,vx),Univariate_gaussian_data_generator(my,vy),1.0])
	
	return list

def Gradient_descent(data_class1,data_class2):
	X = data_class1 + data_class2 
	y_class1 = [0.0 for i in range(len(data_class1))]
	y_class2 = [1.0 for i in range(len(data_class2))]
	y = y_class1 + y_class2

	W = [[5] for i in range(3)]

	min_iter = 5000

	while 1:
		XiW = matrix_mul(X,W)
		Xt = transpose(X)
		gradient_mul = []
		for i in range(2*len(data_class1)):
			gradient_mul.append([y[i] - 1.0 / (np.exp(-XiW[i][0]) + 1.0)])

		#print(gradient_mul)
		gradient = matrix_mul(Xt,gradient_mul)

		

		for i in range(3):
			W[i] = W[i] + 0.3*gradient[i][0]
		
		eplison = 1e-2
		
		if abs(gradient[0][0]) < eplison and abs(gradient[1][0]) <  eplison and abs(gradient[2][0]) <  eplison:
			break
		if min_iter == 0:
			break
		min_iter = min_iter - 1
	return W

def get_inverse(x):
	L,U = get_LU(x)
	invt = []
	for i in range(len(L)):
		invt.append(get_inv_column(L,U,i))

	inv = transpose(invt)

	return inv

def Newton(data_class1,data_class2):
	X = data_class1 + data_class2 
	W = [[0] for i in range(3)]
	y_class1 = [0.0 for i in range(len(data_class1))]
	y_class2 = [1.0 for i in range(len(data_class2))]
	y = y_class1 + y_class2
	Xt = transpose(X)
	while 1:
		D = []
		XiW = matrix_mul(X,W)
		for i in range(len(data_class1)*2):
			temp = []
			for j in range(len(data_class1)*2):
				if i == j:
					exp = np.exp(-XiW[i][0])
					temp.append(exp / np.power(1+exp,2))
				else:
					temp.append(0.0)
			D.append(temp)
		
		
		Hessian = matrix_mul(Xt,D)
		Hessian= matrix_mul(Hessian,X)
		Hessian_inv = get_inverse(Hessian)

		gradient_mul = []
		for i in range(2*len(data_class1)):
			gradient_mul.append([y[i] - 1.0 / (np.exp(-XiW[i][0]) + 1.0)])

		#print(gradient_mul)
		gradient = matrix_mul(Xt,gradient_mul)
		

		update = matrix_mul(Hessian_inv,gradient)


		for i in range(3):
			W[i] = W[i] + update[i][0]

		eplison = 1e-15
		if gradient[0][0] < eplison and gradient[1][0] <  eplison and gradient[2][0] <  eplison:
			break

	return(W)


# w1x1 + x2y1 + w3, yi = 0 or 1
def Logistic_Regression(data_class1,data_class2):
	W_gra = Gradient_descent(data_class1,data_class2)
	W_newton = Newton(data_class1,data_class2)

	ground_truth_x1 = []
	ground_truth_y1 = []
	ground_truth_x2 = []
	ground_truth_y2 = []

	predict_class1_gra_x = []
	predict_class1_gra_y = []
	predict_class2_gra_x = []
	predict_class2_gra_y = []

	predict_class1_new_x = []
	predict_class1_new_y = []
	predict_class2_new_x = []
	predict_class2_new_y = []

	true_x1_gra = 0
	for i in range(len(data_class1)):
		XW = matrix_mul([data_class1[i]],W_gra)
		predict = 1/ (1+np.exp(-XW[0][0]))
		if predict < 0.5:
			true_x1_gra = true_x1_gra + 1
			predict_class1_gra_x.append(data_class1[i][0])
			predict_class1_gra_y.append(data_class1[i][1])
		else:
			predict_class2_gra_x.append(data_class1[i][0])
			predict_class2_gra_y.append(data_class1[i][1])
		ground_truth_x1.append(data_class1[i][0])
		ground_truth_y1.append(data_class1[i][1])
	
	true_x2_gra = 0
	for i in range(len(data_class2)):
		XW = matrix_mul([data_class2[i]],W_gra)
		predict = 1/ (1+np.exp(-XW[0][0]))
		if predict > 0.5:
			true_x2_gra = true_x2_gra + 1
			predict_class2_gra_x.append(data_class2[i][0])
			predict_class2_gra_y.append(data_class2[i][1])
		else:
			predict_class1_gra_x.append(data_class2[i][0])
			predict_class1_gra_y.append(data_class2[i][1])

		ground_truth_x2.append(data_class2[i][0])
		ground_truth_y2.append(data_class2[i][1])


	true_x1_new = 0
	for i in range(len(data_class1)):
		XW = matrix_mul([data_class1[i]],W_newton)
		predict = 1/ (1+np.exp(-XW[0][0]))
		if predict < 0.5:
			true_x1_new = true_x1_new + 1
			predict_class1_new_x.append(data_class1[i][0])
			predict_class1_new_y.append(data_class1[i][1])
		else:
			predict_class2_new_x.append(data_class1[i][0])
			predict_class2_new_y.append(data_class1[i][1])
		
	
	true_x2_new = 0
	for i in range(len(data_class2)):
		XW = matrix_mul([data_class2[i]],W_newton)
		predict = 1/ (1+np.exp(-XW[0][0]))
		if predict > 0.5:
			true_x2_new = true_x2_new + 1
			predict_class2_new_x.append(data_class2[i][0])
			predict_class2_new_y.append(data_class2[i][1])
		else:
			predict_class1_new_x.append(data_class2[i][0])
			predict_class1_new_y.append(data_class2[i][1])

	print("Gradient_descent:")

	print("w:")
	print("%f\n%f\n%f" %(W_gra[2][0],W_gra[1][0],W_gra[0][0]))

	print("Confusion matrix:")

	print('%20s cluster 1 %10s cluster 2' %('Predictor','Predictor'))
	print("Is cluster 1%10d%20d" %(true_x1_gra,len(data_class1)-true_x1_gra))
	print("Is cluster 2%10d%20d" %(len(data_class2)-true_x2_gra,true_x2_gra))

	print("Sensitivity (Successfully predict cluster 1): %2f" %(true_x1_gra/50))
	print("Specificity (Successfully predict cluster 2): %2f" %(true_x2_gra/50))

	print("Newton method:")

	print("w:")
	print("%f\n%f\n%f" %(W_newton[2][0],W_newton[1][0],W_newton[0][0]))

	print("Confusion matrix:")

	print('%20s cluster 1 %10s cluster 2' %('Predictor','Predictor'))
	print("Is cluster 1%10d%20d" %(true_x1_new,len(data_class1)-true_x1_new))
	print("Is cluster 2%10d%20d" %(len(data_class2)-true_x2_new,true_x2_new))

	print("Sensitivity (Successfully predict cluster 1): %2f" %(true_x1_new/50))
	print("Specificity (Successfully predict cluster 2): %2f" %(true_x2_new/50))

	plt.figure()

	plt.subplot(1,3,1)
	plt.title("Ground truth")
	plt.scatter(ground_truth_x1,ground_truth_y1,color = 'r')
	plt.scatter(ground_truth_x2,ground_truth_y2,color = 'b')

	plt.subplot(1,3,2)
	plt.title("Gradient descent")
	plt.scatter(predict_class1_gra_x,predict_class1_gra_y,color = 'r')
	plt.scatter(predict_class2_gra_x,predict_class2_gra_y,color = 'b')

	plt.subplot(1,3,3)
	plt.title("Newton's method")
	plt.scatter(predict_class1_new_x,predict_class1_new_y,color = 'r')
	plt.scatter(predict_class2_new_x,predict_class2_new_y,color = 'b')
	plt.show()

def main():
	argument = sys.argv[1:]
	if len(argument) != 9:
		print("You should input all the arguments")
		sys.exit(1)

	number_of_data_points = int(argument[0])
	mx1 = int(argument[1])
	vx1 = int(argument[2])
	my1 = int(argument[3])
	vy1 = int(argument[4])
	mx2 = int(argument[5])
	vx2 = int(argument[6])
	my2 = int(argument[7])
	vy2 = int(argument[8])
	
	data_class1 = Create_data(number_of_data_points,mx1,vx1,my1,vy1)
	data_class2 = Create_data(number_of_data_points,mx2,vx2,my2,vy2)

	Logistic_Regression(data_class1,data_class2)

if __name__ == '__main__':
	main()
