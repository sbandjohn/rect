import sys
import random

n_test = 2000
n_train = 20000

row = 10
col = 10

def zero():
	return [ [0 for j in range(col)] for i in range(row) ]

noise_num = 0

def draw_rectangle(a, x1, y1, x2, y2):
	for i in range(x2-x1+1):
		for j in range(y2-y1+1):
			a[x1+i][y1+j] = 1

def make_image(num):
	image = zero()
	for i in range(num):
		r = random.randint(1, 1)
		c = random.randint(1, 1)
		x1 = random.randint(0, row-r)
		y1 = random.randint(0, col-c)
		x2 = x1 + r - 1
		y2 = y1 + c - 1
		draw_rectangle(image, x1, y1, x2, y2)
	return image

def noise(a):
	num = random.randint(0, noise_num)
	r = len(a)
	c = len(a[0])
	for i in range(num):
		x = random.randint(0, r-1)
		y = random.randint(0, c-1)
		a[x][y] = 1 - a[x][y]

def make_data(n):
	images = []
	labels = []
	for i in range(n):
		num = random.randint(0,1)
		image = make_image(num)
		noise(image)
		images.append(image)
		labels.append(num)
	return images, labels

def print_labels(filename, labels):
	with open(filename, "w") as f:
		n = len(labels)
		s = [str(n)]
		s += [str(label) for label in labels]
		s = [x+'\n' for x in s]
		f.writelines(s)

def print_images(filename, images):
	with open(filename, "w") as f:
		n = len(images)
		s = [str(n)]
		s.append(str(row) + ' ' + str(col))
		for image in images:
			for line in image:
				s.append(' '.join(map(str, line)))
		s = [x+'\n' for x in s]
		f.writelines(s)

def main():
	train_images, train_labels = make_data(n_train)
	test_images, test_labels = make_data(n_test)
	print_labels("rect_train_labels.txt", train_labels)
	print_images("rect_train_images.txt", train_images)
	print_labels("rect_test_labels.txt", test_labels)
	print_images("rect_test_images.txt", test_images)
	
if __name__ == "__main__":
	main()