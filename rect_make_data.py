import sys
import random

n_test = 2000
n_train = 200000

row = 10
col = 10


def zero():
    return [[0 for j in range(col)] for i in range(row)]


noise_num = 0


def draw_rectangle(a, x1, y1, x2, y2):
    for i in range(x2 - x1 + 1):
        for j in range(y2 - y1 + 1):
            a[x1 + i][y1 + j] = 1


def in_range(a, x0, y0, x3, y3, cut_corner=False):
    mr = x3 - x0 + 1
    mc = y3 - y0 + 1
    r = random.randint(2, mr)
    c = random.randint(2, mc)
    x1 = x0 + random.randint(0, mr - r)
    y1 = y0 + random.randint(0, mc - c)
    x2 = x1 + r - 1
    y2 = y1 + c - 1
    draw_rectangle(a, x1, y1, x2, y2)
    if (cut_corner):
        a[x1][y1] = 0
        a[x1][y2] = 0
        a[x2][y1] = 0
        a[x2][y2] = 0


def make_image(num, case=1):
    image = zero()

    if case == 1:
        num = 0
        for x0, y0, x3, y3 in ((0, 0, 3, 3), (5, 0, 9, 3), (0, 5, 3, 9), (5, 5, 9, 9)):
            if (random.randint(0, 1) == 0):
                num += 1
                in_range(image, x0, y0, x3, y3)
        return image, num
    elif case == 4:
        num = 0
        for x0, y0, x3, y3 in ((0, 0, 3, 3), (5, 0, 9, 3), (0, 5, 3, 9), (5, 5, 9, 9)):
            if (random.randint(0, 1) == 0):
                num += 1
                in_range(image, x0, y0, x3, y3, cut_corner=True)
        return image, num
    elif case == 3:
        num = 0
        for x0, y0, x3, y3 in ((0, 0, 3, 3), (5, 0, 9, 3), (0, 5, 3, 9), (5, 5, 9, 9)):
            if (random.randint(0, 1) == 0):
                num += 1
                in_range(image, x0, y0, x3, y3)
                in_range(image, x0, y0, x3, y3)
        return image, num
    elif case == 2:
        for i in range(num):
            r = random.randint(1, 4)
            c = random.randint(1, 4)
            x1 = random.randint(1, row - r - 1)
            y1 = random.randint(1, col - c - 1)
            x2 = x1 + r - 1
            y2 = y1 + c - 1
            draw_rectangle(image, x1, y1, x2, y2)
        _image = [[0 for j in range(col + 2)] for i in range(row + 2)]
        for i in range(row):
            for j in range(col):
                _image[i + 1][j + 1] = image[i][j]
        num_of_corners = 0
        for i in range(row + 1):
            for j in range(col + 1):
                cnt = _image[i][j] + _image[i + 1][j] + _image[i][j + 1] + _image[i + 1][j + 1]
                if (cnt == 1): num_of_corners += 1
        return image, num_of_corners

    return image, num


def noise(a):
    num = random.randint(0, noise_num)
    r = len(a)
    c = len(a[0])
    for i in range(num):
        x = random.randint(0, r - 1)
        y = random.randint(0, c - 1)
        a[x][y] = 1 - a[x][y]


def make_data(n, case):
    images = []
    labels = []
    for i in range(n):
        num = random.randint(0, 3)
        image, label = make_image(num, case=case)
        images.append(image)
        labels.append(label)
    return images, labels


def print_labels(filename, labels):
    with open(filename, "w") as f:
        n = len(labels)
        s = [str(n)]
        s += [str(label) for label in labels]
        s = [x + '\n' for x in s]
        f.writelines(s)


def print_images(filename, images):
    with open(filename, "w") as f:
        n = len(images)
        s = [str(n)]
        s.append(str(row) + ' ' + str(col))
        for image in images:
            for line in image:
                s.append(' '.join(map(str, line)))
        s = [x + '\n' for x in s]
        f.writelines(s)


def main():
    train_images, train_labels = make_data(n_train, case=2)
    test_images, test_labels = make_data(n_test, case=2)
    print_labels("rect_train_labels.txt", train_labels)
    print_images("rect_train_images.txt", train_images)
    print_labels("rect_test_labels.txt", test_labels)
    print_images("rect_test_images.txt", test_images)


if __name__ == "__main__":
    main()