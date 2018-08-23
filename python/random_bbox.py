import random

def gen_random_boxes(h=345, w=500, n_boxes=20):
    boxes = []
    for _ in range(n_boxes):
        x_max = random.randint(75, w)
        y_max = random.randint(75, h)
        x_min = random.randint(10, x_max-10)
        y_min = random.randint(10, y_max-10)
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes

if __name__ == '__main__':
    print(gen_random_boxes())

