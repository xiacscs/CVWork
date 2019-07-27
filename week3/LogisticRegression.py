## Logistic Regression
###############################
import numpy as np
import random
import matplotlib.pyplot as plt

# 预测值
def inference(batch_x_Matrix1, wbMatri):
    z = np.dot(batch_x_Matrix1, wbMatri).reshape(-1, 1)
    return 1.0 / (1 + np.exp(-z))

# 损失估算
def eval_loss(batch_pred_y_Matrix, batch_gt_y_Matrix):
    batch_size = batch_gt_y_Matrix.size
    epsilon = 1e-5
    avg_loss = -np.sum(batch_gt_y_Matrix * np.log(batch_pred_y_Matrix + epsilon)
                       + (1 - batch_gt_y_Matrix) * np.log(1 - batch_pred_y_Matrix + epsilon)) / batch_size
    return avg_loss

# 计算每一步的w，b，loss
def cal_step_gradientAndloss(batch_x_list, batch_gt_y_list, w1, w2, b, lr):
    batch_size = len(batch_x_list)
    batch_column = len(batch_x_list[0])
    # 包装成矩阵
    batch_x_Matrix = np.array(batch_x_list).reshape(-1, batch_column)
    batch_x_Matrix1 = np.hstack([batch_x_Matrix, np.ones((batch_size, 1))])
    batch_gt_y_Matrix = np.array(batch_gt_y_list).reshape(-1, 1)
    wbMatrix = np.array([w1, w2, b]).reshape(-1, 1)

    # 计算预测值
    batch_pred_y_Matrix = inference(batch_x_Matrix1, wbMatrix)

    # 计算每组样本的w,b
    diff = (batch_pred_y_Matrix - batch_gt_y_Matrix).reshape(-1, 1)
    d = np.dot(batch_x_Matrix1.T, diff) / batch_size
    cwb = wbMatrix - lr * d
    cw1, cw2, cb = cwb[:, 0]

    # 计算误差
    avg_loss = eval_loss(batch_pred_y_Matrix, batch_gt_y_Matrix)

    return cw1, cw2, cb, avg_loss

# 训练
def train(x1_list, x2_list, gt_y_list, batch_size, lr, max_iter):
    w1 = 0
    w2 = 0
    b = 0
    num_samples = len(x1_list)
    loss = []
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = [[x1_list[j], x2_list[j]] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w1, w2, b, curloss = cal_step_gradientAndloss(batch_x, batch_y, w1, w2, b, lr)
        print('w1:{0}, w2:{1}, b:{2}'.format(w1, w2, b))
        print('loss is {0}'.format(curloss))
        loss.append(curloss)
    return [w1, w2, b], loss

# 生成随机样本数据
def gen_sample_data():
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    x1_list = [random.randint(0, 100) * random.random() for _ in range(num_samples)]
    x2_list = [w * 100 * random.random() + b for _ in range(num_samples)]
    y_list = [1 if (w * x1 + b + random.random() * 20) > x2 else 0 for x1, x2 in zip(x1_list, x2_list)]
    return x1_list, x2_list, y_list, w, b

def draw(x1_list, x2_list, y_list, model, loss):
    fig, axes = plt.subplots(1, 2)
    plt1, plt2 = axes

    plt1.set_xlabel('x1')
    plt1.set_ylabel('x2')

    x1_matrix = np.array(x1_list)
    x2_matrix = np.array(x2_list)
    y_matrix = np.array(y_list)
    c1 = y_matrix == 1
    c2 = y_matrix == 0
    plt1.scatter(x1_matrix[c1], x2_matrix[c1], color='g')
    plt1.scatter(x1_matrix[c2], x2_matrix[c2], color='b')

    x1 = np.linspace(0, 100, 100)
    w1, w2, b = model
    x2 = [-(w1 * _ + b) / w2 for _ in x1]
    plt1.plot(x1, x2, color='red')

    plt2.plot(list(range(len(loss))), loss)
    plt2.set_xlabel('iter')
    plt2.set_ylabel('loss')

    plt.show()

def run():
    x1_list, x2_list, y_list, w, b = gen_sample_data()
    lr = 0.002
    max_iter = 5000
    model, loss = train(x1_list, x2_list, y_list, 50, lr, max_iter)
    draw(x1_list, x2_list, y_list, model, loss)

if __name__ == '__main__':
    run()
