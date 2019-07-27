## Linear Regression
###############################
import numpy as np
import random
import matplotlib.pyplot as plt

# 预测值
def inference(batch_x_Matrix1, wbMatri):
    return np.dot(batch_x_Matrix1, wbMatri).reshape(-1, 1)

# 损失估算
def eval_loss(batch_pred_y_Matrix, batch_gt_y_Matrix):
    batch_size = batch_gt_y_Matrix.size
    diff = (batch_pred_y_Matrix - batch_gt_y_Matrix).reshape(-1, 1)
    avg_loss = np.sum(diff * diff) / batch_size * 0.5
    return avg_loss

# 计算每一步的w，b，loss
def cal_step_gradientAndloss(batch_x_list, batch_gt_y_list, w, b, lr):
    batch_size = len(batch_x_list)
    # 包装成矩阵
    batch_x_Matrix = np.array(batch_x_list).reshape(-1, 1)
    batch_x_Matrix1 = np.hstack([batch_x_Matrix, np.ones((batch_size, 1))])
    batch_gt_y_Matrix = np.array(batch_gt_y_list).reshape(-1, 1)
    wbMatrix = np.array([w, b]).reshape(-1, 1)

    # 计算预测值
    batch_pred_y_Matrix = inference(batch_x_Matrix1, wbMatrix)

    # 计算每组样本的w,b
    diff = (batch_pred_y_Matrix - batch_gt_y_Matrix).reshape(-1, 1)
    d = np.dot(batch_x_Matrix1.T, diff) / batch_size
    cwb = wbMatrix - lr * d
    cw, cb = cwb[:, 0]

    # 计算误差
    avg_loss = eval_loss(batch_pred_y_Matrix, batch_gt_y_Matrix)

    return cw, cb, avg_loss

# 训练
def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = len(x_list)
    loss = []
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b, curloss = cal_step_gradientAndloss(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(curloss))
        loss.append(curloss)

    return [w, b], loss
# 生成随机样本数据
def gen_sample_data():
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 300
    x_list = [random.randint(0, 100) * random.random() for _ in range(num_samples)]
    y_list = [w * x + b + random.random() * random.randint(-1, 1) for x in x_list]
    return x_list, y_list, w, b

def draw(x_list, y_list, model, loss):
    fig, axes = plt.subplots(1, 2)
    plt1, plt2 = axes

    plt1.set_xlabel('x')
    plt1.set_ylabel('y')
    plt1.scatter(x_list, y_list, color='blue')

    x = np.linspace(0, 100, 100)
    w, b = model
    y = [(w * _ + b)for _ in x]
    plt1.plot(x, y, color='red')

    plt2.plot(list(range(len(loss))), loss)
    plt2.set_xlabel('iter')
    plt2.set_ylabel('loss')

    plt.show()

def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.00005
    max_iter = 500
    model, loss = train(x_list, y_list, 50, lr, max_iter)
    draw(x_list, y_list, model, loss)

if __name__ == '__main__':
    run()
