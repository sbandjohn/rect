
import os
import tensorflow as tf
import rect_data
import time

N_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 1000
MAX_STEP = 10000
HEIGHT = 10
WIDTH = 10

class CNN:
    def __init__(self, n_classes, lr, height, width):
    ##构造CNN网络。
        self.n_classes = n_classes      #验证码字符集大小
        self.learning_rate = lr         #学习率
        self.height = height            #图片的高
        self.width = width              #图片的宽
        self.global_step = tf.Variable(0, dtype = tf.int32, trainable = False) #每次从checkpoint读档时可以通过此变量读出当前的训练step值
        # self.DROPOUT = 0.95             #训练时默认的dropout
        self.ckptdir = r'checkpoints\for_rect' #checkpoint目录
        #定义X为输入向量，Y为label
        self.X = tf.placeholder(tf.float32, [None, self.height * self.width])
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.dropout = tf.placeholder(tf.float32)
        w_alpha = 0.01
        b_alpha = 0.1   #w_alpha和b_alpha用于初始化。如果不设置的话可能loss函数值初始值过大
        #第一层卷积层和下采样层
        x = tf.reshape(self.X, shape=[-1, self.width, self.height, 1])
        w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)) #定义激活函数为relu函数
        # conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #下采样层采用最大值采用
        # conv1 = tf.nn.dropout(conv1, self.dropout) #设置dropout，训练时只训练self.dropout比例的节点

        #全连接层
        w_d = tf.Variable(w_alpha*tf.random_normal([self.width*self.height*32, 256]))
        #因为卷积层padding='SAME'，所以只有三个下采用层缩小了图片，一共使得长宽都变成原来的1/8
        b_d = tf.Variable(b_alpha*tf.random_normal([256]))
        dense = tf.reshape(conv1, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        # dense = tf.nn.dropout(dense, self.dropout)
        #输出层
        w_out = tf.Variable(w_alpha*tf.random_normal([256, self.n_classes]))
        b_out = tf.Variable(b_alpha*tf.random_normal([self.n_classes]))
        self.output = tf.add(tf.matmul(dense, w_out), b_out)
        #定义loss函数和优化函数
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
        #这里的accuracy是计算识别所有数字的准确率
        max_idx_p = tf.argmax(self.output, 1)
        max_idx_l = tf.argmax(self.Y, 1)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def train_and_test_model(model, batch_size=BATCH_SIZE, skip_step=SKIP_STEP, max_step=MAX_STEP):
    #训练我们的模型。每次拿出batch_size张图片，每隔skip_step存一次checkpoint，并且生成Test_batch_size张图片进行测试
    input = rect_data.read_data_sets(one_hot=True)
    if (os.path.exists(model.ckptdir) == False):
        os.makedirs(model.ckptdir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model.ckptdir + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:     #读取checkpoint，恢复训练状态
            # print ("found!")
            saver.restore(sess, ckpt.model_checkpoint_path)

        initial_step = model.global_step.eval()
        start_time = time.time()
        total_loss = 0

        for index in range(initial_step, max_step):
            X_batch, Y_batch = input.train.next_batch(batch_size)
            _, loss_batch = sess.run([model.optimizer, model.loss],
                                     feed_dict={model.X: X_batch, model.Y: Y_batch})
            total_loss += loss_batch
            print('loss at step {}: {:5.6f}'.format(index + 1, loss_batch))     #每训练一步都输出loss函数值

            if (index + 1) % skip_step == 0:
                saver.save(sess, model.ckptdir + '/identify-convnet', index)    #保存模型

        print("Optimization Finished!")
        print("Total time: {0} seconds".format(time.time() - start_time))

        Accuracy = sess.run(model.accuracy, feed_dict={model.X: input.test.images, model.Y: input.test.labels})
        print ("Accuracy = {}".format(Accuracy))


if __name__ == '__main__':
    model = CNN(N_CLASSES, LEARNING_RATE, HEIGHT, WIDTH)
    train_and_test_model(model)