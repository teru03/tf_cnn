# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import argparse
import csv
import os

# Parameters
learning_rate = 0.03 # 学習率 高いとcostの収束が早まる
batch_size = 5000     # 学習1回ごと( sess.run()ごと )に訓練データをいくつ利用するか
display_step = 1     # 1なら毎エポックごとにcostを表示
train_size = 400000     # 全データの中でいくつ訓練データに回すか
step_size = 1000     # 何ステップ学習するか
modelsfile = 'models/model.ckpt' #モデル保存ファイル

n_hidden_1 = 256     # 隠れ層1のユニットの数
n_hidden_2 = 128     # 隠れ層2のユニットの数
n_hidden_3 = 64      # 隠れ層3のユニットの数
n_classes = 2        # 分類するクラスの数 今回は生き残ったか否かなので2

usecolumns = []

ngcol = ['id','target','ps_car_03_cat', 'ps_car_05_cat']

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def eval_gini(y_true, y_prob):
    g = gini(y_prob)/gini(y_true)
    return g


# Create model
def multilayer_perceptron(x, n_input):
    # Initializing the variables
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        #'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Store layers weight & bias
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #layer_1 = tf.nn.sigmoid(layer_1)
    #layer_1 = tf.nn.softmax(layer_1)

    ##layer_1 = tf.nn.softmax(layer_1)
    ## Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.sigmoid(layer_2)
    #layer_2 = tf.nn.softmax(layer_2)
#
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    ##layer_3 = tf.nn.sigmoid(layer_3)
    #layer_3 = tf.nn.softmax(layer_3)
    ## Output layer with linear activation
    do = tf.nn.dropout(layer_3,0.8)
    out_layer = tf.add(tf.matmul(do, weights['out']), biases['out'])
    return out_layer

# training
def train(args):

    #csvからdataframeへ
    df = pd.read_csv( "./input/train.csv", index_col="id" )

    for colname in df.columns:
        desc = df[colname].describe()
        if desc['min'] == -1 :
            wdesc = df[colname].where(df[colname]!=-1).describe()
            if colname.find("cat"):
                df[colname] = df[colname].replace(-1,wdesc['75%'])
            elif colname.find("ind"):
                df[colname] = df[colname].replace(-1,wdesc['mean'])
            elif colname.find("reg"):
                df[colname] = df[colname].replace(-1,wdesc['mean'])
            elif colname.find("bin"):
                df[colname] = df[colname].replace(-1,0)

    ngcol.extend([ cname for cname in df.columns if "_cat" in cname] )
    usecolumns = [ cname for cname in df.columns if cname not in ngcol ]

    # 説明変数
    x_np = np.array(df[usecolumns])

    # 目的変数を2次元化
    #df['target'].where(df['target'] == 0, 'no')
    #df['target'].where(df['target'] == 1, 'yes')
    #dd =  pd.get_dummies(df['target'])
    #d = dd.to_dict('record')
    #vectorizer = DictVectorizer(sparse=False)
    #y_np = vectorizer.fit_transform(d)

    y_np = []
    for i in df.index:
        yclass = [0.0,0.0]
        yclass[df['target'][i]] = 1.0
        y_np.append(yclass)
    
    [x_train, x_test] = np.vsplit(x_np, [train_size]) # 入力データを訓練データとテストデータに分ける
    [y_train, y_test] = np.vsplit(y_np, [train_size]) # ラベルを訓練データをテストデータに分ける
    
    print x_train[0:2]
    print y_train[0:2]
    
    # Network Parameters
    n_input = len(usecolumns)
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    

    # Construct model
    pred = multilayer_perceptron(x, n_input)
    y_ = tf.nn.softmax(pred)

    
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # Define loss and optimizer
    v13 = False
    if tf.__version__ == "1.3.0":
        print "version = ",tf.__version__
        v13 = True

    if v13:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    #cost = tf.reduce_mean(tf.square(pred - y))
    #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step)
    #optimizer = tf.train.MomentumOptimizer(0.01,0.97).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(1e-1).minimize(cost)


    if v13:
        auc, op = tf.metrics.auc(y_,y)
        tf.summary.scalar('cross entoroy', cost)
    else:
        auc, op = tf.contrib.metrics.streaming_auc(y_,y)
        tf.scalar_summary('cross entoroy', cost)
    
    # エポック数
    training_epochs = args.epoch
    print 'epoch count = %d'%training_epochs 
    
    # Launch the graph
    with tf.Session() as sess:

        if v13:
            sess.run( tf.global_variables_initializer() )
            summary_writer = tf.summary.FileWriter('tflogs', graph=sess.graph)
            summary_op = tf.summary.merge_all()
        else:
            sess.run( tf.initialize_local_variables() )
            summary_writer = tf.train.SummaryWriter('tflogs', graph=sess.graph)
            summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()
        
        ckpt = tf.train.get_checkpoint_state('./models')
        if ckpt: # checkpointがある場合
            last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
            print "load " + last_model
            saver.restore(sess, last_model) # 変数データの読み込み
        else:
            sess.run(tf.initialize_all_variables())


        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
    
            # Loop over step_size
            for i in range(step_size):
                # 訓練データから batch_size で指定した数をランダムに取得
                ind = np.random.choice(batch_size, batch_size)
                x_train_batch = x_train[ind]
                y_train_batch = y_train[ind]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: x_train_batch,
                                                              y: y_train_batch})
                # Compute average loss
                avg_cost += c

            # Display logs per epoch step
#           print x_train_batch
#           print y_train_batch
            predicted = sess.run(y_, feed_dict={x:x_np})
            p = [ v[1] for v in predicted ]
            g = eval_gini(np.array(df['target']+0.0),np.array(p))
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost/step_size), \
                    "{:.9f}".format(g)
                    #"auc=%f:" % op.eval(feed_dict={x: x_train_batch, y: y_train_batch}) 

            current_step = tf.train.global_step(sess, global_step)
            saver.save(sess, './models/model', global_step=current_step)
            summary_str = sess.run(summary_op, feed_dict={x: x_train_batch,
                                                          y: y_train_batch})
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush() 

        print "Optimization Finished!"
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        if v13:
            tf.summary.scalar("Accuracy on Train", accuracy)
        else:
            tf.scalar_summary("Accuracy on Train", accuracy)
        print "Accuracy:", accuracy.eval({x: x_test, y: y_test})
        #print "auc %f:" % auc.eval({x: x_test, y: y_test}) 
        
        saver.save(sess, modelsfile)
        

    return

def test(args):
    
    #csvからdataframeへ
    df = pd.read_csv( "./input/test.csv" )

    # 説明変数
    ngcol.extend([ cname for cname in df.columns if "_cat" in cname] )
    usecolumns = [ cname for cname in df.columns if cname not in ngcol ]
    x_np = np.array(df[usecolumns])


    # Network Parameters
    n_input = len(usecolumns)
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Construct model
    pred = multilayer_perceptron(x,n_input)
    y_ = tf.nn.softmax(pred)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, modelsfile )
            
        with open('tf_submit_cnn.csv', 'wt') as outf:
            fo = csv.writer(outf, lineterminator='\n')
            fo.writerow(['id','target'])
                
            #predicted = sess.run(pred, feed_dict={x:x_np})
            predicted = sess.run(y_, feed_dict={x:x_np})
            print predicted
            for i, p in enumerate(predicted):
                fo.writerow([df['id'][i],'%1.4f'%p[1]])
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', dest='mode', default='train', help='train or test')
    parser.add_argument('--epoch', type=int, dest='epoch', default=10, help='epoch number')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.exists('./models'):
            os.makedirs('./models')
        train(args)
    else:
        test(args)
    
