import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GNNModel
from opt import Optimizer

def PredictScore(train_drug_side_matrix, drug_matrix, side_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_side_matrix, drug_matrix, side_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_side_matrix.sum()
    X = constructNet(train_drug_side_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_side_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))
    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }

    model = GNNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_side_matrix.shape[0], name='AGRL-DSE')

    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            lr=lr, num_u=train_drug_side_matrix.shape[0], num_v=train_drug_side_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

def cross_validation_experiment(drug_side_matrix, drug_matrix, side_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(drug_side_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp

    basic_metrics_list = []
    additional_metrics_list = []
    print("seed=%d, evaluating drug-side...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(drug_side_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_side_matrix.shape[0]
        side_len = drug_side_matrix.shape[1]
        drug_side_res = PredictScore(
            train_matrix, drug_matrix, side_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = drug_side_res.reshape(drug_len, side_len)
        basic_metrics_tmp, additional_metrics_tmp = cv_model_evaluate(
            drug_side_matrix, predict_y_proba, train_matrix)
        print(basic_metrics_tmp, additional_metrics_tmp)
        basic_metrics_list.append(basic_metrics_tmp)
        additional_metrics_list.append(additional_metrics_tmp)
        del train_matrix
        gc.collect()

    average_basic_metrics = np.mean(basic_metrics_list, axis=0)
    average_additional_metrics = {key: np.mean([metrics[key] for metrics in additional_metrics_list]) for key in additional_metrics_list[0]}

    average_metrics = {
        'aupr': average_basic_metrics[0],
        'auc': average_basic_metrics[1],
        'f1_score': average_basic_metrics[2],
        'accuracy': average_basic_metrics[3],
        'recall': average_basic_metrics[4],
        'specificity': average_basic_metrics[5],
        'precision': average_basic_metrics[6],
        'rmse': average_basic_metrics[7],
        'mae': average_basic_metrics[8],
        **average_additional_metrics
    }

    print(average_metrics)
    return average_metrics

if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/drug-similarity.csv', delimiter=',')
    side_sim = np.loadtxt('../data/side-similarity.csv', delimiter=',')
    drug_side_matrix = np.loadtxt('../data/drug_side.csv', delimiter=',')
    epoch = 6000
    emb_dim = 128
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    simw = 6
    result = []
    average_result = {}
    circle_time = 1
    for i in range(circle_time):
        result.append(cross_validation_experiment(
            drug_side_matrix, drug_sim*simw, side_sim*simw, i, epoch, emb_dim, dp, lr, adjdp))

    for key in result[0]:
        average_result[key] = np.mean([res[key] for res in result])

    print(average_result)

    print("AUC:", average_result['auc'])
    print("ACC:", average_result['accuracy'])
    print("recall:", average_result['recall'])
    print("specificity:", average_result['specificity'])
    print("precision:", average_result['precision'])
    print("RMSE:", average_result['rmse'])
    print("MAE:", average_result['mae'])
    print("MAP:", average_result['map'])
    print("recall@1:", average_result['recall@1'])
    print("recall@15:", average_result['recall@15'])
    print("precision@1:", average_result['precision@1'])
    print("precision@15:", average_result['precision@15'])

