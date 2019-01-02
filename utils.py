import sonnet as snt


def load_linear(linear, dic, sess, bias=False):
    linear.w.load(dic['w'], sess)
    if bias:
        linear.b.load(dic['b'], sess)


def load_batchnorm(batchnorm, dic, sess):
    batchnorm.beta.load(dic['beta'], sess)
    batchnorm.gamma.load(dic['gamma'], sess)
    batchnorm.moving_mean.load(dic['moving_mean'], sess)
    batchnorm.moving_variance.load(dic['moving_variance'], sess)


def save_linear(linear, sess, bias=False):
    dic = {'w': linear.w.eval(sess)}
    if bias:
        dic['b'] = linear.b.eval(sess)
    return dic


def save_batchnorm(batchnorm, sess):
    return {'beta': batchnorm.beta.eval(sess),
            'gamma': batchnorm.gamma.eval(sess),
            'moving_mean': batchnorm.moving_mean.eval(sess),
            'moving_variance': batchnorm.moving_variance.eval(sess)}
