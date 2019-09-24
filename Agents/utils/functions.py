
import tensorflow as tf
import numpy as np
import math

def reward_to_go(rews):
    '''
    Calculating parameterized policy(weights)
    '''
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    #returns [200. 199. ... 1.]
    return rtgs

def create_model(obs_dim, n_acts, sizes=[32,], activation = tf.tanh):
    '''
    Creating model
    obs_dim = env.observation_space.shape -> (4,), use .shape[0]
    sizes [hidden layers] + [env.action_space.n]
    ''' 
    assert obs_dim, "obs_dim has to be number, not {}".format(obs_dim)
    assert n_acts , "n_acts has to be number, not {}".format(n_acts)
    obs_ph = tf.placeholder(dtype = tf.float32, shape=(None, obs_dim))
    hidden = tf.layers.dense(obs_ph, units = sizes[0], activation = activation)
    for x in sizes[1:-1]:
        hidden = tf.layers.dense(hidden, units = x, \
                                     activation = activation)
    logits = tf.layers.dense(hidden, units = n_acts)

    #action = tf.argmax(input = logits, axis = 1) bad idea 
    #tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1) better

    return obs_ph, logits

def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Sample actions and compute log-probs of actions.

    Args:
        x: Input tensor of states. Shape [batch, obs_dim].

        a: Input tensor of actions. Shape [batch, act_dim].

        hidden_sizes: Sizes of hidden layers for action network MLP.

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer (action layer).

        action_space: A gym.spaces object describing the action space of the
            environment this agent will interact with.
    Returns:
        pi: A symbol for sampling stochastic actions from a Gaussian 
            distribution.

        logp: A symbol for computing log-likelihoods of actions from a Gaussian 
            distribution.

        logp_pi: A symbol for computing log-likelihoods of actions in pi from a 
            Gaussian distribution.
    """
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

def categorical_loss_func_weights(logits, n_acts):
    """Calculating loss, returning specific placeholders
     Args:
        logits: NN that needs to be optimized

        n_acts: num of acts

    Returns:
        weights_ph: placegolder for weights (None, ), tf.float32

        act_ph: placeholder for act samples (None, ) tf.int32

        loss: float32 number
    """
    weights_ph = tf.placeholder(shape=(None,), dtype = tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype = tf.int32)
        
    action_masks = tf.one_hot(act_ph, n_acts)#array([[0., 1.]...], dtype=float32)

    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    #array([-0.6952652,...], dtype=float32)
    loss = -tf.reduce_mean(weights_ph* log_probs)#9.812662
    return weights_ph, act_ph, loss

