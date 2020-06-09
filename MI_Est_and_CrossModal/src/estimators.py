import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from functools import partial

from src.utils import *

import random

##############################################################
#
# Estimator for log {p(x,y)/p(x)p(y)} & Lower Bound for MI
#
# #############################################################
# optimal critic f(x,y) = log {p(x,y)/p(x)p(y)} + 1
def nwj_lower_bound_obj(scores):
    return tuba_lower_bound(scores - 1.)

# optimal critic = log {p(x,y)/p(x)p(y)}
# mine bound has extremely high variance in practice
def mine_lower_bound(f, buffer=None, momentum=0.9):
    if buffer is None:
        buffer = torch.tensor(1.0).cuda()
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update


# optimal critic f(x,y) = log {p(x,y)/p(x)p(y)}
def js_fgan_lower_bound_obj(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


##############################################################
#
# Estimator for log {p(x,y)/p(x)p(y)} & Approximation for MI
#
# #############################################################
# optimal critic = log {p(x,y)/p(x)p(y)}
def dv_upper_lower_bound_obj(f):
    """DV lower bound, but upper bounded by using log outside."""
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    return first_term - second_term

'''
def regularized_dv_bound(f, l=0.0):
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    reg_term = l * (second_term.exp() - 1) ** 2

    with torch.no_grad():
        reg_term_no_grad = reg_term

    return first_term - second_term + reg_term - reg_term_no_grad
'''

##############################################################
#
# Lower Bound for MI
#
# #############################################################
def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    batch_size = scores.size(0)

    # First term is an expectation over samples from the joint,
    # which are the diagonal elmements of the scores matrix.
    joint_term = scores.diag().mean()

    # Second term is an expectation over samples from the marginal,
    # which are the off-diagonal elements of the scores matrix.
    marg_term = logmeanexp_nodiag(scores).exp()
    return 1. + joint_term - marg_term

def infonce_lower_bound_obj(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Alternative implementation:
    # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi

##############################################################
#
# Approximation for MI
#
# #############################################################
# # if the input has no log form
def log_density_ratio_mi(f):
    return torch.log(torch.clamp(f, min=1e-4)).diag().mean()
   
# if the input is in log form
def direct_log_density_ratio_mi(f):
    return f.diag().mean()   
    
def dv_clip_upper_lower_bound(f, alpha=1.0, clip=None):
    z = renorm_q(f, alpha, clip)
    dv_clip = f.diag().mean() - z

    return dv_clip
    


##############################################################
# #############################################################
# #############################################################
# #############################################################
#
# Training & Evaluation
#
# #############################################################
# #############################################################
# #############################################################
# #############################################################
def MI_Estimator(f, train_type='nwj_lower_bound_obj', eval_type='nwj_lower_bound_obj',
                                  **kwargs):   
    # A little hacky here..... since we use if/else to control the train/eval_type
    # Some methods are deriving for log_density_ratio
    # other methods are deriving for density_ratio
    
    if train_type == 'tuba_lower_bound' or train_type == 'mine_lower_bound':
        assert train_type == eval_type
        train_val = getattr(sys.modules[__name__], train_type)(f, **kwargs)
    else:
        train_val = getattr(sys.modules[__name__], train_type)(f)
    
    if train_type == eval_type:
        return train_val, train_val
    
    if train_type == 'nwj_lower_bound_obj' and eval_type == 'direct_log_density_ratio_mi':
        eval_val = getattr(sys.modules[__name__], eval_type)(f-1.) 
    elif eval_type == 'tuba_lower_bound' or eval_type == 'dv_clip_upper_lower_bound'\
                                                      or eval_type == 'mine_lower_bound':
        eval_val = getattr(sys.modules[__name__], eval_type)(f, **kwargs)
    # note that especially when we use JS to train, and use nwj to evaluate
    elif eval_type == 'nwj_lower_bound_obj':
        eval_val = getattr(sys.modules[__name__], eval_type)(f+1., **kwargs)
    else:
        eval_val = getattr(sys.modules[__name__], eval_type)(f)
    
    with torch.no_grad():
        eval_train = eval_val - train_val
        
    return train_val + eval_train, train_val


###############################################################
# #############################################################
#
# Baselines
#
# #############################################################
# ##############################################################
def nwj_lower_bound(f):
    return MI_Estimator(f, train_type='nwj_lower_bound_obj', eval_type='nwj_lower_bound_obj')

def infonce_lower_bound(f):
    return MI_Estimator(f, train_type='infonce_lower_bound_obj', eval_type='infonce_lower_bound_obj')

def js_lower_bound(f):
    return MI_Estimator(f, train_type='js_fgan_lower_bound_obj', eval_type='nwj_lower_bound_obj')

def dv_upper_lower_bound(f):
    return MI_Estimator(f, train_type='dv_upper_lower_bound_obj', eval_type='dv_upper_lower_bound_obj')

def smile_lower_bound(f, alpha=1.0, clip=5.0):
    return MI_Estimator(f, train_type='js_fgan_lower_bound_obj', 
                        eval_type='dv_clip_upper_lower_bound', alpha=alpha, clip=clip)

###############################################################
# #############################################################
#
# Proposed Methods
#
# #############################################################
# ##############################################################
# # Proposed Method 1: Variational Representation of f-divergence (NWJ)
def variational_f_nwj(f):
    return MI_Estimator(f, train_type='nwj_lower_bound_obj', 
                       eval_type='direct_log_density_ratio_mi')

## Proposed Method 1: Variational Representation of f-divergence (DV)
def variational_f_dv(f):
    return MI_Estimator(f, train_type='dv_upper_lower_bound_obj',
                       eval_type='direct_log_density_ratio_mi')

## Proposed Method 1: Variational Representation of f-divergence (JS)
def variational_f_js(f):
    return MI_Estimator(f, train_type='js_fgan_lower_bound_obj', 
                       eval_type='direct_log_density_ratio_mi')

## Proposed Method 2: Density Matching
def density_matching(f):
    # we change the variables here (ratio -> log_ratio) because of non-negativity
    return MI_Estimator(f, train_type='density_matching_obj',
                        eval_type='direct_log_density_ratio_mi')
    #return MI_Estimator(f, train_type='density_matching_loss',
    #                    eval_type='log_density_ratio_mi')
def density_matching_lagrange(f):
    # we change the variables here (ratio -> log_ratio) because of non-negativity
    return MI_Estimator(f, train_type='density_matching_lagrange_obj',
                        eval_type='direct_log_density_ratio_mi')
    #return MI_Estimator(f, train_type='density_matching_loss',
    #                    eval_type='log_density_ratio_mi')

## Proposed Method 3: Probabilistic Classifier
def probabilistic_classifier(f):
    return MI_Estimator(f, train_type='probabilistic_classifier_obj',
                        eval_type='probabilistic_classifier_eval')


## Proposed Method 5: Density-Ratio Fitting
def density_ratio_fitting(f):
    #return MI_Estimator(f, train_type='density_ratio_fitting_obj', 
    #                   eval_type='log_density_ratio_mi')
    return MI_Estimator(f, train_type='density_ratio_fitting_obj', 
                       eval_type='log_density_ratio_mi')

def squared_mutual_information(f):
    return MI_Estimator(f, train_type='density_ratio_fitting_obj', 
                       eval_type='squared_mutual_information_eval')
    
def js_squared_mutual_information(f):
    return MI_Estimator(f, train_type='js_fgan_lower_bound_obj', 
                       eval_type='squared_mutual_information_from_log_eval')

###############################################################
# #############################################################
#
# Proposed Method Helper Functions
#
# #############################################################
# ##############################################################

#### Density-Ratio Fitting
# optimal critic f(x,y) = p(x,y)/p(x)p(y)
def density_ratio_fitting_obj(f, l=0.0, relative_ratio=0.0):
    f_square = f ** 2
    n = f.size(0)
    
    joint_term = f.diag().mean()
    
    marg_term = ((f_square.sum() - f_square.diag().sum()) /
                 (n*(n-1.)))
    
    if not l == 0.0:
        normalization_contraint = (f.sum() - f.diag().sum()) / (n * (n-1.))

        reg_term = l * (normalization_contraint - 1)**2
    else:
        reg_term = 0.0
        
    if not relative_ratio == 0.0:
        joint_term = joint_term - 0.5*relative_ratio*f_square.diag().mean()
        marg_term = (1.-relative_ratio)*marg_term
        
    return joint_term - 0.5*marg_term - reg_term

def squared_mutual_information_from_log_eval(f):
    joint_term = f.diag().exp().mean()
    
    return 0.5*joint_term - 0.5

def squared_mutual_information_eval(f):
    joint_term = f.diag().mean()
    
    return 0.5*joint_term - 0.5

#### Density Matching
def density_matching_obj(f, l=1.0):
    # l is the regularization coefficient
    # we change the variables here (ratio -> log_ratio) because of non-negativity
    joint_term = f.diag().mean()

    n = f.size(0)
    marg_term = (f.exp().sum() - f.exp().diag().sum()) / (n * (n-1.))
    
    marg_term = torch.clamp(marg_term, 1e-4)
    marg_term = marg_term.log()

    #reg_term = l * (marg_term - 1)**2
    reg_term = l * (marg_term)**2


    # minimizing reg_term equals to maximizing -reg_term
    return joint_term - reg_term

def density_matching_lagrange_obj(f, l=1.0):
    # l is the regularization coefficient
    # we change the variables here (ratio -> log_ratio) because of non-negativity
    joint_term = f.diag().mean()

    n = f.size(0)
    marg_term = (f.exp().sum() - f.exp().diag().sum()) / (n * (n-1.))
    
    #marg_term = marg_term.log()

    #reg_term = l * (marg_term - 1)**2
    #reg_term = l * (marg_term)**2
    reg_term = marg_term - 1.


    # minimizing reg_term equals to maximizing -reg_term
    return joint_term - reg_term
    

#### probabilistic classifier
def probabilistic_classifier_obj(f):
    criterion = nn.BCEWithLogitsLoss()
    
    batch_size = f.shape[0]
    labels = [0.]*(batch_size*batch_size)
    labels[::(batch_size+1)] = [1.]*batch_size
    labels = torch.tensor(labels).type_as(f)
    labels = labels.view(-1,1)

    logits = f.contiguous().view(-1,1)

    Loss = -1.*criterion(logits, labels)

    return Loss
def probabilistic_classifier_eval(f):
    batch_size = f.shape[0]
    joint_feat = f.contiguous().view(-1)[::(batch_size+1)]
    joint_logits = torch.sigmoid(joint_feat)

    MI = torch.mean(torch.log((batch_size-1)*joint_logits/(1.-joint_logits)))
    # we have batch_size*(batch_size-1) product of marginal samples
    # we have batch_size joint density samples 

    return MI
