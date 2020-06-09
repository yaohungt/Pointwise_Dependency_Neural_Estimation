import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import *
from src.estimators import *
from src.models import *

def estimate_mutual_information(estimator, x, y, critic_fn,
                                baseline_fn=None, alpha_logit=None, clamping_values=None, **kwargs):
    """Estimate variational lower bounds on mutual information.

  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated', ...
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input 
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound

  Returns:
    scalar estimate of mutual information, train_loss
    # note that train_loss may not be MI estimation
    """
    x, y = x.cuda(), y.cuda()
    scores = critic_fn(x, y)
    if clamping_values is not None:
        scores = torch.clamp(scores, min=clamping_values[0], max=clamping_values[1])
    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
    if estimator == 'nwj':
        return nwj_lower_bound(scores)
    elif estimator == 'infonce':
        return infonce_lower_bound(scores)
    elif estimator == 'js':
        return js_lower_bound(scores)
    elif estimator == 'dv':
        return dv_upper_lower_bound(scores)
    elif estimator == 'smile':
        return smile_lower_bound(scores, **kwargs)
    elif estimator == 'variational_f_js':
        return variational_f_js(scores)
    elif estimator == 'probabilistic_classifier':
        return probabilistic_classifier(scores)
    elif estimator == 'density_matching':
        return density_matching(scores)
    elif estimator == 'density_matching_lagrange':
        return density_matching_lagrange(scores)
    elif estimator == 'density_ratio_fitting':
        return density_ratio_fitting(scores)
    elif estimator == 'squared_mutual_information':
        return squared_mutual_information(scores)
    elif estimator == 'js_squared_mutual_information':
        return js_squared_mutual_information(scores) 

def train_estimator(critic_params, data_params, mi_params, opt_params, **kwargs):
    """Main training loop that estimates time-varying MI."""
    
    CRITICS = {
        'separable': SeparableCritic,
        'concat': ConcatCritic,
    }
    
    BASELINES = {
        'constant': lambda: None,
        'unnormalized': lambda: mlp(dim=data_params['dim'], \
                        hidden_dim=512, output_dim=1, layers=2, activation='relu').cuda(),
    }

    # Ground truth rho is only used by conditional critic
    critic = CRITICS[mi_params.get('critic', 'separable')](
        rho=None, **critic_params).cuda()
    baseline = BASELINES[mi_params.get('baseline', 'constant')]()
    
    opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])
    if isinstance(baseline, nn.Module):
        opt_base = optim.Adam(baseline.parameters(),
                              lr=opt_params['learning_rate'])
    else:
        opt_base = None

    def train_step(rho, data_params, mi_params):
        # Annoying special case:
        # For the true conditional, the critic depends on the true correlation rho,
        # so we rebuild the critic at each iteration.
        opt_crit.zero_grad()
        if isinstance(baseline, nn.Module):
            opt_base.zero_grad()

        if mi_params['critic'] == 'conditional':
            critic_ = CRITICS['conditional'](rho=rho).cuda()
        else:
            critic_ = critic

        x, y = sample_correlated_gaussian(
            dim=data_params['dim'], rho=rho,\
                batch_size=data_params['batch_size'], cubic=data_params['cubic'])
        if False:
            mi, p_norm = estimate_mutual_information(
                mi_params['estimator'], x, y, critic_, baseline,\
                    mi_params.get('alpha_logit', None), **kwargs)
        else:
            mi, train_obj = estimate_mutual_information(
                mi_params['estimator'], x, y, critic_, baseline,\
                    mi_params.get('alpha_logit', None), **kwargs)
        loss = -mi

        loss.backward()
        opt_crit.step()
        if isinstance(baseline, nn.Module):
            opt_base.step()
        
        if False:
            return mi, p_norm
        else:
            return mi, train_obj

    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

    if False:
        estimates = []
        p_norms = []
        for i in range(opt_params['iterations']):
            mi, p_norm = train_step(
                rhos[i], data_params, mi_params)
            mi = mi.detach().cpu().numpy()
            p_norm = p_norm.detach().cpu().numpy()
            estimates.append(mi)
            p_norms.append(p_norm)
        
        return np.array(estimates), np.array(p_norms)
    else:
        estimates = []
        train_objs = []
        for i in range(opt_params['iterations']):
            mi, train_obj = train_step(
                rhos[i], data_params, mi_params)
            mi = mi.detach().cpu().numpy()
            train_obj = train_obj.detach().cpu().numpy()
            estimates.append(mi)
            train_objs.append(train_obj)

        return np.array(estimates), np.array(train_objs)
