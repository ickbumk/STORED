import torch
import time
import gpytorch
import gc
import numpy as np
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RQKernel, LinearKernel,ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval

class GPModel(ExactGP):
    def __init__(self,train_x,train_y,likelihood):
        super(GPModel, self).__init__(train_x,train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = LinearKernel()+ScaleKernel(RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def tune_gpr(device, normalized_xyz_outlier, GP_model, n_sample = 500, n_iter = 100, min_iter = 10, convergence_tol = 1e-2, early_stopping_patience = 3):

    
    
    random_int = np.random.randint(0, normalized_xyz_outlier.shape[0],n_sample)
    
    train_x = torch.tensor(normalized_xyz_outlier[random_int,:2]).to(device)
    train_y = torch.tensor(normalized_xyz_outlier[random_int,2]).to(device)

    noises = torch.ones(n_sample)*0.0001
    likelihood = FixedNoiseGaussianLikelihood(noise = noises, learn_additional_noise=False).to(device)
    best_val_loss = float('inf')
    previous_loss = float('inf')

    model = GP_model(train_x, train_y, likelihood).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    start_time = time.time()

    
    for i in range(n_iter):         
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if i > min_iter:
    
            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
    
            if patience_counter >= early_stopping_patience:
                print(f'Earling stopping with patience loss at iteration {i+1}')
                break
            if abs(previous_loss - loss.item()) < convergence_tol:
                print(f"Converged at iteration {i+1}")
                break

            previous_loss = loss.item()
                
    print('Tuning took:', i+1, 'iterations.')
    param_list = []
    for param_name, param in model.named_parameters():
        param_list.append(param.item())

        # Uncomment if you want to print
        # print(f'Parameter name: {param_name:42} value = {param.item()}')


    end_time = time.time()

    torch.cuda.empty_cache()
    gc.collect()

    print('Time taken for tuning is: ', end_time-start_time , 'seconds')

    return param_list, model

def exact_gp(device,normalized_xyz_outlier, GPModel, hyp_opt, n_sample = 500, num_epochs = 2, pred_batch = False, batch_size = False, pred_dataset = []):

    
    constant, linear, outputscale, lengthscale, alpha = optimized_hyperparameter = hyp_opt
    
    
    train_xy, train_z = torch.tensor(normalized_xyz_outlier[:,:2]).contiguous(), torch.tensor(normalized_xyz_outlier[:,2]).contiguous()
    train_xy, train_z = train_xy.to(device), train_z.to(device)
    

    if len(pred_dataset) == 0:
        print('Predicting with the initial train dataset')
        pred_xy = train_xy.clone()
    else:
        pred_xy = torch.tensor(pred_dataset[:,:2]).to(device)
        print('Predicting with the given prediction dataset of size:', pred_dataset.shape)
        

    
    if batch_size == False:
        batch_size = np.ceil(train_z.shape[0]/n_sample).astype(int)
        
    integ = np.ceil(pred_xy.shape[0]/20000)
    pred_lin = np.linspace(0,20000*(integ-1),int(integ)).astype(int)
    
    mean_list = []
    var_list = []


    start_time = time.time()
    
    
    for i in range(num_epochs):
        
        for _ in range(batch_size):
    
            random_int = np.random.randint(0, train_z.shape[0],n_sample)
    
            x_batch = train_xy[random_int,:]
            y_batch = train_z[random_int]
            
    
            noises = torch.full_like(y_batch, 0.0001)

            
    
            
            likelihood = FixedNoiseGaussianLikelihood(noise = noises, learn_additional_noise=False).to(device)

            tuned_model = GPModel(x_batch, y_batch, likelihood)

    
            tuned_model.mean_module.register_parameter(name ='raw_constant', parameter = torch.nn.Parameter(torch.tensor(constant)))
            tuned_model.covar_module.register_parameter(name ='raw_outputscale', parameter = torch.nn.Parameter(torch.tensor(outputscale)))
            tuned_model.covar_module.kernels[0].register_parameter(name = 'raw_variance', parameter = torch.nn.Parameter(torch.tensor(linear)))
            tuned_model.covar_module.kernels[1].base_kernel.register_parameter(name ='raw_lengthscale', parameter = torch.nn.Parameter(torch.tensor(lengthscale)))
            tuned_model.covar_module.kernels[1].base_kernel.register_parameter(name ='raw_alpha', parameter = torch.nn.Parameter(torch.tensor(alpha)))
    
            tuned_model.to(device)
            
            tuned_model.train()
            likelihood.train()
            
            optimizer = torch.optim.Adam(tuned_model.parameters(), lr = 0.1)
            
            mll = ExactMarginalLogLikelihood(likelihood, tuned_model)

    
            optimizer.zero_grad()
            output = tuned_model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            
    
            if pred_batch == True:
            #  # Make predictions
                tuned_model.eval()
                likelihood.eval()
        
        
                mean_iter = []
                var_iter = []
                
               
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
        
                    for start_ind in pred_lin:
                        
                        test_noises = torch.full_like(pred_xy[start_ind:start_ind+20000,0],0.0001)
                        observed_pred = likelihood(tuned_model(pred_xy[start_ind:start_ind+20000,:]), noise = test_noises)
                        mean_iter.append(observed_pred.mean.cpu().numpy())
                        var_iter.append(observed_pred.variance.cpu().numpy())
        
                mean_list.append(np.hstack(mean_iter))
                var_list.append(np.hstack(var_iter))
                continue


            tuned_model.eval()
            likelihood.eval()
            
            # Make predictions
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_noises = torch.full_like(pred_xy[:,0],0.0001)
                observed_pred = likelihood(tuned_model(pred_xy), noise = test_noises)
 
            mean_cpu = observed_pred.mean.cpu().numpy()
            var = observed_pred.variance.cpu().numpy()
    
            mean_list.append(mean_cpu)
            var_list.append(var)

    end_time = time.time()

    torch.cuda.empty_cache()
    gc.collect()
    
    print('GPR iteration is :', batch_size)
    print('Time taken for prediction is :', end_time - start_time, 'seconds')

    return mean_list, var_list

def score(prediction_all, std_all, normalized_xyz_outlier, strict = 3):

    score_list = []
    
    for it in range(len(prediction_all)):
        pred_iter = prediction_all[it]
        std_iter = std_all[it]
    
        lower = pred_iter - strict*std_iter
        upper = pred_iter + strict*std_iter
    
        score_iter = (normalized_xyz_outlier[:,2]<lower)|(normalized_xyz_outlier[:,2]>upper)
    
        score_list.append(score_iter)
    
    score_final = np.mean(score_list, axis = 0)*100

    return score_final

def weighted_pred(prediction_all, std_all):
    std_without_outliers = np.vstack(std_all)
    pred_without_outliers = np.vstack(prediction_all)
    var_without_outliers = std_without_outliers**2
    
    var_without_outliers = std_without_outliers**2
    
    weights = 1/var_without_outliers
    
    weighted_prediction = np.sum(pred_without_outliers*weights, axis = 0)/np.sum(weights, axis = 0)

    return weighted_prediction