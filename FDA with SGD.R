#This code shows the SGD optimization applied in FDA(nonparametic regression)
#I simulate a 1000 elements sample and the original function as y = sin(x) + 0.5 * sin(3 * x) + sigma
#Then with Fourier basis functions, I choose stochastic data from the sample and calculate the gradient
#And make the parameter descent with the gradient


# set seed
set.seed(123)

# sample number
n = 1000  
x = seq(0, 2 * pi, length.out = n) 

# simulate a function
f = function(x) {sin(x) + 0.5 * sin(3 * x)} 

y_true = f(x)
sigma = 1  # noise 
y = y_true + rnorm(n, mean = 0, sd = sigma)

# store the simulate data
data = data.frame(x = x, y = y)

#function for basis
fourier_basis = function(x, K) {
  Phi_list = lapply(1:K, function(k) cbind(sin(k * x), cos(k * x)))
  Phi = do.call(cbind, Phi_list)
  return(Phi)
}

#calculate the basis matrix
K = 10
Phi = fourier_basis(data$x, K)


#second derivative of basis function
fourier_basis_2nd_derivative = function(x, K) {
  Phi_2nd_list = lapply(1:K, function(k) cbind(-k^2 * sin(k * x), -k^2 * cos(k * x)))
  Phi_2nd = do.call(cbind, Phi_2nd_list)
  return(Phi_2nd)
}
Phi_2nd = fourier_basis_2nd_derivative(data$x, K)


# regularization
R = diag(2 * K)  # 简单的单位矩阵，可以根据需要修改


# define loss function
loss_function = function(beta, phi, y, lambda, R, K) {
  n = nrow(phi)
  mse = 0
  for(i in 1 : n){
    
      mse = mse + (y[i] - phi[i,] * beta)^2
    
  }
  mse = mse / n
  regularization = lambda * t(beta) %*% R %*% beta
  return(mse + regularization)
}


# gradient of loss function
gradient = function(beta, phi, y, lambda, R, K) {
  n = nrow(phi)
  mse_grad = 0
  
  for(i in 1 : n){
      mse_grad = mse_grad + (-2 * (y[i] - phi[i, ] * beta))
  }
  mse_grad = mse_grad/n
  regularization_grad = 0
  for(i in 1 : 2 * K){
    for(j in 1 :2 * K){
      regularization_grad = regularization_grad + 2 * beta[i] * R[i,j]
    }
  }
  return(mse_grad + regularization_grad)
}

# SGD parametor
learning_rate = 0.01
epochs = 1000
lambda = 0.1  
n_samples = 100  # sample number to choose

# initialize beta
beta = runif(2 * K, -0.5, 0.5)

# SGD algorithm
for (epoch in 1:epochs) {
  
  # stochastic sample
  i = sample(1:n, 10)
  phi = Phi[i, , drop = FALSE]
  yi = y[i]
  
  # gradient
  grad <- gradient(beta, phi, yi, lambda, R, K)
  
  # update beta
  I <- matrix(1, nrow = 20, ncol = 1)
  beta <- beta - learning_rate * grad * I
}

# prediction with optimization parameter
y_pred <- Phi %*% beta

# plot the outcome
plot(data$x, data$y, col = 'blue', main = 'Fourier Basis Nonparametric Estimation with SGD')
lines(data$x, y_pred, col = 'green', lwd = 2)
lines(data$x, y_true, col = 'red', lwd = 2)
legend('topright', legend = c('Noisy Data', 'SGD Fit', 'True Function'), col = c('blue', 'green', 'red'), lwd = c(1, 2, 2))




### Above I use SGD to optimazation the nonparameter regression with each experiment I choose 50 stochastic data from the sample
### Now I want to compare the loss when the number of data I choose was different from 10, 50, 100, 500
### In the end, I depict a figure to vitually reflect the loss in different number of stochastic sample                         

                        
# SGD parameter
learning_rate <- 0.01
epochs <- 1000
lambda <- 0.1  
sample_sizes <- c(10, 50, 100, 200, 300, 400, 500)  # different tries of number of sample

# errors or loss
errors <- numeric(length(sample_sizes))

# run the experiment
for (j in seq_along(sample_sizes)) {
  n_samples <- sample_sizes[j]
  beta <- runif(2 * K, -0.5, 0.5)
  
  # SGD
  for (epoch in 1:epochs) {
  
    i = sample(1:n, n_samples)
    phi = Phi[i, , drop = FALSE]
    yi = y[i]
    
    # gradient
    grad = gradient(beta, phi, yi, lambda, R, K)
    
    # update beta
    I = matrix(1, nrow = 20, ncol = 1)
    beta = beta - learning_rate * grad * I
  }
  
  # calculate the loss
  y_pred = Phi %*% beta
  mse = mean((y - y_pred)^2)
  errors[j] = loss_function(beta, Phi, y, lambda, R, K)
}

# plot the out come
plot(sample_sizes, errors, type = 'b', col = 'blue', pch = 19, xlab = 'Sample Size', ylab = 'Loss', main = 'Loss vs Sample Size')
