### In this code, I want to simulate an online functional data
### Through the method mentioned in "Online Estimation for Functional Data"
### I want to achieve a efficient method even when the data base increases compared with traditional batch method


### Noticed that This code is for a dense data situation, you can change the parameter I set to achieve your goal




K_max = 1000 # The total number of data block
L = 5 # the number of candidate band-widths
i_k = 10 # the number of subject observed in the k_th data block
j_i_k = 100 # the number of measurement for i_th subject in the k_th data block
### For dense data, we need to make sure that m_i >> n_k^(5/4). And the parameter above can be adjusted by your wish


### mean function
mu_function = function(i){
  mu = matrix(0, nrow = i, ncol = 1)
  for(m in 1 : i){
    mu[m] = 2 * sin(2 * pi * t[m])
  }
  return(mu)
}


### stochastic fuction
stochastic_function = function(i, j){
  c = matrix(0, nrow = j, ncol = 1)
  for(m in 1 : j){
    c[m] = rnorm(1, mean = 0, sd = sqrt( 0.4 * m^(-2)))
  }
  stoch = matrix(0, nrow = i, ncol = 1)
  phi_function = matrix(0, nrow = i, ncol = j)
  phi_function[,1] = 1
  for(m in 1 : i){
    for(n in 2 : j){
      phi_function[m, n] = ( sqrt(2) * cos((n - 1) * pi * t[m]))
    }
  }
  stoch = phi_function %*% c
  return(stoch)
}


### simulate data function
data_function = function(mu, stoch, i){
  x_t = mu + stoch
  error = rnorm(j_i_k, mean = 0, sd = 0.5)
  y_t = x_t + error
  return(data.frame(t = t, y_t = y_t))
}


### create all data we need
all_data = vector("list", K_max)
for (k in 1:K_max) {
  t = runif(j_i_k, min = 0, max = 1)
  mu = mu_function(j_i_k)
  stoch = stochastic_function(j_i_k, i_k)
  new_data = data_function(mu, stoch, j_i_k)
  all_data[[k]] = new_data
}


### visualize the data we simulated
# Combine all data blocks into one large data frame
combined_data = do.call(rbind, all_data)
# Plot the combined data
plot(combined_data$t, combined_data$y_t, 
     type = "p",                     # plot style
     pch = 16,                       # point with dense kernel
     cex = 0.5,                      # size of the point
     col = rgb(0, 0, 1, alpha = 0.2),# color
     main = "Simulated Functional Data",
     xlab = "Time", 
     ylab = "Value")


### Now that we have a simulation for the functional data we need to research
### next we want to find a better way to estimate our model
### I apply SGD algorithm to estimate the model from the 1st data_block to the K_max st data-block
### 


fourier_basis = function(x, K) {
  Phi_list = lapply(1:K, function(k) cbind(sin(k * x), cos(k * x)))
  Phi = do.call(cbind, Phi_list)
  return(Phi)
}


# Fourier basis & estimation

estimation = function(K, learning_rate, K_max) {
  beta = runif(2 * K, -0.5, 0.5)
  loss_value = numeric(K_max)
  for(i in 1:K_max) {
    data = all_data[[i]]
    n = length(data$t)
    sample = sample(1:n, 20)
    yi = data$y_t[sample]
    xi = data$t[sample]
    basis_matrix = fourier_basis(xi, K) 
    y_pred = basis_matrix %*% beta
    loss <- mean((yi - y_pred)^2)
    loss_value[i] = loss
    grad = (-2/length(y_pred)) * t(basis_matrix) %*% (yi - y_pred)
    beta <- beta - learning_rate * grad
  }
  return(list(beta = beta, loss_value = loss_value))
}

  
K = 5

learning_rate = 0.01
est = estimation(K, learning_rate, K_max)


data = all_data[[1]]
Basis = fourier_basis(all_data[[1]]$t ,5)
Y = Basis %*% est$beta
plot(all_data[[1]]$t, all_data[[1]]$y_t, 
     type = "p",                     
     pch = 16,                       
     cex = 0.5,                      
     col = rgb(0, 0, 1, alpha = 0.2),
     main = "Fitted Curve vs Actual Data",
     xlab = "Time", 
     ylab = "Value")
ord <- order(data$t)
lines(all_data[[1]]$t[ord], Y[ord], col = "red", lwd = 2)



plot(1:K_max, est$loss_value, type = "l", col = "blue", lwd = 2,
     main = "Loss over Iterations",
     xlab = "Iteration", ylab = "Loss")
loess_fit = loess(est$loss_value ~ I(1:K_max), span = 0.1)  
lines(1:K_max, predict(loess_fit), col = "red", lwd = 2)

legend("topright", legend = c("Loss", "LOESS Trend"), col = c("blue", "red"), lty = 1, lwd = 2)
