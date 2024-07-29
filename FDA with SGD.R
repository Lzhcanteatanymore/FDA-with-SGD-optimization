# set seed
set.seed(123)

# sample number
n <- 1000  
x <- seq(0, 2 * pi, length.out = n) 

# simulate a function
f <- function(x) {sin(x) + 0.5 * sin(3 * x)} 

y_true <- f(x)
sigma <- 1  # noise 
y <- y_true + rnorm(n, mean = 0, sd = sigma)

# store the simulate data
data <- data.frame(x = x, y = y)

#plot the figure of x and y
plot(x, y, col = 'blue', main = 'Generated Data with Noise')
lines(x, y_true, col = 'red', lwd = 2)
legend('topright', legend = c('Noisy Data', 'True Function'), col = c('blue', 'red'), lwd = c(1, 2))

#function for basis
fourier_basis <- function(x, K) {
  Phi_list <- lapply(1:K, function(k) cbind(sin(k * x), cos(k * x)))
  Phi <- do.call(cbind, Phi_list)
  return(Phi)
}

#calculate the basis matrix
K <- 10
Phi <- fourier_basis(data$x, K)

# 定义蒙特卡洛密度函数估计
density_function <- function(x, data, n_samples) {
  # 从数据中随机抽取样本
  sampled_data <- sample(data$x, n_samples, replace = TRUE)
  # 估计密度函数（这里用核密度估计）
  density_est <- density(sampled_data, bw = "nrd0")
  # 返回估计的密度
  return(approx(density_est$x, density_est$y, xout = x, rule = 2)$y)
}


#second derivative of basis function
fourier_basis_2nd_derivative <- function(x, K) {
  Phi_2nd_list <- lapply(1:K, function(k) cbind(-k^2 * sin(k * x), -k^2 * cos(k * x)))
  Phi_2nd <- do.call(cbind, Phi_2nd_list)
  return(Phi_2nd)
}
Phi_2nd <- fourier_basis_2nd_derivative(data$x, K)


#R <- t(Phi_2nd) %*% Phi_2nd / n


# regularization
R <- diag(2 * K)  # 简单的单位矩阵，你可以根据需要修改


# define loss function
loss_function <- function(beta, phi, y, lambda, p, R, K) {
  n <- nrow(phi)
  mse = 0
  for(i in 1 : n){
    for(j in 1 : 2 * K){
      mse = mse + (y[i]-beta[j] * phi[i,j])^2
    }
    
  }
  mse=mse/n
  regularization <- lambda * t(beta) %*% R %*% beta
  return(mse + regularization)
}


# 定义损失函数的梯度
gradient <- function(beta, phi, y, lambda, p, R, K) {
  n <- nrow(phi)
  mse_grad = 0
  for(i in 1 : n){
    for(j in i : 2 * K){
      mse_grad = mse_grad + (-2 * (y[i]-beta[j] * phi[i,j]))
    }
  }
  mse_grad=mse_grad/n
  regularization_grad = 0
  for(i in 1 : 2 * K){
    for(j in 1 :2 * K){
      regularization_grad = regularization_grad + 2 * beta[i] * R[i,j]
    }
  }
  return(mse_grad + regularization_grad)
}

# SGD参数
learning_rate <- 0.01
epochs <- 1000
lambda <- 0.1  # 正则化参数
n_samples <- 100  # 蒙特卡洛模拟的样本数

# 初始化beta随机
beta <- runif(2 * K, -0.5, 0.5)

# SGD算法
for (epoch in 1:epochs) {
  # 随机抽取样本
  i <- sample(1:n, 50)
  phi <- Phi[i, , drop = FALSE]
  yi <- y[i]
  p <- sapply(data$x[i], function(xi) density_function(xi, data, n_samples))
  
  # 计算梯度
  grad <- gradient(beta, phi, y, lambda, p, R, K)
  
  # 更新beta
  beta <- beta - learning_rate * grad
}

# 预测
y_pred <- Phi %*% beta

# 绘制结果
plot(data$x, data$y, col = 'blue', main = 'Fourier Basis Nonparametric Estimation with Density-aware SGD')
lines(data$x, y_pred, col = 'green', lwd = 2)
lines(data$x, y_true, col = 'red', lwd = 2)
legend('topright', legend = c('Noisy Data', 'SGD Fit', 'True Function'), col = c('blue', 'green', 'red'), lwd = c(1, 2, 2))






# SGD参数
learning_rate <- 0.01
epochs <- 1000
lambda <- 0.1  # 正则化参数
sample_sizes <- c(10, 50, 100)  # 不同样本数量

# 用于保存误差
errors <- numeric(length(sample_sizes))

# 遍历不同的样本数量
for (j in seq_along(sample_sizes)) {
  n_samples <- sample_sizes[j]
  beta <- runif(2 * K, -0.5, 0.5)
  
  # SGD算法
  for (epoch in 1:epochs) {
    # 随机抽取多个样本
    i <- sample(1:n, n_samples)
    Xi <- Phi[i, , drop = FALSE]
    yi <- y[i]
    p <- sapply(data$x[i], function(xi) density_function(xi, data, n_samples))
    
    # 计算梯度
    grad <- gradient(beta, Xi, yi, lambda, p, R)
    
    # 更新beta
    beta <- beta - learning_rate * grad
  }
  
  # 计算预测误差
  y_pred <- Phi %*% beta
  mse <- mean((y - y_pred)^2)
  errors[j] <- mse
}

# 绘制结果
plot(sample_sizes, errors, type = 'b', col = 'blue', pch = 19, xlab = 'Sample Size', ylab = 'Mean Squared Error', main = 'Error vs Sample Size')
