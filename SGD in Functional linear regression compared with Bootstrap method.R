t_func = function(tn) {
  return(seq(0, 1, length.out = tn))
}

beta_func = function(t) {
  return(0.2*t^11*(10*(1-t))^6+10*(10*t)^3*(1-t)^10)
}

x_func = function(n, tn) {
  t = seq(0, 1, length.out = tn)
  x = matrix(0, nrow = n, ncol = tn)
  for(i in 1:n){
    x[i,] = sin(2 * pi * i* t)
  }
  return(x)
}

integrate_term = function(x, beta, tn) {
  delta_t = 1 / (tn - 1)
  return(x %*% beta * delta_t)
}

integrate_fuc = function(x, tn, basis, K) {
  delta_t = 1 / (tn - 1)
  return(x %*% t(basis) * delta_t)
}

basis_func = function(t, tn, K) {
  Phi = rbind(1, matrix(0, nrow = 2 * K, ncol = tn))
  for (k in 1:K) {
    Phi[2*k, ] = sin(2 * pi * k * t)
    Phi[2*k + 1, ] = cos(2 * pi * k * t)
  }
  return(Phi)
}

penalty_matrix_exact <- function(K) {
  R = diag(0, nrow = 2 * K + 1, ncol = 2 * K + 1)
  for (k in 1:K) {
    R[2 * k, 2 * k] = (2 * pi * k)^4 / 2
    R[2 * k + 1, 2 * k + 1] = (2 * pi * k)^4 / 2
  }
  return(R)
}


######
n = 1000000
tn = 1000
K = 20
t = t_func(tn)
x = x_func(n, tn)
beta = matrix(beta_func(t), nrow = tn, ncol = 1)
integral_tr = integrate_term(x, beta, tn)
error = rnorm(n, sd = 0.01)
y = integral_tr + error
library(splines)

basis <- t(bs(t, df = K, degree = 3, intercept = TRUE))

Z = x %*% t(basis) * (1 / (tn - 1))
basis_d2 <- predict(basis, deriv = 2)
delta_t <- 1 / (tn - 1)
R <- basis_d2 %*% t(basis_d2) * delta_t
print(R)

######
estimation = function(K, r, n, Z, lambda, R) {
  omega_0 = matrix(runif((K), min = -1, max = 1), nrow = (K), ncol = 1)
  omega_i = omega_0
  omega_ii = omega_0
  A_ii = matrix(0, nrow = (K), ncol = K)
  A_i = A_ii
  b_ii = matrix(0, nrow = (K), ncol = 1)
  b_i = b_ii
  
  for (i in 1:n) {
    z = Z[i,]
    y_pred = z %*% omega_ii
    loss = (y[i] - y_pred)^2
    grad = matrix(-2 * c(y[i] - y_pred) * z, nrow = K, ncol = 1)
    omega_i = omega_ii - r * (grad + 2 * lambda * R %*% omega_ii)
    omega_bar = ((i - 1) / i) * omega_ii + (1 / i) * omega_i
    A_i = A_ii + i^2 * omega_bar %*% t(omega_bar)
    b_i = b_ii + i^2 * omega_bar
    V = i^(-2) * (A_i - omega_bar %*% t(b_i) - b_i %*% t(omega_bar) + omega_bar %*% t(omega_bar) * sum((1:i)^2))
    A_ii = A_i
    b_ii = b_i
    omega_ii = omega_i
    #r = c(z%*%grad + 2*lambda*(t(omega_i) %*% R %*% grad)/(2*r*lambda*t(grad) %*% R %*% grad))
  }
  return(list(omega_bar = omega_bar,V = V))
}

r = 0.05
lambda = 0.0000001
start_time <- Sys.time()

sgd_est = estimation(K, r, n, Z, lambda, R)
beta_sgd = t(basis) %*% sgd_est$omega_bar
varbeta_sgd = matrix(0, nrow = tn, ncol = 1)
for (i in 1:tn) {
  varbeta_sgd[i,] = t(basis[,i]) %*% sgd_est$V %*% basis[,i]
}


z_value = 1.96  

sgd_beta_l = beta_sgd - z_value * sqrt(varbeta_sgd)  # 下限
sgd_beta_u = beta_sgd + z_value * sqrt(varbeta_sgd)  # 上限

# 计算覆盖率
coverage = mean(beta >= sgd_beta_l & beta <= sgd_beta_u)
cat("coverage rate:", coverage, "\n")

end_time <- Sys.time()
execution_time <- end_time - start_time
print(execution_time)


#####
bootstrap_estimation = function(K, r, n, Z, lambda, R, B) {
  omega_0 = matrix(runif((K), min = -1, max = 1), nrow = (K), ncol = 1)
  omega_i = omega_0
  omega_ii = omega_0
  V = matrix(0, nrow = K, ncol = K)
  for (i in 1:n) {
    z = Z[i,]
    y_pred = z %*% omega_ii
    loss = (y[i] - y_pred)^2
    grad = matrix(-2 * c(y[i] - y_pred) * z, nrow = K, ncol = 1)
    omega_i = omega_ii - r * (grad)# + 2 * lambda * R %*% omega_ii)
    omega_bar = ((i - 1) / i) * omega_ii + (1 / i) * omega_i
    omega_ii = omega_i
  }
  omega_0 = matrix(runif((K), min = -1, max = 1), nrow = (K), ncol = 1)
  omega_is = omega_0
  omega_iis = omega_0
  for(b in 1:B) {
    W = rexp(K, rate = 1)
    for (j in 1:n) {
      z = Z[i,]
      y_pred = z %*% omega_iis
      loss = (y[i] - y_pred)^2
      grad = matrix(-2 * c(y[i] - y_pred) * z, nrow = K, ncol = 1)
      omega_is = omega_iis - r*W*(grad + 2 * lambda * R %*% omega_iis)
      omega_bars = ((i - 1)/i)*omega_iis + (1/i)*omega_is
      omega_iis = omega_is
    }
    V = V + (1/B)*(omega_bars - omega_bar) %*% t(omega_bars - omega_bar)
  }
  return(list(omega_bar = omega_bar,V = V))
}


r = 0.05
lambda = 0.01
B = 100
start_time <- Sys.time()

sgd_boot = bootstrap_estimation(K, r, n, Z, lambda, R, B)
beta_boot = t(basis) %*% sgd_boot$omega_bar
varbeta_boot = matrix(0, nrow = tn, ncol = 1)
for (i in 1:tn) {
  varbeta_boot[i,] = t(basis[,i]) %*% sgd_boot$V %*% basis[,i]
}
sdbeta_boot = sqrt(varbeta_boot)
z_alpha <- 1.96
boot_beta_l = beta_boot - z_alpha*sdbeta_boot
boot_beta_u = beta_boot + z_alpha*sdbeta_boot

coverage = mean(beta >= boot_beta_l & beta <= boot_beta_u)
cat("coverage rate:", coverage, "\n")

end_time <- Sys.time()
execution_time <- end_time - start_time
print(execution_time)



#####
