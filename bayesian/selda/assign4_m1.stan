
data {
  int<lower=0> N;        // Number of observations
  vector[N] X;           // Predictor variable X
  vector[N] Y;           // Outcome variable Y
}


parameters {
  real alpha;            // Intercept
  real beta;             // Slope coefficient for X
  real<lower=0> sigma;   // Error standard deviation
}

model {
  // Bad priors
  alpha ~ normal(200, 100);
  beta ~ normal(-5, 10);
  sigma ~ normal(30, 10);
  
  // Likelihood
  Y ~ normal(alpha + beta * X, sigma);
}

