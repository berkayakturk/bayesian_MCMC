
data {
  int<lower=0> N;        // Number of observations
  vector[N] X;           // Predictor variable X
  vector[N] Y;           // Outcome variable Y
  int<lower=1> Group[N]; // Group membership variable
}


parameters {
  real alpha;            // Intercept
  real beta;             // Slope coefficient for X
  real<lower=0> sigma;   // Error standard deviation
  vector[2] mu_alpha;    // Group-specific intercepts
  }

model {
  // Bad priors
  alpha ~ normal(200, 100);
  beta ~ normal(-5, 10);
  sigma ~ normal(30, 10);
  mu_alpha ~ normal(150, 100);
  
  // Likelihood
  for (i in 1:N) {
    Y[i] ~ normal(alpha + beta * X[i] + mu_alpha[Group[i]], sigma);
  }
}



