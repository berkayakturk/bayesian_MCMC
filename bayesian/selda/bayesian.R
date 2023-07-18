# Load required packages
library(dplyr)
library(rstan) 
library(ggplot2)
library(rethinking)
library(readr)

# Task 1: Ecological Fallacy Simulation
# Set seed for reproducibility
set.seed(123)

# Define the total number of observations
N <- 500

# Define the group sizes
N1 <- 250
N2 <- 250

# Generate the grouping variable
G <- sample(c(1, 2), size = N, replace = TRUE)

# Generate the predictor variable X
X1 <- rnorm(N1, mean = 100, sd = 15)
X2 <- rnorm(N2, mean = 80, sd = 15)
X <- c(X1, X2)

# Generate the outcome variable Y
Y1 <- rnorm(N1, mean = 50, sd = 10)  # Generate 'N1' random numbers from a normal distribution with mean 50 and standard deviation 10, assign to 'Y1'
Y2 <- rnorm(N2, mean = 50 - 0.5 * X2, sd = 10)  # Generate 'N2' random numbers from a normal distribution with mean (50 - 0.5 * X2) and standard deviation 10, assign to 'Y2'
Y <- c(Y1, Y2)  # Combine 'Y1' and 'Y2' into one vector 'Y'


# Combine the variables into a data frame
data <- data.frame(Group = G, X = X, Y = Y)
# task complete: Simulation generated based on constraints
head(data)

# Task 2: Visualization of the relationship between X and Y

# Create a scatter plot with group differentiation
plot <- ggplot(data, aes(x = X, y = Y, color = as.factor(Group))) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "X", y = "Y", color = "Group") +
  theme_minimal()

# Display the plot
plot

# Task complete: Scatter plot generated illustrating group differences and potential ecological fallacy.
# Regression curves were drawn and the effects of both groups were positive and the same.

# Task 3: Analyze the simulated data using a Bayesian linear model predicting 𝑌 from 𝑋

# To analyze the simulated data using Bayesian linear models, the Stan programming language can be employed. 
# Stan provides powerful tools for Bayesian inference, including both MCMC and variational inference methods. 
# In this case, I will use MCMC to estimate the models.

# Model 1: Bayesian linear regression without group differences

# Specify the data
data_model1 <- list(
  N = 500,      # Number of observations
  X = data$X,    # Predictor variable X 
  Y = data$Y     # Outcome variable Y 
)

# Specify and fit the Stan model code for Model1
m1stan <- stan("C:/Users/OneDrive/Belgeler/bayesian/assign4_m1.stan", 
               data = data_model1,  # Data for the model
               chains = 4,  # Number of Markov chains
               cores = 4,  # Number of CPU cores to use
               iter = 2000  # Number of iterations per chain
               )

m1stan

# Print the summary of the posterior distribution for Model1
print(summary(m1stan))

stan_dens(m1stan)
# Plot posterior distributions for Model1
plot(m1stan)
traceplot(m1stan)


# Specify the data for Model2
data_model2 <- list(
  N = 500,      # Number of observations
  X = data$X,    # Predictor variable X 
  Y = data$Y,    # Outcome variable Y 
  Group = data$Group    # Group membership variable 
)

# Specify and fit the Stan model code for Model2
m2stan <- stan("C:/Users/OneDrive/Belgeler/bayesian/assign4_m2.stan", 
               data=data_model2, 
               chains = 4, 
               cores = 4, 
               iter = 4000)

m2stan

# Print the summary of the posterior distribution for Model1
print(summary(m2stan))

stan_dens(m2stan)
# Plot posterior distributions for Model1
plot(m2stan)
traceplot(m2stan)


# For both models, intentionally set "bad" priors that do not match the parameters from the generative 
# simulation. This allows to observe how the models learn from the data and update their beliefs.

# The number of priors required to estimate the model2 is as follows:

# Model 2: Requires 4 priors (alpha, beta, sigma, and two group-specific intercepts mu_alpha).



# Task 4: Posterior Distribution of Group-Specific Slope Parameters
# Compute and plot the posterior distribution of differences for the group-specific slope parameters

# Specify the number of samples
num_samples <- 1000

# Extract samples from the joint posterior distribution
posterior_samples_model2 <- as.matrix(m2stan)

# Compute differences for the group-specific slope parameters
group_diffs <- posterior_samples_model2[, "mu_alpha[2]"] - posterior_samples_model2[, "mu_alpha[1]"]

# Plot the posterior distribution of differences
plot_task4 <- hist(group_diffs, breaks = 30, main = "Posterior Distribution of Group Differences",
                   xlab = "Group-Specific Slope Differences", ylab = "Frequency")


# Task 5: Posterior Predictive Check
# Conduct a posterior predictive check

# Draw samples from the posterior distribution
posterior_samples <- as.matrix(m2stan, pars = c("alpha", "beta", "sigma", "mu_alpha"))

# Predict Y values for each X value using the posterior distributions
X_values <- data$X
N_values <- rep(N, length(X_values))
group_sizes <- table(data$Group)

predicted_Y <- array(NA, dim = c(length(X_values), N))
for (i in 1:length(X_values)) {  # Loop over each element in 'X_values' array
  # Compute 'N_samples' as a weighted sum of 'alpha' and 'mu_alpha' from posterior samples
  N_samples <- group_sizes[1] * posterior_samples$alpha + group_sizes[2] * (posterior_samples$alpha + posterior_samples$mu_alpha[, 2])
  
  # Generate 'Y_samples' from a normal distribution with mean as a function of 'beta' and 'N_samples', and standard deviation 'sigma'
  Y_samples <- rnorm(N, mean = posterior_samples$beta * X_values[i] + N_samples, sd = posterior_samples$sigma)
  
  # Store 'Y_samples' in the i-th row of 'predicted_Y' matrix
  predicted_Y[i, ] <- Y_samples
}



# Create visualizations of the posterior predictions
plot_task5 <- ggplot() +
  geom_point(data = data, aes(x = X, y = Y, color = as.factor(Group)), alpha = 0.3) +
  geom_line(data = data.frame(X = rep(X_values, each = N), Y = predicted_Y), aes(x = X, y = Y), color = "blue", alpha = 0.3) +
  labs(x = "X", y = "Y", color = "Group") +
  theme_minimal()


#More Groups in Data

#Task 6 Compute the frequency distribution for the variable region

# Read the data
worlddata <- read_csv("C:/Users/OneDrive/Belgeler/bayesian/WorldData.csv")

# Compute frequency distribution for the variable region
region_freq <- table(worlddata$region)

# Display the frequency distribution
print(region_freq)

# One potential problem when stratifying for region is the presence of imbalanced sample 
# sizes across regions. If some regions have a significantly larger number of observations 
# compared to others, it can lead to biased results or insufficient representation of smaller regions 
# in the analysis. Another potential problem is the heterogeneity within regions. Regions may contain 
# diverse countries with different socio-economic characteristics, which can introduce confounding 
# factors and complicate the interpretation of the results. Additionally, if there are missing or 
# incomplete data for certain regions, it can further affect the analysis and interpretation.
# Ayrıca regresyon modeli kurulurken gerekli varsayımların sağlanması beklenmelidir.
# There should not be a problem of multicollinearity between regions, that is, regions 
# should not be too much related to each other and should provide a normal distribution. 
# The variance of the errors should be constant at the levels of all regions.

# Task 7: Analyze the data while stratifying for region using quadratic approximation or MCMC.

# Standardize numerical variables (except log_gdp)
worlddata$life_expectancy <- scale(worlddata$life_expectancy)
worlddata$freedom_of_choice <- scale(worlddata$freedom_of_choice)

# Subset the data for Model 3 and Model 4
subset <- worlddata[, c("region_index", "life_expectancy")]
subset$model4 <- worlddata$freedom_of_choice



# Model 3: Bayesian Gaussian model for life expectancy
model3 <- ulam(
  alist(
    life_expectancy ~ dnorm(mu, sigma),  # Dependent variable follows a normal distribution
    mu <- a[region_index],  # Mean 'mu' is determined by region-specific parameter 'a'
    a[region_index] ~ dnorm(a_bar[region_index], a_tau[region_index]),  # 'a' follows a normal distribution, parameters differ by region
    a_bar[region_index] ~ dnorm(0, 2),  # Hyperprior for mean of 'a' is normally distributed
    a_tau[region_index] ~ dexp(1),  # Hyperprior for standard deviation of 'a' follows an exponential distribution
    sigma ~ dexp(1)  # Prior for the standard deviation of the dependent variable follows an exponential distribution
  ),
  data = subset,  # Model is fit using the 'subset' dataset
  chains = 8,  # Model uses 8 Markov chains
  cores = 8,  # Model uses 8 cores for parallel computation
  iter = 4000  # Each chain runs for 4000 iterations
)

# Model 4: Bayesian linear model for life expectancy from freedom of choice
model4 <- ulam(
  alist(
    life_expectancy ~ dnorm(mu, sigma),  # Dependent variable follows a normal distribution
    mu <- a[region_index] + b[region_index] * model4,  # Mean 'mu' depends on region-specific parameters 'a' and 'b'
    a[region_index] ~ dnorm(a_bar, a_tau),  # 'a' follows a normal distribution, parameters do not differ by region
    b[region_index] ~ dnorm(b_bar, b_tau),  # 'b' follows a normal distribution, parameters do not differ by region
    a_bar ~ dnorm(0, 2),  # Hyperprior for mean of 'a' is normally distributed
    a_tau ~ dexp(1),  # Hyperprior for standard deviation of 'a' follows an exponential distribution
    b_bar ~ dnorm(0, 2),  # Hyperprior for mean of 'b' is normally distributed
    b_tau ~ dexp(1),  # Hyperprior for standard deviation of 'b' follows an exponential distribution
    sigma ~ dexp(1)  # Prior for the standard deviation of the dependent variable follows an exponential distribution
  ),
  data = subset,  # Model is fit using the 'subset' dataset
  chains = 8,  # Model uses 8 Markov chains
  cores = 8,  # Model uses 8 cores for parallel computation
  iter = 4000  # Each chain runs for 4000 iterations
)


# Task 8

# In Model 4, a simple Bayesian linear model was used to predict life expectancy from freedom of choice. 
# However, in Model 5, a more complex multilevel model is employed, allowing for region-specific variation 
# in the association. The estimates in Model 5 (m5) differ from Model 4 due to the inclusion of 
# region-specific random effects (a[region_index] and b[region_index]). These random effects capture the 
# variability in the intercept (a) and slope (b) of the association between freedom of choice and life 
# expectancy across different regions. Compared to Model 4, Model 5 provides estimates for both fixed 
# effects (a_bar, a_tau, b_bar, b_tau) and random effects (a[region_index], b[region_index]) for each 
# region. The estimates of a[region_index] and b[region_index] reflect the deviations from the average 
# intercept (a_bar) and slope (b_bar) values, respectively, specific to each region. The estimates of 
# a[region_index] and b[region_index] in Model 5 indicate how the intercept and slope of the association 
# between freedom of choice and life expectancy vary across different regions. These estimates take into 
# account the region-specific characteristics and provide a more nuanced understanding of the relationship.