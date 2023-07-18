# bayesian_MCMC
 bayesian_algorithm

 Ecological Fallacy
1. Write a generative simulation involving a predictor variable 𝑋, an outcome variable 𝑌
and a grouping variable 𝐺 ∈ {1, 2}. The simulation should randomly generate a total
number of 𝑁 = 500 observations and match the following constraints:
• Approximately half of the observations are clustered in each group, 𝑁1, 𝑁2 ≈ 250.
• 𝑋𝐺 is normally distributed with group means 𝑋1 ≈ 100 and 𝑋2 ≈ 80 and standard
deviations of 15.
• 𝑌𝐺,𝑖 is normally distributed with a standard deviation of 10 and group means 𝑌 𝐺,𝑖 =
50 + 𝑏𝐺𝑋𝑖.
• The effects of 𝑋 on 𝑌 are 𝑏1 = 0 and 𝑏2 = −.5.

2. Create one or more visualizations of the relation between 𝑋 and 𝑌 illustrating the group
differences in 𝑋, 𝑌 , and 𝑏 and the ecological fallacy that would result from ignoring the
group differences.

3. Analyze the simulated data using a Bayesian linear model predicting 𝑌 from 𝑋. Use
quadratic approximation or MCMC. Use informative priors that don’t match the parameters
from the generative simulation, e.g., place a larger proportion of prior mass on
parameter values you know to be false. (Intentionally setting “bad” priors here illustrates
how the models learn from the data.)
• Model 1: Run a model that doesn’t account for group differences to statistically corroborate
the ecological fallacy shown in the previous visualizations.
• Model 2: Run a model that does account for group differences and captures the parameter
settings from the generative simulation. How many priors are required to estimate the
model?

For Tasks 4 and 5, continue with Model 2.
4. Compute and plot the posterior distribution of differences for the group-specific slope
parameters using 1,000 samples from the joint posterior distribution.

5. Conduct a posterior predictive check.
• For each 𝑋 value simulated in Task 1, predict a 𝑌 value using the posterior distributions
from Model 2.
• Create one or more visualizations of the posterior predictions illustrating the ability of
the model to capture and predict the association of 𝑋 and 𝑌 .
Tipp: To predict values for N people, you should use/draw N samples from the posterior
distribution. Also consider possible differences in group size.


More Groups in Data

Read in the data set WorldData.csv. For the remaining tasks, you will focus on the relations between life expectancy, freedom of choice, and the regional data (region/
region_index/country). Briefly familiarize yourself with the variables using visualizations
and descriptive statistics.
Tipp: All numerical variables but log_gdp are standardized.
The goal is to study the association between freedom of choice and life expectancy. There
will be many possible and rather complex causal routes linking the two measures, involving
the influence of large set of confounds. A proper statistical model would require a lot more thought than can be asked for in this assignment. In the following, we only stratify by region,
assuming that some of the more impactful confounds are embedded therein, and then we hope
for the best.
6. Compute the frequency distribution for the variable region. Briefly discuss potential
problems we could run in when stratifying for region—that is, when computing a separate
regression model for each region.

7. Analyze the data while stratifying for region using quadratic approximation or MCMC.
• Model 3: Estimate a Bayesian Gaussian model for the variable life expectancy.
• Model 4: Estimate a Bayesian linear model predicting life expectancy from freedom of
choice.


8. Below, you find the code and output of a multilevel model for the association between
of freedom of choiceand life expectancy. How and why do the estimates differ compared
to the previous Model 4?
