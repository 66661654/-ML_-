import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
np.random.seed(10086)
n=100
p=1
X=np.random.randn(n,p)
y=2 * X[:,0] + np.random.randn(n)
X=sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()

betas = results.params
stderr = results.bse
t_stats = betas / stderr
f_stat = results.fvalue
f_p_value = results.f_pvalue
print("Regression Coefficients:", betas)
print("Standard Errors:", stderr)
print("t-statistics:", t_stats)
print("F-statistic:", f_stat)
print("F-statistic p-value:", f_p_value)

# Monte Carlo Simulation
n_simulations = 10000
t_stats_sim = np.zeros(n_simulations)
f_stats_sim = np.zeros(n_simulations)
for i in range(n_simulations):
    y_sim = 2 * X[:, 1] + np.random.randn(n)  # Simulate data
    model_sim = sm.OLS(y_sim, X)
    results_sim = model_sim.fit()

    # Record t-statistics and F-statistics for each simulation
    t_stats_sim[i] = results_sim.params[1] / results_sim.bse[1]
    f_stats_sim[i] = results_sim.fvalue

# Calculate p-value for t-statistics
t_p_value_sim = np.mean(np.abs(t_stats_sim) >= np.abs(t_stats[1]))

# Calculate p-value for F-statistics
f_p_value_sim = np.mean(f_stats_sim >= f_stat)

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.hist(t_stats_sim, bins=50, color='dodgerblue', alpha=0.7, edgecolor='black', label='Simulated t-statistics')
plt.axvline(t_stats[1], color='firebrick', linestyle='--', label=f'Observed t-statistic {t_stats[1]:.3f}')
plt.title('t-statistics Distribution', fontsize=14)
plt.xlabel('t-statistic', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.hist(f_stats_sim, bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black', label='Simulated F-statistics')
plt.axvline(f_stat, color='firebrick', linestyle='--', label=f'Observed F-statistic {f_stat:.3f}')
plt.title('F-statistics Distribution', fontsize=14)
plt.xlabel('F-statistic', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print(f"Monte Carlo simulated p-value for t-statistic: {t_p_value_sim:.4f}")
print(f"Monte Carlo simulated p-value for F-statistic: {f_p_value_sim:.4f}")
