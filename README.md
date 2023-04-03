# LinearInverseModel

## Linear_Inverse_Model.py

```
class LIM(Init: np.ndarray, data: np.ndarray, lag: int, ntimestep: int)
```
This is a statistic model based on lead-lag linear regression.

### Usage

**Init**: Initial condition that you want to start integrating.

**data**: Data which is used to build the model.

**lag**: lag of calculating lead-lag regression

**ntimesteps**: Number of integrating timesteps

```
from Linear_Inverse_Model import LIM

# initial condition
initial_condition = np.array(initial_condition) # initial_condition.shape = (modes,)

# data
data = np.array(data) # data.shape = (modes, time)

model = LIM(initial_condition, data, lag = 5, ntimesteps = 100)
model.build()
model.run()
#model.run(stochastic = False) # turn off the stochastic forcing
```

## test_tools.py

```
auto_corr(PCs: np.ndarray, tau_list: int, plot: bool = False, **plot_additional)
```
This function helps you calculate lead-lag correlation by given lag list (tau_list).

If plot == True, this function returns noneType and show the figure directly.

```
tau_test(PCs: np.ndarray, tau_list: np.ndarray, plot: bool = False, **plot_additional)
```
This function helps you do tau test by given lag list (tau_list).

If plot == True, this function returns noneType and show the figure directly.
