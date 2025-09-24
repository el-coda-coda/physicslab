# Physicslab-coda

A Python library for **data analysis in physics laboratories**, including:
- linear interpolation with uncertainties
- statistical variables
- common statistical tests (χ², t-Student, F-test, etc.)

The goal is to provide ready-to-use tools for students and researchers working with experimental data.

---

## Statistics

### `Interpolation`
Class for linear interpolation with experimental uncertainties.

- `linear_function(x, a, b)`  
  Simple linear function.

- `erry_var_errx_null(x, y, erry, verbose=False)`  
  Linear interpolation with variable error on **Y** and negligible error on **X**.

- `linear_interpolation(x, y, errx, erry, threshold=30, tries=10, verbose=False)`  
  Linear interpolation with variable errors on both **X** and **Y**.  
  Automatically chooses the best approximation depending on error magnitudes.

---

### `StatisticalVariables`
Class for calculating basic statistical variables.

- `r(x, y)`  
  Computes the sample correlation coefficient.  
  **Output:** `r` (sample correlation coefficient)

- `mean(x)`  
  Computes the mean of a data set.  
  **Output:** `mean` (data set mean)

- `var(x, sample=True)`  
  Computes the sample variance of a data set.  
  **Output:** `var` (data set variance)

- `sdev(x, sample=True)`  
  Computes the sample standard deviation of a data set.  
  **Output:** `sdev` (data set standard deviation)

- `post_var(f, parameters, x, y)`  
  Computes the posterior variance of a data set.  
  **Output:** `pv` (data set posterior variance)

- `cov(x, y, sample=True)`  
  Computes the sample covariance of a data set.  
  **Output:** `cov` (data set covariance)

---

### `StatisticalTests`
Class for applying common statistical tests.

- `chi_sqrd(f, parameters, x, y, erry)`  
  Compute **chi-squared**.  
  **Output:** `chi2` (chi-squared value), `df` (degrees of freedom)

- `reduced_chi_sqrd(f, parameters, x, y, erry)`  
  Compute **reduced chi-squared**.  
  **Output:** `reduced_chi2` (reduced chi-squared value), `df` (degrees of freedom)

- `post_error(f, parameters, x, y)`  
  Compute **posterior error**.  
  **Output:** `pe` (posterior error), `df` (degrees of freedom)

- `r_t_student(x, y)`  
  Compute Student’s t-value for correlation.  
  **Output:** `t` (t-value), `df` (degrees of freedom)

- `t_student(x, mu)`  
  Compute Student’s t-value for a sample mean.  
  **Output:** `t` (t-value), `df` (degrees of freedom)

- `alternative_f_test(f_0, f_a, v_0, v_a, x, y)`  
  Perform **F-test** to compare two models.  
  **Output:** `f_test` (F-test value), `df1`, `df2` (degrees of freedom)

- `t_student_comp(x, y)`  
  Student’s t-test for two samples with different variances.  
  **Output:** `t` (t-value), `df` (degrees of freedom)

---

## Installation

```bash
git clone https://github.com/yourusername/physics-lab-data-analysis.git
cd physics-lab-data-analysis
pip install -r requirements.txt
```

---

## Usage Example

```python
import numpy as np
from physicslab-coda import Interpolation, StatisticalVariables, StatisticalTests

# Example data
x = np.array([1, 2, 3, 4])
y = np.array([2.1, 4.2, 5.9, 8.2])
erry = np.array([0.1, 0.2, 0.1, 0.3])
errx = np.array([0.05, 0.05, 0.05, 0.05])

# Linear interpolation
interp = Interpolation()
a, b, s_a, s_b = interp.linear_interpolation(x, y, errx, erry, verbose=True)
print("a =", a, " b =", b, " σa =", s_a, " σb =", s_b)

# Correlation coefficient
stat_vars = StatisticalVariables()
r = stat_vars.r(x, y)
print("r =", r)

# Reduced chi-squared
tests = StatisticalTests()
y_fit = a + b*x
rchi2, df = tests.reduced_chi_sqrd(lambda x: a + b*x, parameters=2, x=x, y=y, erry=erry)
print("Reduced χ² =", rchi2, " with degrees of freedom =", df)
```

---

### Author

**Andrea Codarin**