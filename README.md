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

---

### `StatisticalTests`
Class for applying common statistical tests.

- `chi2(parameters, y_original, y_calculated, erry)`  
  Compute **chi-squared**.

- `reduced_chi2(parameters, y_original, y_calculated, erry)`  
  Compute **reduced chi-squared**.

- `post_error(parameters, y_original, y_calculated)`  
  Compute **posterior error**.

- `r_t_student(x, y)`  
  Compute Student’s t-value for correlation.

- `t_student(mean, stddev, n, mu)`  
  Compute Student’s t-value for a sample.

- `f_test(var1, var2, n1, n2)`  
  Perform **F-test** to compare two variances.

- `double_var_t_student(mean1, mean2, var1, var2, n1, n2)`  
  Student’s t-test for two samples with different variances.

---

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
r = StatisticalVariables.r(x, y)
print("r =", r)

# Reduced chi-squared
tests = StatisticalTests()
y_fit = a + b*x
rchi2, df = tests.reduced_chi2(parameters=2, y_original=y, y_calculated=y_fit, erry=erry)
print("Reduced χ² =", rchi2, " with degrees of freedom =", df)
```

---

### Author

**Andrea Codarin**