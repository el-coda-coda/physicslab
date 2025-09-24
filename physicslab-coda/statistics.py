from typing import Tuple, Callable
import numpy as np

_ERR_XY_LENGTH = "x and y have different length"
_ERR_XERRX_LENGTH = "x and errx have different length"
_ERR_YERRY_LENGTH = "y and erry have different length"
_ERR_DIFF_LENGHT = "Input arrays have different length"
_ERR_FEW_POINTS = "Number of points is not sufficient for this calculation"
_ERR_EMPTY_ARRAY = "Input array is empty"
_ERR_DEG_FREEDOM = "The number of degrees of freedom are wrong"

class Interpolation:
    """
    <h2>Interpolation</h2>
    <p>This class collects the interpolation methods used in data analysis in the physics laboratory.</p>
    <h3>Methods:</h3>
    <ul>
        <li><code>erry_var_errx_null(x, y, erry, verbose=False)</code> Linear interpolation with variable error on Y and negligible error on X</li>
        <li><code>linear_interpolation(x, y, errx, erry, threshold=30, tries=10, verbose=False)</code> Linear interpolation with variable error on Y and variable error on X</li>
    </ul>
    <p><strong>Author:</strong> Andrea Codarin</p>
    """

    def __init__(self) -> None:
        # This constructor is intentionally left empty as the class does not require initialization of instance variables.
        pass
    
    @staticmethod
    def erry_var_errx_null(x: np.ndarray, y: np.ndarray, erry: np.ndarray, verbose: bool=False) -> Tuple[float, float, float, float]:
        """
        <p><code>erry_var_errx_null(x, y, erry, verbose=True)</code></p>
        <p>This function allows you to calculate the parameters and provide the intermediate details of the linear interpolation 
        of a sample of data. In this case, the error on Y will be variable and the error on X is negligible.</p>
        <p><strong>Output:</strong></p>
        <p>a: intercept, s_a: intercept error, b: slope, s_b: slope error</p>
        <p><strong>Author:</strong> Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_XY_LENGTH)    
        elif(len(y)!=len(erry)):
            raise ValueError(_ERR_YERRY_LENGTH)
        
        n = len(x) 
        sum_xx_ss = 0 #x^2/sy^2
        sum_x_ss = 0 #x/sy^2
        sum_xy_ss = 0 #x*y/sy^2
        sum_y_ss = 0 #y/sy^2
        sum_1_ss = 0 #1/sy^2

        #Calculating parameters for the linear interpolation
        for i in range(n):
            sum_xx_ss += (x[i]**2)/(erry[i]**2)
            sum_x_ss += x[i]/(erry[i]**2)
            sum_xy_ss += x[i]*y[i]/(erry[i]**2)
            sum_y_ss += y[i]/(erry[i]**2)
            sum_1_ss += 1/(erry[i]**2)
        delta = sum_1_ss*sum_xx_ss-sum_x_ss**2
        a = (sum_xx_ss*sum_y_ss-sum_x_ss*sum_xy_ss)/delta
        b = (sum_1_ss*sum_xy_ss-sum_x_ss*sum_y_ss)/delta
        s_a = np.sqrt(sum_xx_ss/delta)
        s_b = np.sqrt(sum_1_ss/delta)

        if(verbose):
            print(f'Intermediate calculations:\nSum 1/erry^2: {sum_1_ss}\nSum x/erry^2: {sum_x_ss}\nSum y/erry^2: {sum_y_ss}\nSum x^2/erry^2: {sum_xx_ss}\nSum xy/erry^2: {sum_xy_ss}\nDelta: {delta}')
        return a, b, s_a, s_b
    
    def linear_interpolation(self, x: np.ndarray, y: np.ndarray, errx: np.ndarray, erry: np.ndarray, threshold: float=30, tries: int=10, verbose: bool=False) -> Tuple[float, float, float, float]:
        """
        <p><code>standard_interpolation(x, y, errx, erry)</code></p>
        <p>This function allows you to calculate the parameters and provide the intermediate details of the linear interpolation 
        of a sample of data. In this case, the error on Y will be variable and the error on X is variable, both are not neglegible.
        It choose the better approximation.</p>
        <p><strong>Output:</strong></p>
        <p>a: intercept, s_a: intercept error, b: slope, s_b: slope error</p>
        <p><strong>Author:</strong> Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_XY_LENGTH)    
        elif(len(x)!=len(errx)):
            raise ValueError(_ERR_XERRX_LENGTH)
        elif(len(y)!=len(erry)):
            raise ValueError(_ERR_YERRY_LENGTH)

        n = len(x)
        #Calculating parameters for the linear interpolation
        errymin = min(erry)
        errxmax = max(errx)
        approx_b = (max(y)-min(y))/(max(x)-min(x))

        if(errymin > threshold*errxmax*approx_b):
            print(f'\033[1;31mAPPROXIMATED TO THE CASE OF NEGLIGIBLE ERRORS ON X\033[0m\n Min err on Y > threshold*b*(max err on X): {errymin} > {threshold*errxmax*approx_b}')
            a, b, s_a, s_b = Interpolation.erry_var_errx_null(x, y, erry, verbose=verbose)
        elif(errymin < errxmax*approx_b/threshold):
            print(f'\033[1;31mAPPROXIMATED TO THE CASE OF NEGLIGIBLE ERRORS ON Y: INVERTED X AND Y\033[0m\n(Min err on Y)*thereshold < b*(max err on X): {errymin*threshold} < {errxmax*approx_b}')
            a, b, s_a, s_b = Interpolation.erry_var_errx_null(y, x, errx, verbose=verbose)
        else:
            print(f'\033[1;31mERRORS ON X AND Y ARE NOT NEGLEGIBLE\033[0m: threshold = {threshold}')
            for _ in range(tries):
                temperr = []
                for i in range(n):
                    temperr.append(np.sqrt(erry[i]**2+(errx[i]*b)**2))
                a, b, s_a, s_b = Interpolation.erry_var_errx_null(x, y, np.array(temperr), verbose=verbose)
        
        return a, b, s_a, s_b
    
class StatisticalVariables:
    """
    <h2>Statistical Variables</h2>
    <p>This class collects the statistical variables used in data analysis in the physics laboratory.</p>
    <h3>Methods:</h3>
    <ul>
        <li><code>r(x, y)</code> Calculate the sample correlation coefficient of a data set</li>
        <li><code>mean(x)</code> Calculate the mean of a data set</li>
        <li><code>var(x, sample=True)</code> Calculate the sample variance of a data set
        <li><code>sdev(x, sample=True)</code> Calculate the sample standard deviation of a data set
        <li><code>post_var(f, parameters, x, y)</code> Calculate the posterior variance of a data set
        <li><code>cov(x, y, sample=True)</code> Calculate the sample covariance of a data set
    </ul>
    <p><strong>Author:</strong> Andrea Codarin</p>    
    """

    def __init__(self) -> None:
        # This constructor is intentionally left empty as the class does not require initialization of instance variables.
        pass    

    def r(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        <p><code>coeff_corr(x, y)</code></p>
        <p>This function allows you to calculate the sample correlation coefficient of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>r: sample correlation coefficient</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x*y)
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)

        r = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x**2)*(n*sum_y2-sum_y**2))
        return r
    
    def mean(self, x: np.ndarray) -> float:
        """
        <p><code>mean(x)</code></p>
        <p>This function allows you to calculate the mean of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>mean: data set mean</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x) == 0):
            raise ValueError(_ERR_EMPTY_ARRAY)
        n = len(x)
        mean = np.sum(x)/n
        return float(mean)
    
    def var(self, x: np.ndarray, sample: bool=True) -> float:
        """
        <p><code>variance(x, sample=True)</code></p>
        <p>This function allows you to calculate the sample variance of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>var: data set variance</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        n = len(x)
        mean = self.mean(x)
        correction = 0
        if(sample):
            correction = 1

        variance = np.sum((x-mean)**2)/(n-correction)
        return float(variance)
    
    def sdev(self, x: np.ndarray, sample: bool=True) -> float:
        """
        <p><code>stddev(x, sample=True)</code></p>
        <p>This function allows you to calculate the sample standard deviation of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>sdev: data set standard deviation</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        variance = self.var(x, sample=sample)
        stddev = np.sqrt(variance)
        return float(stddev)
    
    def post_var(self, f: Callable[[np.ndarray], float], parameters: int, x: np.ndarray, y: np.ndarray) -> float:
        """
        <p><code>post_variance(f, parameters, x, y)</code></p>
        <p>This function allows you to calculate the posterior variance of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>pv: data set posterior variance</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_XY_LENGTH)    
        
        n = len(x)
        if(n<=parameters):
            raise ValueError(_ERR_FEW_POINTS)
        
        s = np.sum((y-f(x))**2)
        pv = s/(n-parameters)
        return float(pv)

    def cov(self, x: np.ndarray, y: np.ndarray, sample: bool=True) -> float:
        """
        <p><code>covar(x, y, sample=True)</code></p>
        <p>This function allows you to calculate the sample covariance of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>cov: data set covariance</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_XY_LENGTH)    
        
        n = len(x)
        mean_x = self.mean(x)
        mean_y = self.mean(y)
        correction = 0
        if(sample):
            correction = 1
        covariance = np.sum((x-mean_x)*(y-mean_y))/(n-correction)
        return float(covariance)

class StatisticalTests:
    """
    <h2>Statistical Tests</h2>
    <p>This class collects the statistical tests used in data analysis in the physics laboratory.</p>
    <h3>Methods:</h3>
    <ul>
        <li><code>chi_sqrd(f, parameters, x, y, erry)</code> Calculate the chi squared of a data set</li>
        <li><code>reduced_chi_sqrd(f, parameters, x, y, erry)</code> Calculate the reduced chi squared of a data set</li>
        <li><code>post_error(f, parameters, x, y)</code> Calculate the posterior error of a data set</li>
        <li><code>r_t_student(x, y)</code> Calculate the r-t-student value of a cluster of data</li>
        <li><code>t_student(x, mu)</code> Calculate the t-student value of a cluster of data</li>
        <li><code>alternative_f_test(f_0, f_a, v_0, v_a, x, y)</code> Calculate the f-test value of two clusters of data</li>
        <li><code>t_student_comp(x, y)</code> Calculate the t-student value of two clusters of data with different variance</li>
    </ul>
    <p><strong>Author:</strong> Andrea Codarin</p>    
    """

    def __init__(self) -> None:
        self.stat_variables = StatisticalVariables()
        

    def chi_sqrd(self, f: Callable[[np.ndarray], float], parameters: int, x: np.ndarray, y: np.ndarray, erry: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>chi_sqrd(f(np.ndarray) -> float, parameters, x, y, erry)</code></p>
        <p>This function allows you to calculate the chi squared of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>chi2: data set χ squared, df: degrees of freedom</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_DIFF_LENGHT)    
        if(len(y)!=len(erry)):
            raise ValueError(_ERR_YERRY_LENGTH)
        
        n = len(x)
        
        if(n<=parameters):
            raise ValueError()
        
        chi2 = np.sum(((y-f(x))/erry)**2)
        df = n-parameters
        return float(chi2), df
    
    def reduced_chi_sqrd(self, f: Callable[[np.ndarray], float], parameters: int, x: np.ndarray, y: np.ndarray, erry: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>reduced_chi_sqrd(f(np.ndarray) -> float, parameters, x, y, erry)</code></p>
        <p>This function allows you to calculate the reduced chi squared of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>rchi2: data set χ squared reduced, df: degrees of freedom</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        chi2, df = self.chi_sqrd(f, parameters, x, y, erry)
        rchi2 = chi2/df
        return rchi2, df    
    
    def post_error(self, f: Callable[[np.ndarray], float], parameters: int, x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>post_error(f(np.ndarray) -> float, parameters, x, y)</code></p>
        <p>This function allows you to calculate the posterior error of a data set.</p>
        <p><strong>Output:</strong></p>
        <p>pe: data set posterior error, df: degrees of freedom</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_DIFF_LENGHT)  
         
        s=sum((y-f(x))**2)
        df=len(y)-parameters 
        pe=np.sqrt(s/(len(y)-parameters))
        return pe, df
    
    def r_t_student(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        <p><code>r_t_student(x, y)</code></p>
        <p>This function allows you to calculate the r-t-student value of a cluster of data.</p>
        <p><strong>Output:</strong></p>
        <p>t: data set t-student, df: degrees of freedom</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_XY_LENGTH)    
        
        df = len(x)-2
        r = self.stat_variables.r(x, y)
        t = r*np.sqrt(df)/(np.sqrt(1-r**2))
        return t, df
    
    def t_student(self, x: np.ndarray, mu: float) -> Tuple[float, int]:
        """
        <p><code>t_student(x, mu)</code></p>
        <p>This function allows you to calculate the t-student value of a cluster of data.</p>
        <p><strong>Output:</strong></p>
        <p>t: data set t-student, df: degrees of freedom</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        n = len(x)
        if(n<2):
            raise ValueError(_ERR_FEW_POINTS)
            
        mean = self.stat_variables.mean(x)
        sdev = self.stat_variables.sdev(x)
        df = n-1
        t = (mean-mu)/(sdev/np.sqrt(n))
        return t, df
    
    def alternative_f_test(self, f_0:Callable[[np.ndarray], float], f_a:Callable[[np.ndarray], float], v_0: int, v_a: int, x: np.ndarray, y: np.ndarray) -> Tuple[float, int, int]:
        """
        <p><code>alternative_f_test(f_0(np.ndarray) -> float, f_a(np.ndarray) -> float, v_0, v_a, x, y)</code></p>
        <p>This function allows you to calculate the f-test value of two clusters of data.</p>
        <p><strong>Output:</strong></p>
        <p>f_test: data set f_test, df1: degrees of freedom 1, df2: degrees of freedom 2</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_XY_LENGTH) 
        if(len(x)<2):
            raise ValueError(_ERR_FEW_POINTS)
        if(v_0>=v_a):
            raise ValueError(_ERR_DEG_FREEDOM)
        
        n = len(y)
        var_0 = self.stat_variables.post_var(f_0, v_0, x, y)
        var_a = self.stat_variables.post_var(f_a, v_a, x, y)
        f_test = ((n-v_0)*var_0-(n-v_a)*var_a)/(var_a*(v_a-v_0))
        df1 = v_a-v_0
        df2 = n-v_a
        return f_test, df1, df2
    
    def t_student_comp(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>t_student_comp(x, y)</code></p>
        <p>This function allows you to calculate the t-student value of two clusters of data with different variance.</p>
        <p><strong>Output:</strong></p>
        <p>t: data set t-student, df: degrees of freedom</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(x)==0 or len(y)==0):
            raise ValueError(_ERR_EMPTY_ARRAY)
        
        n_x = len(x)
        n_y = len(y)
        df = n_x+n_y-2

        if(df < 1):
            raise ValueError(_ERR_DEG_FREEDOM)

        mean_x = self.stat_variables.mean(x)
        mean_y = self.stat_variables.mean(y)
        var_x = self.stat_variables.var(x)
        var_y = self.stat_variables.var(y)
        s = ((n_x-1)*var_x+(n_y-1)*var_y)/df
        t = np.abs((mean_x-mean_y))/(s*np.sqrt(1/n_x+1/n_y))
        return t, df