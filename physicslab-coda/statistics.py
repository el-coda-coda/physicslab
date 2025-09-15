from typing import Tuple
import numpy as np

_ERR_XY_LENGTH = "x and y have different length"
_ERR_XERRX_LENGTH = "x and errx have different length"
_ERR_YERRY_LENGTH = "y and erry have different length"
_ERR_DIFF_LENGHT = "Input arrays have different length"
_ERR_FEW_POINTS = "Number of points is not sufficient for this calculation"

class Interpolation:
    """
    <h2>Interpolation</h2>
    <p>This class collects the interpolation methods used in data analysis in the physics laboratory.</p>
    <h3>Methods:</h3>
    <ul>
        <li><code>linear_function(x, a, b)</code> Linear function</li>
        <li><code>erry_var_errx_null(x, y, erry, verbose=False)</code> Linear interpolation with variable error on Y and negligible error on X</li>
        <li><code>linear_interpolation(x, y, errx, erry, threshold=30, tries=10, verbose=False)</code> Linear interpolation with variable error on Y and variable error on X</li>
    </ul>
    <p><strong>Author:</strong> Andrea Codarin</p>
    """

    def __init__(self) -> None:
        # This constructor is intentionally left empty as the class does not require initialization of instance variables.
        pass

    def linear_function(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a+b*x
    
    @staticmethod
    def erry_var_errx_null(x: np.ndarray, y: np.ndarray, erry: np.ndarray, verbose: bool=False) -> Tuple[float, float, float, float]:
        """
        <p><code>erry_var_errx_null(x, y, erry, verbose=True)</code></p>
        <p>This function allows you to calculate the parameters and provide the intermediate details of the linear interpolation 
        of a sample of data. In this case, the error on Y will be variable and the error on X is negligible.</p>
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
    </ul>
    <p><strong>Author:</strong> Andrea Codarin</p>    
    """

    def __init__(self) -> None:
        # This constructor is intentionally left empty as the class does not require initialization of instance variables.
        pass    

    @staticmethod
    def r(x: np.ndarray, y: np.ndarray) -> float:
        """
        <p><code>coeff_corr(x, y)</code></p>
        <p>This function allows you to calculate the sample correlation coefficient of a data set.</p>
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


class StatisticalTests:
    """
    <h2>Statistical Tests</h2>
    <p>This class collects the statistical tests used in data analysis in the physics laboratory.</p>
    <h3>Methods:</h3>
    <ul>
        <li><code>chi2(parameters, y_original, y_calculated, erry)</code> Calculate the chi squared of a data set</li>
        <li><code>reduced_chi2(parameters, y_original, y_calculated, erry)</code> Calculate the reduced chi squared of a data set</li>
        <li><code>post_error(parameters, y_original, y_calculated)</code> Calculate the posterior error of a data set</li>
        <li><code>r_t_student(x, y)</code> Calculate the r t-student value of a cluster of data</li>
        <li><code>t_student(mean, stddev, n, mu)</code> Calculate the t-student value of a cluster of data</li>
        <li><code>f_test(var1, var2, n1, n2)</code> Calculate the f-test value of two clusters of data</li>
        <li><code>double_var_t_student(mean1, mean2, var1, var2, n1, n2)</code> Calculate the t-student value of two clusters of data with different variance</li>
    </ul>
    <p><strong>Author:</strong> Andrea Codarin</p>    
    """

    def __init__(self) -> None:
        # This constructor is intentionally left empty as the class does not require initialization of instance variables.
        pass

    def chi2(self, parameters: int, y_original: np.ndarray, y_calculated: np.ndarray, erry: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>chi_squared(y, y_calculated, erry)</code></p>
        <p>This function allows you to calculate the chi squared of a data set (only for linear function).</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(y_original)!=len(y_calculated)):
            raise ValueError(_ERR_DIFF_LENGHT)    
        elif(len(y_original)!=len(erry)):
            raise ValueError(_ERR_YERRY_LENGTH)
        n = len(y_original)
        if(n<=parameters):
            raise ValueError()
        chi2 = np.sum(((y_original-y_calculated)/erry)**2)
        df = n-parameters
        return float(chi2), df
    
    def reduced_chi2(self, parameters: int, y_original: np.ndarray, y_calculated: np.ndarray, erry: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>reduced_chi_squared(y, y_calculated, erry)</code></p>
        <p>This function allows you to calculate the reduced chi squared of a data set (only for linear function).</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        chi2, df = self.chi2(parameters, y_original, y_calculated, erry)
        rchi2 = chi2/df
        return rchi2, df    
    
    def post_error(self, parameters: int, y_original: np.ndarray, y_calculated: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>post_error(x, y, a, b)</code></p>
        <p>This function allows you to calculate the posterior error of a data set (only for linear function).</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        if(len(y_original)!=len(y_calculated)):
            raise ValueError(_ERR_DIFF_LENGHT)   
        s=sum((y_original-y_calculated)**2)
        df=len(y_calculated)-parameters 
        pe=np.sqrt(s/(len(y_calculated)-parameters))
        return pe, df
    
    def r_t_student(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        <p><code>r_t_student(x, y)</code></p>
        <p>This function allows you to calculate the r t-student value of a cluster of data.</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(_ERR_XY_LENGTH)    
        
        df = len(x)-2
        r = StatisticalVariables.r(x, y)
        t = r*np.sqrt(df)/(np.sqrt(1-r**2))
        return t, df
    
    def t_student(self, mean: float, stddev: float, n: int, mu: float) -> Tuple[float, int]:
        """
        <p><code>t_student(mean, stddev, n, mu)</code></p>
        <p>This function allows you to calculate the t-student value of a cluster of data.</p>
        """
        if(n<2):
            raise ValueError(_ERR_FEW_POINTS)    
        
        df = n-1
        t = (mean-mu)/(stddev/np.sqrt(n))
        return t, df
    
    def f_test(self, var1: float, var2: float, n1: int, n2: int) -> Tuple[float, int, int]:
        """
        <p><code>f_test(var1, var2, n1, n2)</code></p>
        <p>This function allows you to calculate the f-test value of two clusters of data.</p>
        """
        if(n1<2 or n2<2):
            raise ValueError(_ERR_FEW_POINTS)    
        
        if(var1>var2):
            f = var1/var2
            dfn = n1-1
            dfd = n2-1
        else:
            f = var2/var1
            dfn = n2-1
            dfd = n1-1
        return f, dfn, dfd  
    
    def double_var_t_student(self, mean1: float, mean2: float, var1: float, var2: float, n1: int, n2: int) -> Tuple[float, int]:
        """
        <p><code>double_var_t_student(mean1, mean2, var1, var2, n1, n2)</code></p>
        <p>This function allows you to calculate the t-student value of two clusters of data with different variance.</p>
        """
        if(n1<2 or n2<2):
            raise ValueError(_ERR_FEW_POINTS)    
        
        df = ((var1/n1)+(var2/n2))**2/((var1/n1)**2/(n1-1)+(var2/n2)**2/(n2-1))
        t = (mean1-mean2)/np.sqrt(var1/n1+var2/n2)
        return t, int(df)