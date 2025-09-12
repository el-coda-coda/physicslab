from typing import Tuple
import numpy as np

class Interpolation:
    """
    <h2>Interpolation</h2>
    <p>This class collects the interpolation methods used in data analysis in the physics laboratory.</p>
    <h3>Methods:</h3>
    <ul>
        <li><code>poisson_distribution(x, y, return_intermediate=False)</code> Calculate parameters for interpolation of data with Poisson distribution</li>
    </ul>
    <p><strong>Author:</strong> Andrea Codarin</p>
    """

    _ERR_XY_LENGTH = "x and y have different length"
    _ERR_XERRX_LENGTH = "x and errx have different length"
    _ERR_YERRY_LENGTH = "y and erry have different length"

    def __init__(self) -> None:
        # This constructor is intentionally left empty as the class does not require initialization of instance variables.
        pass

    def linear_function(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a+b*x
    
    def coeff_corr(self, x: np.ndarray, y: np.ndarray) -> float:
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
    
    def r_t_student(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        <p><code>r_t_student(x, y)</code></p>
        <p>This function allows you to calculate the r t-student value of a cluster of data.</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(self._ERR_XY_LENGTH)    
        
        df = len(x)-2
        r = self.coeff_corr(x, y)
        t = r*np.sqrt(df)/(np.sqrt(1-r**2))
        return t, df
    
    def post_error(self, parameters: int, y_original: np.ndarray, y_calculated: np.ndarray) -> Tuple[float, int]:
        """
        <p><code>post_error(x, y, a, b)</code></p>
        <p>This function allows you to calculate the posterior error of a data set (only for linear function).</p>
        <p><strong>Author: </strong>Andrea Codarin</p>
        """
        s=sum((y_original-y_calculated)**2)
        df=len(y_calculated)-parameters 
        pe=np.sqrt(s/(len(y_calculated)-parameters))
        return pe, df
    
    @staticmethod
    def erry_var_errx_null(x: np.ndarray, y: np.ndarray, erry: np.ndarray, verbose: bool=False) -> Tuple[float, float, float, float]:
        """
        <p><code>erry_var_errx_null(x, y, erry, verbose=True)</code></p>
        <p>This function allows you to calculate the parameters and provide the intermediate details of the linear interpolation 
        of a sample of data. In this case, the error on Y will be variable and the error on X is negligible.</p>
        """
        if(len(x)!=len(y)):
            raise ValueError(Interpolation._ERR_XY_LENGTH)    
        elif(len(y)!=len(erry)):
            raise ValueError(Interpolation._ERR_YERRY_LENGTH)
        
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
            raise ValueError(self._ERR_XY_LENGTH)    
        elif(len(x)!=len(errx)):
            raise ValueError(self._ERR_XERRX_LENGTH)
        elif(len(y)!=len(erry)):
            raise ValueError(self._ERR_YERRY_LENGTH)

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