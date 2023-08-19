import numpy as np
import math
class Scaler:
    """ This class normalizes arrays or values according to a series

    :param s: Array used for normalization
    :type s: array_like
    :param max: max of series s
    :type max: float
    :param min: min of series s
    :type min: float
    """
    def __init__(self, series):
        """ Constructor Method

        :param series: series to use for normlaization
        :type series: array_like
        """
        self.s = series
        self.max = np.max(series)
        self.min = np.min(series)

    def getScaled(self):
        """ Convert s to a normalized array

        :return: normalized array
        :rtype: numpy.ndarray
        """
        return np.subtract(self.s,self.min)/(self.max-self.min)

    def getScaledArray(self, a):
        """ Scales array a to make it comparable with normalized s

        :param a: array to be scaled
        :type a: array_like
        """
        aa = np.asarray(a)
        return np.subtract(aa,self.min)/(self.max-self.min)
    
    def getUnscaled(self):
        """ Returns the original array s

        :return: Original array s
        :rtype: array_like
        """

        return self.s

    def getUnscaledValue(self, v):
        """ Unscales array v to which was scaled using this class

        :param v: value to be corrected
        :type v: float
        :return: true value of v which was scaled using this class
        :rtype: float
        """

        # if v > 1:
        #     print("ERROR out of bounds" ,v)
        u = v*(self.max-self.min) + self.min
        return u

    def getScaledvalue(self, v):
        """ Scale v according to this class's scaled values

        :param v: value to scale
        :type v: float
        :return: true value of v
        :rtype: float
        """
        return (v- self.min)/(self.max-self.min)

class Linear:
    """ Class to store endpoints of a line and handle required requests
    """
    def __init__(self, x1, y1, x2, y2, startIndex=None, endIndex=None):
        """ Constructor Method
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.m = (y1-y2)/(x1-x2)
        self.c = y1 - self.m * x1

    def getY(self,x):
        """ Function to get Y for x for a point in a line

        :param x: x coordinate of a point
        :type x: float
        :return: y coordinate of the point
        :rtype: float
        """
        return self.m*x + self.c

    def isInRange(self,x):
        """ Checks if x is on the line segment

        :param x: x coordinate of point to check
        :type x: float
        :return: Is point on line segment
        :rtype: bool
        """
        if(self.x1 <= x and x <= self.x2):
            return True
        return False

    def getAngle(self, inDegrees=True):
        """ Gives angle of line with x-axis

        :param inDegrees: Specify if angle required is in degress
        :type inDegress: bool
        :return: required angle
        :rtype: float
        """
        tanTheta = self.m
        theta = math.atan(tanTheta)
        if not inDegrees:
            return theta
        else:
            return theta * 180/math.pi

    def getMangnitude(self):
        """ Length of line segment

        :return: Length of Line Segment
        :rtype: float
        """
        return math.sqrt((self.y2 - self.y1)*(self.y2 - self.y1) + (self.x2 - self.x1)*(self.x2 - self.x1))
