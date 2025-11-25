import numpy as np

def map_HenyeyGreenshtein(g:float, theta)->np.ndarray:
    """
    Docstring for fHenyeyGreenshtein

    Function of HenyeyGreenshtein formula
    
    :param g: isotropy contans [-1., 1.]
    :type g: float
    :param theta: input theta array for getting intensivity in this point
    :return: Array of intensivities in each point of theta, with choosen isotropy constant
    :rtype: ndarray
    """
    for theta_i in theta:
        if 1 + g * g + 2 * g * np.cos(theta_i) <= 0:
            print("Inf")
            return 
    return 1/(4 * np.pi) * (1 - g * g) / np.power(1 + g * g - 2 * g * np.cos(theta), 3/2)

print('\n   '.join([f"[DEBUG]",
      f"Check if {map_HenyeyGreenshtein.__name__} give correct out put",
      f" g = 0.9, thetas: [0, np.pi/2, pi - 0.00001, pi]",
      f"Result: {map_HenyeyGreenshtein(0.9, np.array([0, np.pi/2, np.pi - 0.00001, np.pi]))}\n"]))

def alm_HenyeyGreenshtein(g: float, L_max: int) -> np.ndarray:
    """
    Docstring for alm_HenyeyGreenshtein

    We have exact alm formula for HenyeyGreenshtein function: 
    
    >>> alm = g**l / np.sqrt(4*np.pi/(2*l+1)) 

    we normalize it via 1 / sqrt(4*np.pi/(2*l+1)) for simplicity
    
    :param g: isotropy contans [-1., 1.]
    :type g: float
    :param L_max: Orbital moment order
    :type L_max: int
    :return: alm array
    :rtype: ndarray
    """
    return np.array([g**i / np.sqrt(4*np.pi/(2*i+1)) for i in range(L_max+1)])