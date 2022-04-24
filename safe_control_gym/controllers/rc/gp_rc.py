"""
TODO: https://www.dynsyslab.org/wp-content/papercite-data/pdf/berkenkamp-ecc15.pdf
"""

#TODO Add imports

from safe_control_gym.controllers.base_controller import BaseController

class GPRC(BaseController):
	"""
		Robust controller with Gaussian Process as dynamics residual.
	"""
#TODO: Complete methods as specified in base_controller.py