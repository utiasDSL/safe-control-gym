import numpy as np

from enum import Enum
import rospy
from geometry_msgs.msg import TransformStamped
from crazyswarm.msg import FullState


try:
    import pycffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print("Module 'cffirmware' available:", FIRMWARE_INSTALLED)


import math
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


class Command(Enum):
    NONE = 0 # Args: Empty
    FULLSTATE = 1 # Args: [pos, vel, acc, yaw, rpy_rate, iteration] 
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.cmdFullState
    TAKEOFF = 2 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.takeoff
    LAND = 3 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.land
    STOP = 4 # Args: Empty
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.stop
    GOTO = 5 # Args: [[x, y, z], yaw, duration, relative (bool)]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.goTo


class Controller():
    """Template controller class.

    """

    def __init__(self):
        rospy.init("playback_node")
        self.vicon_sub = rospy.Subscriber("/vicon/cf9/cf9", TransformStamped, self.vicon_callback)
        # self.cmdFullState_sub = rospy.Subscriber("/cf9/cmd_full_state", FullState, self.cmdFullState_callback)

    def vicon_callback(self, data):
        self.child_frame_id = data.child_frame_id
        self.pos = np.array([
            data.transform.translation.x,
            data.transform.translation.y,
            data.transform.translation.z,
        ])
        self.quat = np.array([
            data.transform.rotation.x,
            data.transform.rotation.y,
            data.transform.rotation.z,
            data.transform.rotation.w,
        ])
    
    # def cmdFullState_callback(self, data):
    #     '''
    #     Header header
    #     geometry_msgs/Pose pose
    #     geometry_msgs/Twist twist
    #     geometry_msgs/Vector3 acc

    #     Args: [pos, vel, acc, yaw, rpy_rate, iteration] 

    #     '''
    #     self.cmd_full_state_args = [
    #         [data.pose.position.x, data.pose.position.y, data.pose.position.z],
    #         [data.twist.linear.x, data.twist.linear.y, data.twist.linear.z],
    #         [data.acc.x, data.acc.y, data.acc.z],
    #         euler_from_quaternion(
    #             data.pose.orientation.x
    #         ),

    #     ]

    def cmdFirmware(self,
                    time,
                    obs,
                    vicon_pos=None,
                    est_vel=None,
                    est_acc=None,
                    est_rpy=None,
                    est_rpy_rates=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this function to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call. 

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            vicon_pos (ndarray, optional): Feedback from the vicon tracking system about where your drone marker is (mm).
            est_vel (ndarray, optional): Estimation of drone velocity from Vicon system.
            est_acc (ndarray, optional): Estimation of drone acceleration from Vicon system.
            est_rpy (ndarray, optional): Estimation of drone attitude from Vicon system
            est_rpy_rates (ndarray, optional): Estimation of drone body rates from vicon system.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if not self.use_hardware:
            if self.ctrl is not None:
                raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")
            if not FIRMWARE_INSTALLED:
                raise RuntimeError("[ERROR] Module 'cffirmware' not installed.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if iteration < len(self.ref_x):
            target_pos = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_pos = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_vel = np.zeros(3)
        target_acc = np.zeros(3)
        target_yaw = 0
        target_rpy_rates = np.zeros(3)

        command_type = Command(1)
        args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, time] ##
        #########################

        if iteration < len(self.ref_x):
            target_pos = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_pos = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_vel = np.zeros(3)
        target_acc = np.zeros(3)
        target_yaw = 0
        target_rpy_rates = np.zeros(3)

        command_type = Command(1)
        args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, time]

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args