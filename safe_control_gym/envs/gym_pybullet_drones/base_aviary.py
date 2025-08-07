'''BaseAviary module.

This module contains the superclass of the Quadrotor environment, BaseAviary.
BaseAviary implements most of the integration with PyBullet.

The module also contains enumerations for drone models, PyBullet physics updates
image types captured by PyBullet's camera rendering.
'''

import os
import pkgutil
import time
import xml.etree.ElementTree as etxml
from datetime import datetime
from enum import Enum

import numpy as np
import pybullet as p
import pybullet_data

from safe_control_gym.envs.benchmark_env import BenchmarkEnv

egl = pkgutil.get_loader('eglRenderer')


class DroneModel(str, Enum):
    '''Drone models enumeration class.'''

    CF2X = 'cf2x'  # Bitcraze Craziflie 2.0 in the X configuration.


class Physics(str, Enum):
    '''Physics implementations enumeration class.'''

    PYB = 'pyb'  # Base PyBullet physics update.
    DYN = 'dyn'  # Update with an explicit model of the dynamics.
    PYB_GND = 'pyb_gnd'  # PyBullet physics update with ground effect.
    PYB_DRAG = 'pyb_drag'  # PyBullet physics update with drag.
    PYB_DW = 'pyb_dw'  # PyBullet physics update with downwash.
    PYB_GND_DRAG_DW = 'pyb_gnd_drag_dw'  # PyBullet physics update with ground effect, drag, and downwash.


class ImageType(int, Enum):
    '''Camera capture image type enumeration class.'''

    RGB = 0  # Red, green, blue (and alpha).
    DEP = 1  # Depth.
    SEG = 2  # Segmentation by object id.
    BW = 3  # Black and white.


class BaseAviary(BenchmarkEnv):
    '''Base class for 'drone aviary' Gym environments.'''
    NAME = 'base_aviary'
    URDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 record=False,
                 gui=False,
                 verbose=False,
                 **kwargs):
        '''Initialization of a generic aviary environment.

        Args:
            drone_model (DroneModel, optional): The desired drone type (detailed in an .urdf file
                                                in folder `assets`).
            physics (Physics, optional): The desired implementation of PyBullet physics/custom
                                         dynamics.
            record (bool, optional): Whether to save a video of the simulation in folder
                                     `files/videos/`.
            gui (bool, optional): Whether to use PyBullet's GUI.
            verbose (bool, optional): If to suppress environment print statetments.
        '''
        # Constants.
        self.GRAVITY_ACC = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        # Parameters.
        self.DRONE_MODEL = DroneModel(drone_model)
        self.URDF_PATH = os.path.join(self.URDF_DIR, self.DRONE_MODEL.value + '.urdf')
        self.PHYSICS = Physics(physics)
        self.RECORD = record
        # Load the drone properties from the .urdf file.
        self.MASS, \
            self.L, \
            self.THRUST2WEIGHT_RATIO, \
            self.J, \
            self.J_INV, \
            self.KF, \
            self.KM, \
            self.COLLISION_H, \
            self.COLLISION_R, \
            self.COLLISION_Z_OFFSET, \
            self.MAX_SPEED_KMH, \
            self.GND_EFF_COEFF, \
            self.PROP_RADIUS, \
            self.DRAG_COEFF, \
            self.DW_COEFF_1, \
            self.DW_COEFF_2, \
            self.DW_COEFF_3, \
            self.PWM2RPM_SCALE, \
            self.PWM2RPM_CONST, \
            self.MIN_PWM, \
            self.MAX_PWM = self._parse_urdf_parameters(self.URDF_PATH)
        self.GROUND_PLANE_Z = -0.05
        if verbose:
            print(
                '[INFO] BaseAviary.__init__() loaded parameters from the drone\'s .urdf: \
                \n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f}, \
                \n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f}, \
                \n[INFO] gnd_eff_coeff {:f}, prop_radius {:f}, \
                \n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f}, \
                \n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f} \
                \n[INFO] pwm2rpm_scale {:f}, pwm2rpm_const {:f}, min_pwm {:f}, max_pwm {:f}'
                .format(self.MASS, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2],
                        self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                        self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS,
                        self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1,
                        self.DW_COEFF_2, self.DW_COEFF_3, self.PWM2RPM_SCALE,
                        self.PWM2RPM_CONST, self.MIN_PWM, self.MAX_PWM))
        # Compute constants.
        self.GRAVITY = self.GRAVITY_ACC * self.MASS
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        # BenchmarkEnv constructor.
        super().__init__(gui=gui, verbose=verbose, **kwargs)
        # Connect to PyBullet.
        self.PYB_CLIENT = -1
        if gui:
            # With debug GUI.
            self.PYB_CLIENT = p.connect(p.GUI)  # p.connect(p.GUI, options='--opengl2')
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.PYB_CLIENT)
            ret = p.getDebugVisualizerCamera(physicsClientId=self.PYB_CLIENT)
            if verbose:
                print('viewMatrix', ret[2])
                print('projectionMatrix', ret[3])
        else:
            # Without debug GUI.
            self.PYB_CLIENT = p.connect(p.DIRECT)
            # Uncomment the following line to use EGL Render Plugin #
            # Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == 'linux':
            #     p.setAdditionalSearchPath(pybullet_data.getDataPath())
            #     plugin = p.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
            #     print('plugin=', plugin)
        self.RENDER_WIDTH = int(640)
        self.RENDER_HEIGHT = int(480)
        self.FRAME_PER_SEC = 24
        self.CAPTURE_FREQ = int(self.PYB_FREQ / self.FRAME_PER_SEC)
        self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(
            distance=3,
            yaw=-30,
            pitch=-30,
            roll=0,
            cameraTargetPosition=[0, 0, 0],
            upAxisIndex=2,
            physicsClientId=self.PYB_CLIENT)
        self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                    aspect=self.RENDER_WIDTH / self.RENDER_HEIGHT,
                                                    nearVal=0.1,
                                                    farVal=1000.0)
        # Set default initial poses when loading drone's urdf model.
        # can be overriden later for specific tasks (as sub-classes) in reset()
        self.INIT_XYZ = np.array([0, 0, self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET])
        self.INIT_RPY = np.zeros((3))

    def close(self):
        '''Terminates the environment.'''
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.PYB_CLIENT)
        if self.PYB_CLIENT >= 0:
            p.disconnect(physicsClientId=self.PYB_CLIENT)
        self.PYB_CLIENT = -1

    def _reset_simulation(self):
        '''Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        '''
        # Initialize/reset counters and zero-valued variables.
        self.RESET_TIME = time.time()
        self.first_render_call = True
        self.X_AX = -1
        self.Y_AX = -1
        self.Z_AX = -1
        self.GUI_INPUT_TEXT = -1
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros(4)
        self.gui_input = np.zeros(4)
        # Initialize the drones kinematic information.
        self.pos = np.zeros(3)
        self.quat = np.zeros(4)
        self.rpy = np.zeros(3)
        self.vel = np.zeros(3)
        self.ang_v = np.zeros(3)
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros(3)
        # Set PyBullet's parameters.
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.PYB_CLIENT)
        # Load ground plane, drone and obstacles models.
        self.PLANE_ID = p.loadURDF('plane.urdf', [0, 0, self.GROUND_PLANE_Z],
                                   physicsClientId=self.PYB_CLIENT)
        self.DRONE_ID = p.loadURDF(self.URDF_PATH,
                                   self.INIT_XYZ,
                                   p.getQuaternionFromEuler(self.INIT_RPY),
                                   flags=p.URDF_USE_INERTIA_FROM_FILE,
                                   physicsClientId=self.PYB_CLIENT)
        p.changeDynamics(self.DRONE_ID, -1, linearDamping=0, angularDamping=0)
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()
        # Start video recording.
        self._start_video_recording()

    def _advance_simulation(self, clipped_action, disturbance_force=None):
        '''Advances the environment by one simulation step.

        Args:
            clipped_action (ndarray): The input action for one or more drones,
                                         as RPMs by the specific implementation of
                                         `_preprocess_action()` in each subclass.
            disturbance_force (ndarray, optional): Disturbance force, applied to all drones.
        '''
        clipped_action = np.reshape(clipped_action, 4)
        # Repeat for as many as the aggregate physics steps.
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # Update and store the drones kinematic info for certain
            # Between aggregate steps for certain types of update.
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [
                    Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG,
                    Physics.PYB_DW, Physics.PYB_GND_DRAG_DW
            ]:
                self._update_and_store_kinematic_information()
            # Step the simulation using the desired physics update.
            if self.PHYSICS == Physics.PYB:
                self._physics(clipped_action)
            elif self.PHYSICS == Physics.DYN:
                self._dynamics(clipped_action)
            elif self.PHYSICS == Physics.PYB_GND:
                self._physics(clipped_action)
                self._ground_effect(clipped_action)
            elif self.PHYSICS == Physics.PYB_DRAG:
                self._physics(clipped_action)
                self._drag(clipped_action)
            elif self.PHYSICS == Physics.PYB_DW:
                self._physics(clipped_action)
                self._downwash()
            elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                self._physics(clipped_action)
                self._ground_effect(clipped_action)
                self._drag(clipped_action)
                self._downwash()
            # Apply disturbance
            if disturbance_force is not None:
                pos = self._get_drone_state_vector()[:3]
                p.applyExternalForce(
                    self.DRONE_ID,
                    linkIndex=4,  # Link attached to the quadrotor's center of mass.
                    forceObj=disturbance_force,
                    posObj=pos,
                    flags=p.WORLD_FRAME,
                    physicsClientId=self.PYB_CLIENT)
            # PyBullet computes the new state, unless Physics.DYN.
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.PYB_CLIENT)
            # Save the last applied action (e.g. to compute drag).
            self.last_clipped_action = clipped_action
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()

    def render(self, mode='human', close=False):
        '''Prints a textual output of the environment.

        Args:
            mode (str, optional): Unused.
            close (bool, optional): Unused.
        '''
        if self.first_render_call and not self.GUI:
            print(
                '[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet\'s graphical interface'
            )
            self.first_render_call = False
        if self.VERBOSE:
            print(
                '\n[INFO] BaseAviary.render() ——— it {:04d}'.format(
                    self.pyb_step_counter),
                '——— wall-clock time {:.1f}s,'.format(time.time() - self.RESET_TIME),
                'simulation time {:.1f}s@{:d}Hz ({:.2f}x)'.format(
                    self.pyb_step_counter * self.TIMESTEP, self.SIM_FREQ,
                    (self.pyb_step_counter * self.TIMESTEP) / (time.time() - self.RESET_TIME)))
            print(
                '[INFO] BaseAviary.render() ——— drone 0',
                '——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}'.format(
                    self.pos[0], self.pos[1], self.pos[2]),
                '——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}'.format(
                    self.vel[0], self.vel[1], self.vel[2]),
                '——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}'.format(
                    self.rpy[0] * self.RAD2DEG,
                    self.rpy[1] * self.RAD2DEG,
                    self.rpy[2] * self.RAD2DEG),
                '——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— '.
                format(self.ang_v[0], self.ang_v[1], self.ang_v[2]))

    def _update_and_store_kinematic_information(self):
        '''Updates and stores the drones kinematic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        '''
        self.pos, self.quat = p.getBasePositionAndOrientation(
            self.DRONE_ID, physicsClientId=self.PYB_CLIENT)
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(
            self.DRONE_ID, physicsClientId=self.PYB_CLIENT)

    def _start_video_recording(self):
        '''Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.
        '''
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.join(self.output_dir, 'videos/video-{}.mp4'.format(datetime.now().strftime('%m.%d.%Y_%H.%M.%S'))),
                physicsClientId=self.PYB_CLIENT)
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.output_dir, 'quadrotor_videos/video-{}/'.format(datetime.now().strftime('%m.%d.%Y_%H.%M.%S')))
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    def _get_drone_state_vector(self):
        '''Returns the state vector of the drone.

        Returns:
            ndarray. (20,)-shaped array of floats containing the state vector of the drone.
                     Check the only line in this method and `_update_and_store_kinematic_information()`
                     to understand its format.
        '''
        state = np.hstack([
            self.pos, self.quat,
            self.rpy, self.vel,
            self.ang_v, self.last_clipped_action
        ])
        return state.reshape(20,)

    def _physics(self, rpm):
        '''Base PyBullet physics implementation.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
        '''
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_ID,
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.PYB_CLIENT)
        p.applyExternalTorque(self.DRONE_ID,
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT)

    def _ground_effect(self, rpm):
        '''PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
        '''
        # Kin. info of all links (propellers and center of mass)
        link_states = np.array(
            p.getLinkStates(self.DRONE_ID,
                            linkIndices=[0, 1, 2, 3, 4],
                            computeLinkVelocity=1,
                            computeForwardKinematics=1,
                            physicsClientId=self.PYB_CLIENT))
        # Simple, per-propeller ground effects.
        prop_heights = np.array([
            link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2],
            link_states[3, 0][2]
        ])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF \
            * (self.PROP_RADIUS / (4 * prop_heights))**2
        if np.abs(self.rpy[0]) < np.pi / 2 and np.abs(
                self.rpy[1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_ID,
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.PYB_CLIENT)
        # TODO: a more realistic model accounting for the drone's
        # Attitude and its z-axis velocity in the world frame.

    def _drag(self, rpm):
        '''PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
        '''
        # Rotation matrix of the base.
        base_rot = np.array(p.getMatrixFromQuaternion(
            self.quat)).reshape(3, 3)
        # Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(
            np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot, drag_factors * np.array(self.vel))
        p.applyExternalForce(self.DRONE_ID,
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.PYB_CLIENT)

    def _downwash(self):
        '''PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.
        '''
        delta_z = self.pos[2] - self.pos[2]
        delta_xy = np.linalg.norm(
            np.array(self.pos[0:2]) - np.array(self.pos[0:2]))
        if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
            alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z))**2
            beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
            downwash = [0, 0, -alpha * np.exp(-.5 * (delta_xy / beta)**2)]
            p.applyExternalForce(self.DRONE_ID,
                                 4,
                                 forceObj=downwash,
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.PYB_CLIENT)

    def _dynamics(self, rpm):
        '''Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
        '''
        # Current state.
        pos = self.pos
        quat = self.quat
        rpy = self.rpy
        vel = self.vel
        rpy_rates = self.rpy_rates
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # Compute forces and torques.
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2) * self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL == DroneModel.CF2X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.MASS
        # Update state.
        vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.TIMESTEP * rpy_rates_deriv
        pos = pos + self.TIMESTEP * vel
        rpy = rpy + self.TIMESTEP * rpy_rates
        # Set PyBullet's state.
        p.resetBasePositionAndOrientation(self.DRONE_ID,
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.PYB_CLIENT)
        # Note: the base's velocity only stored and not used #
        p.resetBaseVelocity(
            self.DRONE_ID,
            vel,
            rpy_rates,  # ang_vel not computed by DYN
            physicsClientId=self.PYB_CLIENT)
        # Store the roll, pitch, yaw rates for the next step #
        self.rpy_rates = rpy_rates

    def _show_drone_local_axes(self):
        '''Draws the local frame of the drone in PyBullet's GUI.'''
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self.DRONE_ID,
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.X_AX),
                physicsClientId=self.PYB_CLIENT)
            self.Y_AX = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self.DRONE_ID,
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Y_AX),
                physicsClientId=self.PYB_CLIENT)
            self.Z_AX = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self.DRONE_ID,
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Z_AX),
                physicsClientId=self.PYB_CLIENT)

    def _parse_urdf_parameters(self, file_name):
        '''Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        '''
        URDF_TREE = etxml.parse(file_name).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        PWM2RPM_SCALE = float(URDF_TREE[0].attrib['pwm2rpm_scale'])
        PWM2RPM_CONST = float(URDF_TREE[0].attrib['pwm2rpm_const'])
        MIN_PWM = float(URDF_TREE[0].attrib['pwm_min'])
        MAX_PWM = float(URDF_TREE[0].attrib['pwm_max'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, \
            PWM2RPM_SCALE, PWM2RPM_CONST, MIN_PWM, MAX_PWM
