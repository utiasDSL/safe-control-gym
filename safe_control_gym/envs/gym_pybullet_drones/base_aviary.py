"""BaseAviary module.

This module contains the superclass of the Quadrotor environment, BaseAviary.
BaseAviary implements most of the integration with PyBullet.

The module also contains enumerations for drone models, PyBullet physics updates
image types captured by PyBullet's camera rendering.

"""
import os
import time
import collections
import pkgutil
from sys import platform
import xml.etree.ElementTree as etxml
from datetime import datetime
from enum import Enum
import numpy as np
import pybullet as p
import pybullet_data

from safe_control_gym.envs.benchmark_env import BenchmarkEnv, Cost

egl = pkgutil.get_loader('eglRenderer')


class DroneModel(Enum):
    """Drone models enumeration class.

    """

    CF2X = "cf2x"  # Bitcraze Craziflie 2.0 in the X configuration.


class Physics(Enum):
    """Physics implementations enumeration class.

    """

    PYB = "pyb"  # Base PyBullet physics update
    DYN = "dyn"  # Update with an explicit model of the dynamics.
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect.
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag.
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash.
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"  # PyBullet physics update with ground effect, drag, and downwash.


class ImageType(Enum):
    """Camera capture image type enumeration class.

    """

    RGB = 0  # Red, green, blue (and alpha).
    DEP = 1  # Depth.
    SEG = 2  # Segmentation by object id.
    BW = 3  # Black and white.


class BaseAviary(BenchmarkEnv):
    """Base class for "drone aviary" Gym environments.

    """

    def __init__(self,
                 seed: int = 1337,
                 info_in_reset: bool = False,
                 episode_len_sec: int = 5,
                 cost: Cost = Cost.QUADRATIC,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 verbose=False,
                 record=False):
        """Initialization of a generic aviary environment.

        Args:
            seed (int, optional): Seed for the random number generator.
            info_in_reset (bool, optional): Whether .reset() returns a dictionary with the
                                            environment's symbolic model.
            episode_len_sec (int, optional): Maximum episode duration in seconds.
            cost: (Cost, optional): Cost function choice used to compute the reward in .step().
            drone_model (DroneModel, optional): The desired drone type (detailed in an .urdf file
                                                in folder `assets`).
            num_drones (int, optional): The desired number of drones in the aviary.
            initial_xyzs (ndarray | None, optional): (NUM_DRONES, 3)-shaped array containing the
                                                     initial XYZ position of the drones.
            initial_rpys (ndarray | None, optional): (NUM_DRONES, 3)-shaped array containing the
                                                     initial orientations of the drones
                                                     (in radians).
            physics (Physics, optional): The desired implementation of PyBullet physics/custom
                                         dynamics.
            freq (int, optional): The frequency (Hz) at which the physics engine steps.
            aggregate_phy_steps (int, optional): The number of physics steps within one call to
                                                 `BaseAviary.step()`.
            gui (bool, optional): Whether to use PyBullet's GUI.
            record (bool, optional): Whether to save a video of the simulation in folder
                                     `files/videos/`.

        """
        # Constants.
        self.GRAVITY_ACC = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1. / self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        # Parameters.
        self.NUM_DRONES = num_drones
        # Options.
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.VERBOSE = verbose
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        # Load the drone properties from the .urdf file.
        self.MASS, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
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
        self.MAX_PWM = self._parse_urdf_parameters(self.URDF)
        print(
            "[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf: \
             \n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f}, \
             \n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f}, \
             \n[INFO] gnd_eff_coeff {:f}, prop_radius {:f}, \
             \n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f}, \
             \n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f} \
             \n[INFO] pwm2rpm_scale {:f}, pwm2rpm_const {:f}, min_pwm {:f}, max_pwm {:f}"
            .format(self.MASS, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2],
                    self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                    self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS,
                    self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1,
                    self.DW_COEFF_2, self.DW_COEFF_3, self.PWM2RPM_SCALE,
                    self.PWM2RPM_CONST, self.MIN_PWM, self.MAX_PWM))
        # Compute constants.
        self.GRAVITY = self.GRAVITY_ACC * self.MASS
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) /
                               (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF)
            / self.MAX_THRUST)
        # Connect to PyBullet.
        self.PYB_CLIENT = -1
        if self.GUI:
            # With debug GUI.
            self.PYB_CLIENT = p.connect(
                p.GUI)  # p.connect(p.GUI, options="--opengl2")
            # for i in [
            #         p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            #         p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            #         p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW
            # ]:
            #     p.configureDebugVisualizer(i, 0, physicsClientId=self.PYB_CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.PYB_CLIENT)
            ret = p.getDebugVisualizerCamera(physicsClientId=self.PYB_CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
        else:
            # Without debug GUI.
            self.PYB_CLIENT = p.connect(p.DIRECT)
            # Uncomment the following line to use EGL Render Plugin #
            # Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux":
            #     p.setAdditionalSearchPath(pybullet_data.getDataPath())
            #     plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            #     print("plugin=", plugin)
        self.RENDER_WIDTH = int(640)
        self.RENDER_HEIGHT = int(480)
        self.FRAME_PER_SEC = 24
        self.CAPTURE_FREQ = int(self.SIM_FREQ / self.FRAME_PER_SEC)
        self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(
            distance=3,
            yaw=-30,
            pitch=-30,
            roll=0,
            cameraTargetPosition=[0, 0, 0],
            upAxisIndex=2,
            physicsClientId=self.PYB_CLIENT)
        self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                    aspect=self.RENDER_WIDTH
                                                    / self.RENDER_HEIGHT,
                                                    nearVal=0.1,
                                                    farVal=1000.0)
        # Set initial poses.
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET)
                                        ]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES, 3):
            self.INIT_XYZS = initial_xyzs
        else:
            raise ValueError(
                "[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)"
            )
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            raise ValueError(
                "[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)"
            )
        self.GROUND_PLANE_Z = -0.05
        # Create action and observation spaces.
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()
        # Metadata.
        # TODO: Check metadata passed to base in gym_control
        self.metadata = {'render.modes': ['human']}
        # Housekeeping.
        self._housekeeping()
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()
        # Start video recording.
        self._start_video_recording()
        # Call BenchmarkEnv constuctor.
        super().__init__(seed=seed, info_in_reset=info_in_reset, episode_len_sec=episode_len_sec, cost=cost)

    def reset(self):
        """Resets the environment.

        Returns:
            ndarray | dict[..]. The initial observation, check the specific implementation of
                                `_get_observation()` in each subclass for its format.

        """
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        # Housekeeping.
        self._housekeeping()
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()
        # Start video recording.
        self._start_video_recording()
        # Return the initial observation.
        return self._get_observation()

    def step(self, action, disturbance_force=None):
        """Advances the environment by one simulation step.

        Args:
            action (ndarray | dict[..]): The input action for one or more drones, translated into
                                         RPMs by the specific implementation of
                                         `_preprocess_action()` in each subclass.
            disturbance_force (ndarray, optional): Disturbance force, applied to all drones.

        Returns:
            ndarray | dict[..]. The step's observation, check the specific implementation of `_get_observation()`
                                in each subclass for its format.
            float | dict[..]. The step's reward value(s), check the specific implementation of `_get_reward()`
                              in each subclass for its format.
            bool | dict[..]. Whether the current epoisode is over, check the specific implementation of `_get_done()`
                             in each subclass for its format.
            dict[..]. Additional information as a dictionary, check the specific implementation of `_get_info()`
                      in each subclass for its format.

        """
        # Save, preprocess, and clip the action to the max. RPM #
        self._save_last_action(action)
        clipped_action = np.reshape(self._preprocess_control(action),
                                    (self.NUM_DRONES, 4))
        # Repeat for as many as the aggregate physics steps.
        for _ in range(self.AGGR_PHY_STEPS):
            # Update and store the drones kinematic info for certain
            # Between aggregate steps for certain types of update.
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [
                    Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG,
                    Physics.PYB_DW, Physics.PYB_GND_DRAG_DW
            ]:
                self._update_and_store_kinematic_information()
            # Step the simulation using the desired physics update.
            for i in range(self.NUM_DRONES):
                # Apply disturbance
                if disturbance_force is not None:
                    pos = self._get_drone_state_vector(i)[:3]
                    p.applyExternalForce(
                        self.DRONE_IDS[i],
                        linkIndex=
                        4,  # Link attached to the quadrotor's center of mass.
                        forceObj=disturbance_force,
                        posObj=pos,
                        flags=p.WORLD_FRAME,
                        physicsClientId=self.PYB_CLIENT)
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._ground_effect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._ground_effect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            # PyBullet computes the new state, unless Physics.DYN.
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.PYB_CLIENT)
            # Save the last applied action (e.g. to compute drag).
            self.last_clipped_action = clipped_action
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()
        # Prepare the return values.
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        # Advance the step counter.
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        """Prints a textual output of the environment.

        Args:
            mode (str, optional): Unused.
            close (bool, optional): Unused.

        """
        if self.first_render_call and not self.GUI:
            print(
                "[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface"
            )
            self.first_render_call = False
        if self.VERBOSE:
            print(
                "\n[INFO] BaseAviary.render() ——— it {:04d}".format(
                    self.step_counter),
                "——— wall-clock time {:.1f}s,".format(time.time()
                                                      - self.RESET_TIME),
                "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(
                    self.step_counter * self.TIMESTEP, self.SIM_FREQ,
                    (self.step_counter * self.TIMESTEP) /
                    (time.time() - self.RESET_TIME)))
            for i in range(self.NUM_DRONES):
                print(
                    "[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                    "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(
                        self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                    "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(
                        self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                    "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(
                        self.rpy[i, 0] * self.RAD2DEG,
                        self.rpy[i, 1] * self.RAD2DEG,
                        self.rpy[i, 2] * self.RAD2DEG),
                    "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".
                    format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i,
                                                                          2]))

    def close(self):
        """Terminates the environment.

        """
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.PYB_CLIENT)
        if self.PYB_CLIENT >= 0:
            p.disconnect(physicsClientId=self.PYB_CLIENT)
        self.PYB_CLIENT = -1

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        # Initialize/reset counters and zero-valued variables.
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_action = -1 * np.ones((self.NUM_DRONES, 4))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        # Initialize the drones kinematic information.
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        # Set PyBullet's parameters.
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.PYB_CLIENT)
        # Load ground plane, drone and obstacles models.
        self.PLANE_ID = p.loadURDF("plane.urdf", [0, 0, self.GROUND_PLANE_Z],
                                   physicsClientId=self.PYB_CLIENT)
        self.DRONE_IDS = np.array([
            p.loadURDF(os.path.dirname(os.path.abspath(__file__)) + "/assets/"
                       + self.URDF,
                       self.INIT_XYZS[i, :],
                       p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                       # flags = p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=self.PYB_CLIENT)
            for i in range(self.NUM_DRONES)
        ])
        # for i in range(self.NUM_DRONES):
        # Show the frame of reference of the drone, note that.
        # It severly slows down the GUI.
        # if self.GUI:
        #     self._show_drone_local_axes(i)

    def _update_and_store_kinematic_information(self):
        """Updates and stores the drones kinematic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range(self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.PYB_CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(
                self.DRONE_IDS[i], physicsClientId=self.PYB_CLIENT)

    def _start_video_recording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.dirname(os.path.abspath(__file__))
                + "/../../files/videos/video-"
                + datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + ".mp4",
                physicsClientId=self.PYB_CLIENT)
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.dirname(os.path.abspath(__file__)) \
                            + "/../../results/quadrotor_videos/video-" \
                            + datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + "/"
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    def _get_drone_state_vector(self, nth_drone):
        """Returns the state vector of the n-th drone.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns:
            ndarray. (20,)-shaped array of floats containing the state vector of the n-th drone.
                     Check the only line in this method and `_update_and_store_kinematic_information()`
                     to understand its format.

        """
        state = np.hstack([
            self.pos[nth_drone, :], self.quat[nth_drone, :],
            self.rpy[nth_drone, :], self.vel[nth_drone, :],
            self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]
        ])
        return state.reshape(20,)

    def _physics(self, rpm, nth_drone):
        """Base PyBullet physics implementation.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.PYB_CLIENT)
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT)

    def _ground_effect(self, rpm, nth_drone):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        # Kin. info of all links (propellers and center of mass)
        link_states = np.array(
            p.getLinkStates(self.DRONE_IDS[nth_drone],
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
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and np.abs(
                self.rpy[nth_drone, 1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.PYB_CLIENT)
        # TODO: a more realistic model accounting for the drone's
        # Attitude and its z-axis velocity in the world frame.

    def _drag(self, rpm, nth_drone):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        # Rotation matrix of the base.
        base_rot = np.array(p.getMatrixFromQuaternion(
            self.quat[nth_drone, :])).reshape(3, 3)
        # Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(
            np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot, drag_factors * np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.PYB_CLIENT)

    def _downwash(self, nth_drone):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(
                np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z))**2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5 * (delta_xy / beta)**2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.PYB_CLIENT)

    def _dynamics(self, rpm, nth_drone):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        # Current state.
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # Compute forces and torques.
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2) * self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL == DroneModel.CF2X:
            x_torque = (forces[0] + forces[1] - forces[2]
                        - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (-forces[0] + forces[1] + forces[2]
                        - forces[3]) * (self.L / np.sqrt(2))
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
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.PYB_CLIENT)
        # Note: the base's velocity only stored and not used #
        p.resetBaseVelocity(
            self.DRONE_IDS[nth_drone],
            vel,
            rpy_rates,  # ang_vel not computed by DYN
            physicsClientId=self.PYB_CLIENT)
        # Store the roll, pitch, yaw rates for the next step #
        self.rpy_rates[nth_drone, :] = rpy_rates

    def _save_last_action(self, action):
        """Stores the most recent action into attribute `self.last_action`.

        The last action can be used to compute aerodynamic effects.
        The method disambiguates between array and dict inputs
        (for single or multi-agent aviaries, respectively).

        Args:
            action (ndarray | dict): (4)-shaped array of ints (or dictionary of arrays) containing
                                     the current RPMs input.

        """
        if isinstance(action, collections.Mapping):
            for k, v in action.items():
                res_v = np.resize(
                    v, (1, 4)
                )  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
                self.last_action[int(k), :] = res_v
        else:
            res_action = np.resize(
                action, (1, 4)
            )  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
            self.last_action = np.reshape(res_action, (self.NUM_DRONES, 4))

    def _show_drone_local_axes(self, nth_drone):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.X_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)
            self.Y_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)
            self.Z_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)

    def _parse_urdf_parameters(self, file_name: str = "cf2x.urdf"):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(
            os.path.dirname(os.path.abspath(__file__)) + "/assets/"
            + file_name).getroot()
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

    def _set_action_space(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    def _preprocess_control(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Args:
            action (ndarray | dict[..]): The input action for one or more drones, to be translated
                                         into RPMs.

        """
        raise NotImplementedError

    def _set_observation_space(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    def _get_observation(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    def _get_reward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    def _get_done(self):
        """Computes the current done value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    def _get_info(self):
        """Computes the current info dict(s).

        Can be re-implemented in a subclass.

        Returns:
            dict. An empy dictionary.

        """
        return {}
