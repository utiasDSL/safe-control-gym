#!/usr/bin/env python
import numpy as np
import time
import math

import pybullet as p
from munch import munchify
from scipy.spatial.transform import Rotation as R

from safe_control_gym.controllers.base_controller import BaseController
import pycffirmware as firm

class FirmwareWrapper(BaseController):
    ACTION_DELAY = 0 # max 2
    SENSOR_DELAY = 0 # doesn't affect ideal environment 
    STATE_DELAY = 0 # anything breaks 

    GYRO_LPF_CUTOFF_FREQ = 80
    ACCEL_LPF_CUTOFF_FREQ = 30
    QUAD_FORMATION_X = True
    MOTOR_SET_ENABLE = True
    MIN_THRUST_VAL = 0

    SENSORS_BMI088_G_PER_LSB_CFG = 2 * 24 / 65536
    SENSORS_BMI088_DEG_PER_LSB_CFG = 2 * 2000 / 65536

    RAD_TO_DEG = 180 / math.pi

    def __init__(self, 
                env_func, 
                firmware_freq,
                ctrl_freq,
                PWM2RPM_SCALE = 0.2685,
                PWM2RPM_CONST = 4070.3,
                KF = 3.16e-10,
                MIN_PWM = 20000,
                MAX_PWM = 65535,
                verbose=False,
                **kwargs):
        super().__init__(env_func, **kwargs)
        self.env_func = env_func
        self.firmware_freq = firmware_freq
        self.ctrl_freq = ctrl_freq

        self.PWM2RPM_SCALE = float(PWM2RPM_SCALE)
        self.PWM2RPM_CONST = float(PWM2RPM_CONST)
        self.KF = float(KF)
        self.MIN_PWM = float(MIN_PWM)
        self.MAX_PWM = float(MAX_PWM)
        self.verbose = verbose

        self.env = self.env_func()

    def __repr__(self):
        ret = ""
        ret += f"======= EMULATOR STATUS =======\n"
        ret += f"  \n"
        ret += f"  Tick: {self.tick}\n"
        ret += f"  \n"
        ret += f"  Setpoint\n"
        ret += f"  -------------------------------\n"
        ret += f"  {'Pos':>6}: {round(self.setpoint.position.x, 5):>8}x, {round(self.setpoint.position.y, 5):>8}y, {round(self.setpoint.position.z, 5):>8}z\n"
        # ret += f"  {'Pos':>6}: {int((self.setpoint.position.x+1)*10) * ' ' + ' ':<25}x, {int((self.setpoint.position.y+1)*10) * ' ' + ' ':<25}y, {int((self.setpoint.position.z+1)*10) * ' ' + ' ':<25}z\n"
        ret += f"  {'Vel':>6}: {round(self.setpoint.velocity.x, 5):>8}x, {round(self.setpoint.velocity.y, 5):>8}y, {round(self.setpoint.velocity.z, 5):>8}z\n"
        ret += f"  {'Acc':>6}: {round(self.setpoint.acceleration.x, 5):>8}x, {round(self.setpoint.acceleration.y, 5):>8}y, {round(self.setpoint.acceleration.z, 5):>8}z\n"
        ret += f"  {'Thrust':>6}: {round(self.setpoint.thrust, 5):>8}\n"
        ret += f"  \n"
        ret += f"  Control\n"
        ret += f"  -------------------------------\n"
        ret += f"  {'Roll':>6}: {self.control.roll:>8}\n"
        ret += f"  {'Pitch':>6}: {self.control.pitch:>8}\n"
        ret += f"  {'Yaw':>6}: {self.control.yaw:>8}\n"
        ret += f"  {'Thrust':>6}: {round(self.control.thrust, 5):>8}\n"
        ret += f"  \n"
        ret += f"  State\n"
        ret += f"  -------------------------------\n"
        ret += f"  {'Pos':>6}: {round(self.state.position.x, 5):>8}x, {round(self.state.position.y, 5):>8}y, {round(self.state.position.z, 5):>8}z\n"
        ret += f"  {'Vel':>6}: {round(self.state.velocity.x, 5):>8}x, {round(self.state.velocity.y, 5):>8}y, {round(self.state.velocity.z, 5):>8}z\n"
        ret += f"  {'Acc':>6}: {round(self.state.acc.x, 5):>8}x, {round(self.state.acc.y, 5):>8}y, {round(self.state.acc.z, 5):>8}z\n"
        ret += f"  {'RPY':>6}: {round(self.state.attitude.roll, 5):>8}, {round(self.state.attitude.pitch, 5):>8}, {round(self.state.attitude.yaw, 5):>8}\n"
        ret += f"  \n"
        ret += f"  Action\n"
        ret += f"  -------------------------------\n"
        ret += f"  {'M1':>6}: {round(self.action[0], 3):>8}\n"
        ret += f"  {'M2':>6}: {round(self.action[1], 3):>8}\n"
        ret += f"  {'M3':>6}: {round(self.action[2], 3):>8}\n"
        ret += f"  {'M4':>6}: {round(self.action[3], 3):>8}\n"
        ret += f"  \n"
        ret += f"===============================\n"
        return ret

    #region Controller functions
    def reset(self):
        # Initialize history  
        self.action_history = [[0, 0, 0, 0] for _ in range(self.ACTION_DELAY)]
        self.sensor_history = [[[0, 0, 0], [0, 0, 0]] for _ in range(self.SENSOR_DELAY)]
        self.state_history = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] for _ in range(self.STATE_DELAY)]

        # Initialize gyro lpf 
        self.acclpf = [firm.lpf2pData() for _ in range(3)]
        self.gyrolpf = [firm.lpf2pData() for _ in range(3)]
        for i in range(3):
            firm.lpf2pInit(self.acclpf[i], self.firmware_freq, self.GYRO_LPF_CUTOFF_FREQ)
            firm.lpf2pInit(self.gyrolpf[i], self.firmware_freq, self.ACCEL_LPF_CUTOFF_FREQ)
        self._error = False

        # Initialize state objects 
        self.control = firm.control_t()
        self.setpoint = firm.setpoint_t()
        self.sensorData = firm.sensorData_t()
        self.state = firm.state_t()
        self.tick = 0
        self.pwms = [0, 0, 0, 0]
        self.action = [0, 0, 0, 0]
        self.command_queue = []

        self.sensorData_set = False
        self.state_set = False
        self.full_state_cmd_override = True # When true, high level commander is not called  

        self.prev_vel = np.array([0, 0, 0])
        self.prev_rpy = np.array([0, 0, 0])
        self.prev_time_s = None
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0

        # Initialize PID controller 
        # TODO: add support for other controllers 
        firm.controllerPidInit()
        print('Controller init test:', firm.controllerPidTest())

        # Initilaize high level commander 
        firm.crtpCommanderHighLevelInit()
        firm.crtpCommanderHighLevelTellState(self.state)

        init_obs, init_info = self.env.reset()
        self.ctrl_dt = 1 / self.ctrl_freq
        self.firmware_dt = 1 / self.firmware_freq
        # initialize emulator state objects 
        # if self.env.NUM_DRONES > 1: 
        #     raise NotImplementedError("Firmware controller wrapper does not support multiple drones.")

        self.results_dict = { 'obs': [],
                        'reward': [],
                        'done': [],
                        'info': [],
                        'action': [],
                        }

        return init_obs, init_info


    def close(self):
        self.env.close()


    def step(self, sim_time, action):
        '''
        Step is to be called at a rate of env.CTRL_FREQ. This corresponds to the rate we can send commands through 
        the CF radio, typically below 60Hz. 
        '''
        self.process_command_queue()
        
        count = 0
        while self.tick / self.firmware_freq < sim_time + self.ctrl_dt:
            count += 1
            # Step the environment and print all returned information.
            obs, reward, done, info = self.env.step(action)
            
            # Get state values from pybullet
            # # obs [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy (euler), ang_v] shape is 12 # NOTE ang_v != body rates 
            cur_pos=np.array([obs[0], obs[2], obs[4]]) # global coord, m
            # cur_quat=np.array(p.getQuaternionFromEuler([obs[6], obs[7], obs[8]])) # global coord --- not used 
            cur_vel=np.array([obs[1], obs[3], obs[5]]) # global coord, m/s
            cur_rpy = np.array([obs[6], obs[7], obs[8]]) # body coord?, rad 
            body_rot = R.from_euler('XYZ', cur_rpy).inv()


            # Estimate rates 
            cur_rotation_rates = (cur_rpy - self.prev_rpy) / self.firmware_dt # body coord, rad/s
            # cur_rotation_rates = np.array([0, 0, 0])
            # cur_rotation_rates = cur_rotation_rates[[0, 2, 1]]
            self.prev_rpy = cur_rpy
            cur_acc = (cur_vel - self.prev_vel) / self.firmware_dt / 9.81 + np.array([0, 0, 1]) # global coord
            # print(cur_acc, cur_vel, self.prev_vel)
            self.prev_vel = cur_vel
            

            # Update state 
            state_timestamp = int(self.tick / self.firmware_freq * 1e3)
            if self.STATE_DELAY:
                self._update_state(state_timestamp, *self.state_history[0])#, quat=cur_quat)
                self.state_history = self.state_history[1:] + [[cur_pos, cur_vel, cur_acc, cur_rpy * self.RAD_TO_DEG]]
            else:
                self._update_state(state_timestamp, cur_pos, cur_vel, cur_acc, cur_rpy * self.RAD_TO_DEG)#, quat=cur_quat)


            # Update sensor data 
            sensor_timestamp = int(self.tick / self.firmware_freq * 1e6)
            if self.SENSOR_DELAY:
                self._update_sensorData(sensor_timestamp, *self.sensor_history[0])
                self.sensor_history = self.sensor_history[1:] + [[body_rot.apply(cur_acc), cur_rotation_rates * self.RAD_TO_DEG]]
            else:
                self._update_sensorData(sensor_timestamp, body_rot.apply(cur_acc), cur_rotation_rates * self.RAD_TO_DEG)


            # Update setpoint 
            self._updateSetpoint(self.tick / self.firmware_freq) # setpoint looks right 


            # Step controller 
            self.step_controller()


            # Get action 
            # front left, back left, back right, front right 
            new_action = self.KF * (self.PWM2RPM_SCALE * np.clip(np.array(self.pwms), self.MIN_PWM, self.MAX_PWM) + self.PWM2RPM_CONST)**2
            new_action = new_action[[3, 2, 1, 0]]

            if self.ACTION_DELAY:
                # Delays action commands to mimic real life hardware response delay 
                action = self.action_history[0]
                self.action_history = self.action_history[1:] + [new_action]
            else:
                action = new_action

            if self._error:
                action = np.zeros(4)
                done = True
                print("Drone firmware error. Motors are killed.")

            self.action = action 
        return obs, reward, done, info, action

    def update_initial_state(self, obs):
        self.prev_vel = np.array([obs[1], obs[3], obs[5]])
        self.prev_rpy = np.array([obs[6], obs[7], obs[8]])

    def run(self,
            iterations,
            traj={},
            **kwargs):
        action = np.zeros(4)
        # action = np.array([38727, 38727, 38727, 38727]) # hover to start 
        # action = self.KF * (self.PWM2RPM_SCALE * np.clip(np.array(action), self.MIN_PWM, self.MAX_PWM) + self.PWM2RPM_CONST)**2
        
        s = time.time()
        obs = self.env._get_observation()
        self.update_initial_state(obs)

        posx, posy, posz, yaw = traj[0]
        self.sendFullStateCmd([posx, posy, posz], [0, 0, 0], [0, 0, 0], [0, 0, yaw], [0, 0, 0], 0)

        # Levels the drone to a hover to begin the trajectory 
        counter = 0
        while True: 
            obs, _, _, _, action = self.step(0, action)

            cur_pos = np.array([obs[0], obs[2], obs[4]])

            if np.sum((cur_pos - np.array(traj[0][:-1]))**2)**0.5 < 0.0001:
                counter += 1
            else:
                counter = 0
            if counter > 10:
                break
        
        print("Hover found")
        # time.sleep(100)

        for i in range(iterations):
            # print(self)
            if self._error:
                # Error code set when CF is tumbling 
                break

            if i in traj:
                # print("Traj finished:", firm.crtpCommanderHighLevelIsTrajectoryFinished())
                # self.sendLandCmd(0, 5)
                # self.sendGotoCmd(1, 1, -0.25, math.pi/6, 5, True)
                posx, posy, posz, yaw = traj[i]
                # print("Traj sent", traj[i], i)
                self.sendFullStateCmd([posx, posy, posz], [0, 0, 0], [0, 0, 0], [0, 0, yaw], [0, 0, 0], i)
            
            # time.sleep(0.05)
            raise NotImplementedError("Check code here.")
            obs, reward, done, info, action = self.step(i, action) #TODO: i should be sim_time
            
            # Record iteration results 
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)

            # (Optional) Enforces realtime execution 
            # time.sleep(max(0, (s + sim_time) - time.time())) 

        self.close_results_dict()

        return self.results_dict

    def close_results_dict(self):
        """Cleanup the rtesults dict and munchify it.

        """
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
        self.results_dict['done'] = np.vstack(self.results_dict['done'])
        self.results_dict['info'] = np.vstack(self.results_dict['info'])
        self.results_dict['action'] = np.vstack(self.results_dict['action'])

        self.results_dict = munchify(self.results_dict)

    #endregion

    #region Sensor update
    def _update_sensorData(self, timestamp, acc_vals, gyro_vals, baro_vals=[1013.25, 25]):
        '''
            Axis3f acc;               // Gs
            Axis3f gyro;              // deg/s
            Axis3f mag;               // gauss
            baro_t baro;              // C, Pa
            #ifdef LOG_SEC_IMU
                Axis3f accSec;            // Gs
                Axis3f gyroSec;           // deg/s
            #endif
            uint64_t interruptTimestamp;   // microseconds 
        '''
        ## ONLY USES ACC AND GYRO IN CONTROLLER --- REST IS USED IN STATE ESTIMATION
        self._update_acc(self.sensorData.acc, *acc_vals)
        self._update_gyro(self.sensorData.gyro, *gyro_vals)
        # self._update_gyro(self.sensorData.mag, *mag_vals)
        # self._update_baro(self.sensorData.baro, *baro_vals)

        # firm.sensfusion6UpdateQ(self.sensorData.gyro.x, self.sensorData.gyro.y, self.sensorData.gyro.z,
        #                self.sensorData.acc.x, self.sensorData.acc.y, self.sensorData.acc.z,
        #                0.001)

        self.sensorData.interruptTimestamp = timestamp
        self.sensorData_set = True
    
    def _update_gyro(self, axis3f, x, y, z):
        axis3f.x = firm.lpf2pApply(self.gyrolpf[0], x)
        axis3f.y = firm.lpf2pApply(self.gyrolpf[1], y)
        axis3f.z = firm.lpf2pApply(self.gyrolpf[2], z)

        
    def _update_acc(self, axis3f, x, y, z):
        axis3f.x = firm.lpf2pApply(self.acclpf[0], x)
        axis3f.y = firm.lpf2pApply(self.acclpf[1], y)
        axis3f.z = firm.lpf2pApply(self.acclpf[2], z)


    def _update_baro(self, baro, pressure, temperature):
        '''
        pressure: hPa 
        temp: C
        asl = m 
        '''
        baro.pressure = pressure #* 0.01 Best guess is this is because the sensor encodes raw reading two decimal places and stores as int 
        baro.temperature = temperature
        baro.asl = (((1015.7 / baro.pressure)**0.1902630958 - 1) * (25 + 273.15)) / 0.0065
    #endregion 

    #region State update 
    # def update_state(self):
    #     state = self.env._get_drone_state_vector(0)
    #     pos, quat, rpy, vel, _, _ = np.split(state, [3, 7, 10, 13, 16])

    #     self._update_state(rpy, pos, vel, quat)

    def _update_state(self, timestamp, pos, vel, acc, rpy, quat=None):
        '''
            attitude_t attitude;      // deg (legacy CF2 body coordinate system, where pitch is inverted)
            quaternion_t attitudeQuaternion;
            point_t position;         // m
            velocity_t velocity;      // m/s
            acc_t acc;                // Gs (but acc.z without considering gravity)
        '''
        self._update_attitude_t(self.state.attitude, timestamp, *rpy)
        # if quat is None:
        #     self._update_attitudeQuaternion(self.state.attitudeQuaternion, timestamp, *rpy)
        # else:
        #     self._update_attitudeQuaternion(self.state.attitudeQuaternion, timestamp, *quat)
        self._update_3D_vec(self.state.position, timestamp, *pos)
        self._update_3D_vec(self.state.velocity, timestamp, *vel)
        # TODO: state->acc is used in sitaw freefall detection, and state estimation. Both of which are not included 
        self._update_3D_vec(self.state.acc, timestamp, *acc)
        self.state_set = True

    def _update_3D_vec(self, point, timestamp, x, y, z):
        point.x = x
        point.y = y
        point.z = z
        point.timestamp = timestamp

    def _update_attitudeQuaternion(self, quaternion_t, timestamp, qx, qy, qz, qw=None):
        '''
        if q4 is present, input is taken as a quat. Else, as roll, pitch, and yaw in rad
            uint32_t timestamp;

            union {
                struct {
                    float q0;
                    float q1;
                    float q2;
                    float q3;
                };
                struct {
                    float x;
                    float y;
                    float z;
                    float w;
                };
            };
        '''
        quaternion_t.timestamp = timestamp

        if qw is None: # passed roll, pitch, yaw 
            qx, qy, qz, qw = _get_quaternion_from_euler(qx, qy, qz) 

        quaternion_t.x = qx
        quaternion_t.y = qy
        quaternion_t.z = qz
        quaternion_t.w = qw

    def _update_attitude_t(self, attitude_t, timestamp, roll, pitch, yaw):
        attitude_t.timestamp = timestamp
        attitude_t.roll = roll
        attitude_t.pitch = -pitch # Legacy representation in CF firmware
        attitude_t.yaw = yaw
    #endregion 

    #region Controller 
    def step_controller(self):
        if not (self.sensorData_set):
            print("WARNING: sensorData has not been updated since last controller call.")
        if not (self.state_set):
            print("WARNING: state has not been updated since last controller call.")
        self.sensorData_set = False
        self.state_set = False

        if self.state.acc.z < 0: 
            # Implementation of sitaw.c tumble check 
            # print('WARNING: CrazyFlie is Tumbling. Killing motors to save propellers.')
            # self.pwms = [0, 0, 0, 0]
            self.tick += 1
            self._error = True
            return 

        cur_time = self.tick / self.firmware_freq
        if (cur_time - self.last_att_pid_call > 0.002) and (cur_time - self.last_pos_pid_call > 0.01):
            _tick = 0
            self.last_pos_pid_call = cur_time
            self.last_att_pid_call = cur_time
        elif (cur_time - self.last_att_pid_call > 0.002):
            self.last_att_pid_call = cur_time
            _tick = 2
        else:
            _tick = 1

        firm.controllerPid(
            self.control,
            self.setpoint,
            self.sensorData,
            self.state,
            _tick
        )
        '''
        Tick should increment self.firmware_freq times / s. Position updates run at 100Hz, attitude runs at 500Hz 

        Idea: Set tick from pybullet based on what loops we need to run. 
        = 0: runs both 
        = 1: runs neither 
        = 2: runs attitude 
        '''
        self._powerDistribution(self.control)
        self.tick += 1

    def _updateSetpoint(self, timestep):
        if not self.full_state_cmd_override:
            firm.crtpCommanderHighLevelTellState(self.state)
            firm.crtpCommanderHighLevelUpdateTime(timestep) # Sets commander time variable --- this is time in s from start of flight 
            firm.crtpCommanderHighLevelGetSetpoint(self.setpoint, self.state)

    def process_command_queue(self):
        if len(self.command_queue) > 0:
            command, args = self.command_queue.pop(0)
            getattr(self, command)(*args)

    def sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        '''
        Adds a fullstate command to processing queue

        Arguments:
        pos -- (list) position of the CF (m) 
        vel -- (list) velocity of the CF (m/s)
        acc -- (list) acceleration of the CF (m/s^2)
        yaw -- yaw (rad)
        rpy_rate -- (list) roll, pitch, yaw rates (deg/s)
        timestep -- (s)
        '''
        self.command_queue += [['_sendFullStateCmd', [pos, vel, acc, yaw, rpy_rate, timestep]]]

    def _sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        # print(f"INFO_{self.tick}: Full state command sent.")
        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]
        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]
        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.setpoint.attitudeRate.roll = rpy_rate[0]
        self.setpoint.attitudeRate.pitch = rpy_rate[1]
        self.setpoint.attitudeRate.yaw = rpy_rate[2]

        quat = _get_quaternion_from_euler(0, 0, yaw)
        self.setpoint.attitudeQuaternion.x = quat[0]
        self.setpoint.attitudeQuaternion.y = quat[1]
        self.setpoint.attitudeQuaternion.z = quat[2]
        self.setpoint.attitudeQuaternion.w = quat[3]

        self.setpoint.attitude.yaw = yaw * 180 / math.pi

        # initilize setpoint modes to match cmdFullState 
        self.setpoint.mode.x = firm.modeAbs
        self.setpoint.mode.y = firm.modeAbs
        self.setpoint.mode.z = firm.modeAbs

        self.setpoint.mode.quat = firm.modeAbs
        self.setpoint.mode.roll = firm.modeDisable
        self.setpoint.mode.pitch = firm.modeDisable
        self.setpoint.mode.yaw = firm.modeAbs

        self.setpoint.timestamp = int(timestep*1000) # TODO: This may end up skipping control loops 
        self.full_state_cmd_override = True

    def sendTakeoffCmd(self, height, duration):
        self.command_queue += [['_sendTakeoffCmd', [height, duration]]]
    def _sendTakeoffCmd(self, height, duration):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoff(height, duration)
        self.full_state_cmd_override = False

    def sendTakeoffYawCmd(self, height, duration, yaw):
        self.command_queue += [['_sendTakeoffYawCmd', [height, duration, yaw]]]
    def _sendTakeoffYawCmd(self, height, duration, yaw):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendTakeoffVelCmd(self, height, vel, relative):
        self.command_queue += [['_sendTakeoffVelCmd', [height, vel, relative]]]
    def _sendTakeoffVelCmd(self, height, vel, relative):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendLandCmd(self, height, duration):
        self.command_queue += [['_sendLandCmd', [height, duration]]]
    def _sendLandCmd(self, height, duration):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLand(height, duration)
        self.full_state_cmd_override = False

    def sendLandYawCmd(self, height, duration, yaw):
        self.command_queue += [['_sendLandYawCmd', [height, duration, yaw]]]
    def _sendLandYawCmd(self, height, duration, yaw):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendLandVelCmd(self, height, vel, relative):
        self.command_queue += [['_sendLandVelCmd', [height, vel, relative]]]
    def _sendLandVelCmd(self, height, vel, relative):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendStopCmd(self):
        self.command_queue += [['_sendStopCmd', []]]
    def _sendStopCmd(self):
        print(f"INFO_{self.tick}: Stop command sent.")
        firm.crtpCommanderHighLevelStop()
        self.full_state_cmd_override = False
        
    def sendGotoCmd(self, x, y, z, yaw, duration_s, relative):
        self.command_queue += [['_sendGotoCmd', [x, y, z, yaw, duration_s, relative]]]
    def _sendGotoCmd(self, x, y, z, yaw, duration_s, relative):
        print(f"INFO_{self.tick}: Go to command sent.")
        firm.crtpCommanderHighLevelGoTo(x, y, z, yaw, duration_s, relative)
        self.full_state_cmd_override = False

    BRUSHED = True
    SUPPLY_VOLTAGE = 3 # QUESTION: Is change of battery life worth simulating?
    def _motorsGetPWM(self, thrust):
        if (self.BRUSHED):
            thrust = thrust / 65536 * 60
            volts = -0.0006239 * thrust**2 + 0.088 * thrust
            percentage = min(1, volts / self.SUPPLY_VOLTAGE)
            ratio = percentage * 65535

            return ratio
        else: 
            raise NotImplementedError("Emulator does not support the brushless motor configuration at this time.")

    def _limitThrust(self, val):
        if val > 65535:
            return 65535
        elif val < 0:
            return 0
        return val

    def _powerDistribution(self, control_t):
        motor_pwms = []
        if self.QUAD_FORMATION_X:
            r = control_t.roll / 2
            p = control_t.pitch / 2

            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - r + p + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - r - p - control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + r - p + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + r + p - control_t.yaw))]
        else:
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.pitch + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.roll - control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.pitch + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.roll - control_t.yaw))]
        
        if self.MOTOR_SET_ENABLE:
            self.pwms = motor_pwms
        else:
            self.pwms = np.clip(motor_pwms, self.MIN_THRUST_VAL).tolist()
    #endregion
    
    def _get_time_from_timestep(step, HZ):
        return step / HZ

#region Utils 
def _get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return [qx, qy, qz, qw]

#endregion
