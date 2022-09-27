   
# action test:
""" test 
#----------------test action---------------------
# if self.iteration == 0:
#     z = 1
#     yaw = 0.
#     # duration 会影响时间
#     duration = 2
#     command_type = Command(5)  # goTo.
#     args = [[0, 0, z], yaw, duration, True]

# elif self.iteration > 0.5 *self.CTRL_FREQ and self.iteration < 1.5 *self.CTRL_FREQ :
#     command_type = Command(4)  # Stop.
#     args = []
# else:
#     command_type = Command(0)  # None.
#     args = []
#----------------test action---------------------
"""

""" test  go to action
#------------------------------------- 
# go to not very exactly(exist error).  if not random , always this (getting_started.yaml)
#         x                                   y                                z
# even if send  command "only change y " from 60steps (2*CTRL_FREQ)  z will continue increase  ,meybe the problem about the Asynchronous 
# init:  [-0.96609923  0.                  -2.90654614  0.                   0.04960551  0. 0.02840174 -0.05744736  0.07373744  0.          0.          0.        ]
# 0-- [-0.96661885 -0.0279078           -2.90698836 -0.00873109            0.04799388 -0.05964756 0.02340614 -0.04682872  0.07295734 -0.60471722  1.66696169 -0.06040739]
# 15- [-0.98957009  0.0177486           -2.91855744  0.01612694            0.08783306  0.40195587 -0.01363278  0.039587    0.07313598  0.1224719  -0.18560217 -0.0133112 ]
# 30- [-9.74126116e-01 -6.12250160e-03  -2.90933867e+00  2.79855108e-02    5.77164080e-01  1.57968746e+00  1.13573342e-03  2.79637762e-04 7.34927365e-02 -3.14627467e-02  1.04491797e-01 -3.82468499e-05]
# 45- [-0.96690019  0.02592517          -2.9244567  -0.04027908            1.58570938  2.21925943 -0.04590852  0.00406897  0.07356948 -0.20060068 -0.07108402 -0.00401296]
# 60- [-0.97490167 -0.00690929          -2.90244781  0.03042487            2.5512753   1.51622867 0.03011076  0.02357646  0.07466138 -0.6881512  -0.51120803  0.0308384 ]
# 75- [-0.95789148  0.04922068          -2.84188733  0.34668242            3.25083697  1.17400743 -0.23051172 -0.00670337  0.07419264 -1.46396087  0.74862298  0.10745695]
# 90- [-0.96644482  0.03573701          -2.35380695  1.60398413            3.55250003 -0.0314102 -0.30132277  0.04541282  0.06064814 -0.13881515  1.02686106 -0.1751778 ]
# 105-[-0.96541722  0.00741155          -1.35492773  2.16873779            3.27599927 -0.96907974 -0.02264677  0.00403805  0.07491015  0.86021552  1.42216285  0.04174302]
# 120-[-9.69886183e-01  3.45681240e-02  -3.70208598e-01  1.58408557e+00    2.75920080e+00 -8.91757966e-01  1.95222977e-01  6.89915684e-04 7.17125806e-02 -2.54350558e-01  9.00435832e-01  2.10910125e-01]
# 135-[-0.93853971  0.14409335          0.299702    1.2196867              2.35446502 -0.66440774 0.08730616  0.0404132   0.0763123   1.39507016 -0.06357986 -0.14670249]
# if self.iteration == 0:
#     z = 3
#     yaw = 0.
#     # duration 会影响时间
#     duration = 3
#     command_type = Command(5)  # goTo.
#     args = [[0, 0, z], yaw, duration, True]
# elif self.iteration == 2  *self.CTRL_FREQ:
#     y = 3
#     yaw = 0.
#     # duration 会影响时间
#     duration = 3
#     command_type = Command(5)  # goTo.
#     args = [[0, y, 0], yaw, duration, True]
# elif self.iteration == 4  *self.CTRL_FREQ  + 1:
#     x = 1
#     yaw = 0.
#     # duration 会影响时间
#     duration = 3
#     command_type = Command(5)  # goTo.
#     args = [[x, 0, 0], yaw, duration, True]
# else:
#     command_type = Command(0)  # None.
#     args = []
#----------------test action end ---------------------
"""

"""test  go to  and stop action
#----------------test  go to  and stop action---------------------
# result :  stop can break the commmand but have to observe the rule about the intertia.
# if self.iteration == 0:
#     z = 1
#     yaw = 0.
#     # duration 会影响时间
#     duration = 0.2
#     command_type = Command(5)  # goTo.
#     args = [[0, 0, z], yaw, duration, True]
# elif self.iteration == 2  *self.CTRL_FREQ :
#     command_type = Command(4)  # Stop.  
#     args = []
# elif self.iteration == 3  *self.CTRL_FREQ:
#     y = -3
#     yaw = 0.
#     # duration 会影响时间
#     duration = 5
#     command_type = Command(5)  # goTo.
#     args = [[0, y, 0], yaw, duration, True]
# elif self.iteration == 5  *self.CTRL_FREQ  :
#     command_type = Command(4)  # Stop.
#     args = []
# elif self.iteration == 6  *self.CTRL_FREQ  :
#     x = -1
#     yaw = 0.
#     # duration 会影响时间
#     duration = 5
#     command_type = Command(5)  # goTo.
#     args = [[x, 0, 0], yaw, duration, True]
# else:
#     command_type = Command(0)  # None.
#     args = []
#----------------test action end ---------------------
"""

"""test the min time and the max distance about the go to  action
#----------------test   the min time and the max distance about the go to  action---------------------
# result :  
if self.iteration == 0:
    z = 0.15
    # duration 会影响时间
    duration = 0.33
    command_type = Command(5)  # goTo.
    args = [[0, 0, z], 0, duration, True]
else:
    command_type = Command(0)  # None.
    args = []
#----------------test action end ---------------------
"""

""" tst the cmdFullState command  the next command will break the last command
x=obs[0]
y=obs[2]
z=obs[4]
if self.iteration == 0:
    height = 1
    duration = 2
    command_type = Command(2)  # Take-off.
    args = [height, duration]
# 90-[-9.75560513e-01  3.98563938e-02 -2.90579683e+00 -2.46277672e-03 9.46152214e-01 -4.69072173e-02  5.22578314e-03  3.41655474e-02 5.11630895e-02 -2.64937344e-01  1.81993538e+00 -7.07496076e-01]
elif self.iteration == 3 *self.CTRL_FREQ:
    target_pos = np.array([x+0.1,y,z])
    target_vel = np.zeros(3)
    target_acc = np.zeros(3)
    target_yaw = 0.
    target_rpy_rates = np.zeros(3)

    command_type = Command(1)  # cmdFullState.
    args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

# 105-[-9.09596873e-01  2.00324028e-01 -2.90993115e+00 -3.19762174e-02 9.08722645e-01 -6.57070694e-02 -4.07341352e-02 -3.57578141e-02 2.78315544e-03 -4.06931208e-01  1.00299233e+00  2.78734319e-01]
elif self.iteration == 3.5 *self.CTRL_FREQ:
    target_pos = np.array([x, y+0.1, z])
    target_vel = np.zeros(3)
    target_acc = np.zeros(3)
    target_yaw = 0.
    target_rpy_rates = np.zeros(3)

    command_type = Command(1)  # cmdFullState.
    args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

# 120-[-9.09082306e-01 -8.87231485e-02 -2.84472628e+00  2.40919527e-01 8.64559996e-01  1.77237485e-02  3.32461760e-02  2.48394834e-02 2.12280677e-03  9.54833499e-01  1.31716466e+00  1.11926138e-01]
elif self.iteration == 4 *self.CTRL_FREQ:
    target_pos = np.array([x, y, z+0.1])
    target_vel = np.zeros(3)
    target_acc = np.zeros(3)
    target_yaw = 0.
    target_rpy_rates = np.zeros(3)

    command_type = Command(1)  # cmdFullState.
    args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
else :
    command_type = Command(0)  # None.
    args = []
#  135-[-0.90734541  0.07334976 -2.83234816 -0.10398782  0.90615406  0.03580154 -0.0083358  -0.00963472  0.00332174  0.67757902  0.5731973   0.09230706]
"""

"""
# Example action : Handwritten solution for GitHub's getting_stated scenario.
# # default action
# if self.iteration == 0:
#     height = 1
#     duration = 2
#     command_type = Command(2)  # Take-off.
#     args = [height, duration]

# elif self.iteration >= 3*self.CTRL_FREQ and self.iteration < 20*self.CTRL_FREQ:
#     step = min(self.iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
#     target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
#     target_vel = np.zeros(3)
#     target_acc = np.zeros(3)
#     target_yaw = 0.
#     target_rpy_rates = np.zeros(3)

#     command_type = Command(1)  # cmdFullState.
#     args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

# elif self.iteration == 20*self.CTRL_FREQ:
#     command_type = Command(6)  # notify setpoint stop.
#     args = []

# elif self.iteration == 20*self.CTRL_FREQ+1:
#     x = self.ref_x[-1]
#     y = self.ref_y[-1]
#     z = 1.5 
#     yaw = 0.
#     duration = 2.5

#     command_type = Command(5)  # goTo.
#     args = [[x, y, z], yaw, duration, False]

# elif self.iteration == 23*self.CTRL_FREQ:
#     x = self.initial_obs[0]
#     y = self.initial_obs[2]
#     z = 1.5
#     yaw = 0.
#     duration = 6

#     command_type = Command(5)  # goTo.
#     args = [[x, y, z], yaw, duration, False]

# elif self.iteration == 30*self.CTRL_FREQ:
#     height = 0.
#     duration = 3

#     command_type = Command(3)  # Land.
#     args = [height, duration]

# elif self.iteration == 33*self.CTRL_FREQ-1:
#     command_type = Command(-1)  # Terminate command to be sent once trajectory is completed.
#     args = []

# else:
#     command_type = Command(0)  # None.
#     args = []
"""
