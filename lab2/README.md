## Run the Lab2

Paste the lab2 folder within the aer-course-project folder

```bash
python3 main.py --overrides ./lab2.yaml
```

## PID Controller 
```python
#---------Lab2: Design a geomtric controller--------#
#---------Task 1: Compute the desired acceration command--------#
k_pos = 10
K_vel = 5
        
a_fb = -k_pos*(-pos_e) - K_vel*(-vel_e)
a_des = a_fb + target_acc + np.array([0,0,self.grav])
mag_a_des = np.linalg.norm(a_des)

#---------Task 2: Compute the desired thrust command--------#
c_cmd = self.mass * mag_a_des
desired_thrust = c_cmd

#---------Task 3: Compute the desired attitude command--------#
x_c = np.array([np.cos(desired_yaw), np.sin(desired_yaw), 0])
y_c = np.array([-np.sin(desired_yaw), np.cos(desired_yaw), 0])

z_bdes = a_des / mag_a_des
x_bdes = np.cross(y_c, z_bdes)/np.linalg.norm(np.cross(y_c, z_bdes))
y_bdes = np.cross(z_bdes, x_bdes)
desired_euler = Rotation.from_matrix(np.column_stack((x_bdes, y_bdes, z_bdes))).as_euler('xyz')

#---------Task 4: Log the desired and current state--------#
cur_rpy = Rotation.from_quat(cur_quat).as_euler('xyz')
log_entry = np.hstack((
    time.time(),          # Timestamp
    cur_pos, target_pos,       # Position
    cur_vel, target_vel,       # Velocity
    cur_rpy, desired_euler,    # Orientation
    desired_thrust             # Thrust command
))

self.log_buffer.append(log_entry)
```
