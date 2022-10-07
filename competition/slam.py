import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time

class SLAM():
    def __init__(self,gs=0.05) -> None:
        # slam建图范围
        self.X_MIN = -3.5
        self.X_MAX = 3.5
        self.Y_MIN = -3.5
        self.Y_MAX = 3.5
        self.Z_MIN = 0.0
        self.Z_MAX = 2.0
        # 栅格/元胞大小 grid shape = 5cm * 5cm * 5cm
        self.gs = gs
        # 障碍物初始位置
        self.obstacles = [
        [1.5, -2.5, 0, 0, 0, 0],
        [0.5, -1, 0, 0, 0, 0],
        [1.5, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0]
        ]
        # 障碍物高度
        self.obstacle_height = 1.05
        # 门的初始位置，用不上，info里有更精确的
        self.gates = [
        [0.5, -2.5, 0, 0, 0, -1.57, 0],
        [2, -1.5, 0, 0, 0, 0, 1],
        [0, 0.2, 0, 0, 0, 1.57, 1],
        [-0.5, 1.5, 0, 0, 0, 0, 0]
        ]
        # 门的尺寸
        self.gate_tall = 1.0
        self.gate_low = 0.525
        self.edge = 0.45/2

    def reset_occ_map(self):
        # reset the occupation map
        self.occ_map = np.zeros((
            round((self.Z_MAX -self.Z_MIN)/self.gs),
            round((self.X_MAX -self.X_MIN)/self.gs),
            round((self.Y_MAX -self.Y_MIN)/self.gs),
        ),dtype=int)

        # # 不用初始化门
        # for gate in self.gates:
        #     self._set_gate_occupied(gate[:-1],gate[-1])
        
        # 初始化障碍物
        for obstacle in self.obstacles:
            self._set_obstacle_occupied(obstacle)
        
        self.current_id = -1

    def update_occ_map(self,info):
        # env reset 时，info的信息里没有这项
        # 另外，视野里没有时，无需更新
        if not info.get('current_target_gate_in_range'):
            return
        else:
            if info.get('current_target_gate_id') > self.current_id:
                self.current_id = info.get('current_target_gate_id')
                self._set_gate_occupied(info.get('current_target_gate_pos'),info.get('current_target_gate_type'))
                self.plot_occ_map(str(self.current_id+1))
                print(self.current_id)

    def _set_pos_occupied(self,x,y,z):
        # 设置绝对坐标(x,y,z)处为障碍占据
        assert self.X_MIN <= x <= self.X_MAX  \
                and self.Y_MIN <= y <= self.Y_MAX \
                and self.Z_MIN <= z <= self.Z_MAX
        x_idx = round((x-self.X_MIN)/self.gs)
        y_idx = round((y-self.Y_MIN)/self.gs)
        z_idx = round((z-self.Z_MIN)/self.gs)
        self.occ_map[z_idx][x_idx][y_idx] = 1
    
    def _set_obstacle_occupied(self,obstacle):
        # 设置(x,y)处的柱子为障碍占据
        x,y,_,_,_,_, = obstacle
        z = 0
        while z <= self.obstacle_height:
            self._set_pos_occupied(x,y,z)
            z += self.gs
    
    # 这个地方不对，并不是只有0度和90度，门的角度yaw也会随机
    # 目前只是简单的按照绝对值做了处理，
    def _set_gate_occupied(self,gate,gate_type=0):
        # 设置(x,y)处类型为type的门为障碍占据
        x,y,_,_,_,direct = gate
        height = self.gate_tall if gate_type == 0 else self.gate_low
        z = 0
        # 基座
        while z <= height - self.edge-0.05:
            self._set_pos_occupied(x,y,z)
            z += self.gs
        # 下门框
        while z <= height - self.edge+0.05:
            if abs(direct) <= 0.15:
                for tmp_x in np.arange(x-self.edge,x+self.edge+self.gs,self.gs/2):
                    self._set_pos_occupied(tmp_x,y,z)
            else:
                for tmp_y in np.arange(y-self.edge,y+self.edge+self.gs,self.gs/2):
                    self._set_pos_occupied(x,tmp_y,z)
            z += self.gs
        # 左右门框
        while z <= height + self.edge-0.05:
            if abs(direct) <= 0.15:
                self._set_pos_occupied(x-self.edge,y,z)
                self._set_pos_occupied(x+self.edge,y,z)
            else:
                self._set_pos_occupied(x,y-self.edge,z)
                self._set_pos_occupied(x,y+self.edge,z)
            z += self.gs
        # 上门框
        while z <= height + self.edge+0.05:
            if abs(direct) <= 0.15:
                for tmp_x in np.arange(x-self.edge,x+self.edge+self.gs,self.gs/2):
                    self._set_pos_occupied(tmp_x,y,z)
            else:
                for tmp_y in np.arange(y-self.edge,y+self.edge+self.gs,self.gs/2):
                    self._set_pos_occupied(x,tmp_y,z)
            z += self.gs
        
    # 绘制
    def plot_occ_map(self,name='test'):
        xs = []
        ys = []
        zs = []
        C,W,H= self.occ_map.shape
        for z in range(C):
            for x in range(W):
                for y in range(H):
                    if self.occ_map[z][x][y] == 1:
                        xs.append(x*self.gs+self.X_MIN)
                        ys.append(y*self.gs+self.Y_MIN)
                        zs.append(z*self.gs+self.Z_MIN)
            
        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        #  将数据点分成三部分画，在颜色上有区分度
        ax.scatter(xs, ys, zs, c='black')  # 绘制数据点
        plt.xlim((self.X_MIN, self.X_MAX))
        plt.ylim((self.Y_MIN, self.Y_MAX))
        plt.savefig(name+'.png')

    # 根据所在高度，生成2d 障碍物图像
    # TODO 出现问题，在刚开始训练的时候会超过约定的边界，
    def generate_obs_img(self,obs,name='test',save=False):
        x,y,z=obs[0],obs[2],obs[4]
        x_idx = round((x-self.X_MIN)/self.gs)
        y_idx = round((y-self.Y_MIN)/self.gs)
        z_idx = round((z-self.Z_MIN)/self.gs)
        if save:
            occ_map_2d = self.occ_map[z_idx] * 255
            try:
                occ_map_2d[x_idx][y_idx] = 125
            except:
                print('out of region')
            cv2.imwrite('obs/'+name+'.png',occ_map_2d)
        z_max=len(self.occ_map) - 1 
        return self.occ_map[z_idx] if z_idx<=z_max else self.occ_map[z_max]
    
    def generate_3obs_img(self,obs,target_vector,name='test',save=False):
        
        z_center=2
        center=11
        tall,height,width=[(2*z_center+1),(2*center+1),(2*center+1)]
        # tall, height,width.  智能体是在最中间的位置
        obs_img=np.zeros([(2*z_center+1),(2*center+1),(2*center+1)])
        
        x,y,z=obs[0],obs[2],obs[4]
        x_idx = round((x-self.X_MIN)/self.gs)
        y_idx = round((y-self.Y_MIN)/self.gs)
        z_idx = round((z-self.Z_MIN)/self.gs)

        z_low=z_idx-z_center
        z_top=z_idx+z_center
        z_max=len(self.occ_map) - 1 
        # fill_z_idx=0
        if z_low<0: # z_low<0 z_top>0
            fill_z_idx=0
            for z_index in range(z_low,0,1):
                obs_img[fill_z_idx]=np.ones([(2*center+1),(2*center+1)])
                fill_z_idx+=1
            for z_index in range(0,z_top+1,1):
                if x_idx>=center and x_idx + center  <=len(self.occ_map[1])-1 and y_idx>=center and y_idx+center <=len(self.occ_map[1][1])-1:
                    for x in range(0,(2*center+1)):
                        obs_img[fill_z_idx][x]=self.occ_map[z_index][x_idx-center+x][y_idx-center:y_idx+(center+1)]
                    fill_z_idx+=1
        elif z_top<=z_max:  # z_low>0,z_top <= z_max
            fill_z_idx=0
            if x_idx>=center and x_idx + center  <=len(self.occ_map[1])-1 and y_idx>=center and y_idx+center <=len(self.occ_map[1][1])-1:
                for z_index in range(z_low,z_top+1,1):
                    for x in range(0,(2*center+1)):
                        obs_img[fill_z_idx][x]=self.occ_map[z_index][x_idx-center+x][y_idx-center:y_idx+(center+1)]
                    fill_z_idx+=1

        else : # z_low>0,z_top > z_max
            fill_z_idx=0
            for z_index in range(z_low,z_max+1,1):
                if x_idx>=center and x_idx + center <=len(self.occ_map[1])-1 and y_idx>=center and y_idx+center <=len(self.occ_map[1][1])-1:
                    for x in range(0,(2*center+1)):
                        obs_img[fill_z_idx][x]=self.occ_map[z_index][x_idx-center+x][y_idx-center:y_idx+(center+1)]
                    fill_z_idx+=1
            while fill_z_idx <2*z_center+1:
                obs_img[fill_z_idx]=np.ones([(2*center+1),(2*center+1)])
                fill_z_idx+=1
        
        # fill target_vector way 1
        # if target_vector[2]>=0:
        #     for _ in range(z_center,2*z_center+1):
        #         if target_vector[0]>0:
        #             obs_img[_][center:center*2+1,center]=0.2
        #         else:
        #             obs_img[_][0:center+1,center]=0.2
        #         if target_vector[1]>0:
        #             obs_img[_][center][center:center*2+1]=0.2
        #         else:
        #             obs_img[_][center][0:center+1]=0.2
        # else:
        #     for _ in range(0,z_center+1):
        #         if target_vector[0]>0:
        #             obs_img[_][center:center*2+1,center]=0.2
        #         else:
        #             obs_img[_][0:center+1,center]=0.2
        #         if target_vector[1]>0:
        #             obs_img[_][center][center:center*2+1]=0.2
        #         else:
        #             obs_img[_][center][0:center+1]=0.2

        # fill target_vector way 2
        if target_vector[2]>=0:
            for _ in range(z_center,2*z_center+1):
                if target_vector[0]>0 :
                    if  target_vector[1]>0:
                        obs_img[_][center:center*2+1,center:center*2+1]=np.where(obs_img[_][center:center*2+1,center:center*2+1]==1 , 1 ,0.5)
                    else:
                        obs_img[_][center:center*2+1,0:center+1]=np.where(obs_img[_][center:center*2+1,0:center+1]==1,1,0.5)
                else:
                    if target_vector[1]>0:
                        obs_img[_][0:center+1,center:center*2+1]=np.where(obs_img[_][0:center+1,center:center*2+1]==1,1,0.5)
                    else:
                        obs_img[_][0:center+1,0:center+1]=np.where(obs_img[_][0:center+1,0:center+1]==1,1,0.5)
        else:
            for _ in range(0,z_center+1):
                if target_vector[0]>0 :
                    if  target_vector[1]>0:
                        obs_img[_][center:center*2+1,center:center*2+1]=np.where(obs_img[_][center:center*2+1,center:center*2+1]==1 , 1 ,0.5)
                    else:
                        obs_img[_][center:center*2+1,0:center+1]=np.where(obs_img[_][center:center*2+1,0:center+1]==1,1,0.5)
                else:
                    if target_vector[1]>0:
                        obs_img[_][0:center+1,center:center*2+1]=np.where(obs_img[_][0:center+1,center:center*2+1]==1,1,0.5)
                    else:
                        obs_img[_][0:center+1,0:center+1]=np.where(obs_img[_][0:center+1,0:center+1]==1,1,0.5)
        # if save:
        #     i=0
        #     obs_img = obs_img * 255
        #     for z_index in range(len(obs_img)) :
        #         occ_map_2d = self.occ_map[z_idx-z_center + z_index if z_idx-z_center + z_index>=0 else 0] * 255
        #         occ_map_2d [x_idx][y_idx] = 125
        #         obs_img[z_index][center][center]=125
        #         cv2.imwrite('obs2/'+str(i)+'.png',obs_img[z_index])
        #         cv2.imwrite('obs2/'+str(i)+'_occ.png',occ_map_2d)
        #         i+=1

        # import pdb;pdb.set_trace()

        return obs_img

# if __name__ == '__main__':
#     m_slam = SLAM()
#     m_slam.reset_occ_map()
#     m_slam.plot_occ_map('test')
    # for i in range(len(m_slam.occ_map)):
    #     occ_map_2d = m_slam.occ_map[i] * 255
    #     time.sleep(0.01)
    #     cv2.imwrite('test/'+str(int(round(time.time() * 1000)))+'.png',occ_map_2d)