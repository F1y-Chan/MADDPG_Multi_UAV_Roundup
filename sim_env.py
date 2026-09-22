import numpy as np                       # 数值计算核心库，向量运算、随机数、数组操作
import itertools                          # 用于将多维观测列表展平为一维
import matplotlib.pyplot as plt           # 绘图：轨迹、障碍物、目标点
import matplotlib.transforms as transforms # 仿射变换：旋转+平移无人机图标使其朝向速度方向
import matplotlib.cm as cm                # 颜色映射：render_anime 中轨迹渐变色
import matplotlib.image as mpimg          # 读取 PNG 图标文件（无人机图片）
from gymnasium import spaces              # Gymnasium 强化学习接口，定义动作/观测空间
from math_tool import *                   # 自定义数学工具：激光测距、三角形面积计算等
import matplotlib.backends.backend_agg as agg  # Agg 后端离屏渲染，输出到 NumPy 数组
from PIL import Image                     # 处理渲染图像帧，用于保存 GIF/视频
import random                             # 内置随机库，reset 中生成随机种子
import copy                               # 深拷贝，将当前位置存入历史轨迹避免引用污染

# ==================== 多无人机围捕仿真环境类 ====================
class UAVEnv:

    # ==================== 初始化环境参数 ====================
    def __init__(self,length=2,num_obstacle=3,num_agents=4):
        # --- 场景几何与规模 ---
        self.length = length               # 方形区域边长
        self.num_obstacle = num_obstacle   # 障碍物数量
        self.num_agents = num_agents       # 智能体总数（3架追击无人机 + 1个逃逸目标）
        
        # --- 动力学约束 ---
        self.time_step = 0.5               # 仿真时间步长 Δt
        self.v_max = 0.1                   # 追击无人机的最大速度
        self.v_max_e = 0.12                # 逃逸目标的最大速度（略快于追击者，增加围捕难度）
        self.a_max = 0.04                  # 追击无人机的最大加速度
        self.a_max_e = 0.05                # 逃逸目标的最大加速度
        
        # --- 激光雷达传感器参数 ---
        self.L_sensor = 0.2                # 激光雷达最大感知距离
        self.num_lasers = 16               # 每个智能体的激光束数量（360°均匀分布）

        # 初始化所有智能体的激光读数（全部设为最大感知距离，表示未探测到障碍物）
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        
        # --- 智能体标识与随机种子 ---
        self.agents = ['agent_0','agent_1','agent_2','target']  # 前3个是追击无人机，最后1个是目标
        self.info = np.random.get_state()  # 保存 NumPy 随机状态，用于复现
        
        # --- 障碍物与轨迹 ---
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]  # 创建障碍物对象列表
        self.history_positions = [[] for _ in range(num_agents)]         # 每个智能体的历史位置轨迹

        # --- 动作空间：每个智能体输出 2 维连续加速度 [a_x, a_y] ---
        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            }
        # --- 观测空间：追击者 26 维，目标 23 维 ---
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }

    # ==================== 重置环境 ====================
    def reset(self):
        """重置环境：随机初始化所有智能体的位置和速度，返回初始观测"""
        SEED = random.randint(1,1000)       # 生成随机种子（1-1000）
        random.seed(SEED)                   # 设置随机种子以确保可复现
        self.multi_current_pos = []         # 所有智能体当前位置列表
        self.multi_current_vel = []         # 所有智能体当前速度列表
        self.history_positions = [[] for _ in range(self.num_agents)]  # 清空历史轨迹
        for i in range(self.num_agents):
            if i != self.num_agents - 1:    # 追击无人机：初始位置在 [0.1, 0.4] 范围内随机
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else:                           # 逃逸目标：初始位置固定在右上角 (0.5, 1.75)
                self.multi_current_pos.append(np.array([0.5,1.75]))
            self.multi_current_vel.append(np.zeros(2))  # 所有智能体初始速度为 [0, 0]

        # 更新所有智能体的激光雷达读数，检测碰撞
        self.update_lasers_isCollied_wrapper()
        # 获取所有智能体的观测向量（展平后的一维列表）
        multi_obs = self.get_multi_obs()
        return multi_obs

    # ==================== 执行一步环境交互 ====================
    def step(self,actions):
        """执行一步环境交互：更新运动状态 → 碰撞检测 → 计算奖励
        参数: actions -- 所有智能体的动作列表 [agent0_action, agent1_action, agent2_action, target_action]
        返回: multi_next_obs, rewards, dones
        """
        # 记录动作执行前各追击者到目标的距离（用于奖励计算）
        last_d2target = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:   # 仅追击者记录与目标的距离
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))
            
            # 根据加速度更新速度：v_new = v_old + a * Δt
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            # 速度限幅：追击者不超过 v_max，目标不超过 v_max_e
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            # 根据速度更新位置：pos_new = pos_old + v * Δt
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # --- 更新移动障碍物位置，处理边界反弹 ---
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            for dim in [0, 1]:  # 检查 x 和 y 两个维度
                if obs.position[dim] - obs.radius < 0:          # 触碰左/下边界
                    obs.position[dim] = obs.radius               # 修正到边界上
                    obs.velocity[dim] *= -1                      # 速度反向（反弹）
                elif obs.position[dim] + obs.radius > self.length:  # 触碰右/上边界
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        # --- 碰撞检测、奖励计算、获取新观测 ---
        # 注意：以下三步顺序不可打乱！
        Collided = self.update_lasers_isCollied_wrapper()          # 更新激光读数，检测碰撞
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)  # 计算奖励和终止标志
        multi_next_obs = self.get_multi_obs()                      # 获取所有智能体的下一观测

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs
    
    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = [] # dim 3 only for target
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ] # dim 4
            S_team = [] # dim 4 for 3 agents 1 target
            S_target = [] # dim 2
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1: 
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length,pos_other[1]/self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d/np.linalg.norm(2*self.length))

            S_obser = self.multi_current_lasers[i] # dim 16

            if i != self.num_agents - 1:
                single_obs = [S_uavi,S_team,S_obser,S_target]
            else:
                single_obs = [S_uavi,S_obser,S_evade_d]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
            
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1 = 0.7 # r_near
        mu2 = 0.4 # r_safe
        mu3 = 0.01 # r_multi_stage
        mu4 = 5 # r_finish
        d_capture = 0.3
        d_limit = 0.75
        ## 1 reward for single rounding-up-UAVs:
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV:
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        # print(rewards[-1])
        # 3.2 stage-1 track
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:3] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:3] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max))
            rewards[0:3] += mu3*r_capture
        
        ## 4 finish rewards
        if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3]):
            rewards[0:3] += mu4*10
            dones = [True] * self.num_agents

        return rewards,dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):

        plt.clf()
        
        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)
            
            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()
        
        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()

class obstacle():
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03 
        speed = 0.00 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)