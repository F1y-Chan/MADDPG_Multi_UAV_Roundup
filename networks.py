import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#####学习率越大学习得越快
#####衰减步骤越大衰减越慢
#####衰减因子越小衰减越快

#####我们的CriticNetwork网络学习速度要快于Actor，而对于Actor的调节要精细多次的慢调

###全局视角，输入无人机和目标状态以及所有信息，输出奖励Q值
# 定义一个类，继承nn.Module，用于定义critic网络的结构
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        
        # 初始化父类的构造函数，把critic网络的父级参数初始化到self中
        super(CriticNetwork, self).__init__()
        
        # 定义检查点文件路径
        self.chkpt_file = os.path.join(chkpt_dir, name)

        # 父级参数调用，初始化critic网络的参数,构造critic网络的结构
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)# 第一层网络，有109维输入和128维输出
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)# 第二层网络，有128维输入和128维输出
        self.q = nn.Linear(fc2_dims, 1)# 第三层网络，128维输入和1维输出
        
        # 定义优化器和学习率调度器
        self.optimizer = optim.Adam(self.parameters(), lr=beta) # 优化器：Adam，学习率：beta
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33) # 学习率调度器：StepLR，每5000个时间步学习率衰减0.33 
        
        # 定义设备并移动模型到设备上
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # 设备：cuda:0，如果没有cuda设备则使用cpu
        self.to(self.device)

# 定义前向传播函数
    def forward(self, state, action):

        #relu激活函数，用于处理负数输入，将负数输入映射为0
        #cat为state和action的拼接，【256 109】。dim=1表示横着拼接         
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))#第一层网络输入X，当前状态和当前动作，输出当前动作X的Q值
        x = F.relu(self.fc2(x))#第二层网络输入X，有128维输入和128维输出
        q = self.q(x)#第三层网络输入X，有128维输入和1维输出 ，输出当前最终动作X的Q值
        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)    
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

###自身视角，输入无人机的状态，输出无人机的X和Y方向的加速度并限制在[-1,1]之间
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)#自己局部观测状态，26->128
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)#第二层网络，128->128
        self.pi = nn.Linear(fc2_dims, n_actions)#第三层网络，128->2

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))#第一层网络输入X自己的状态，26->128
        x = F.leaky_relu(self.fc2(x))#第二层网络输入X，有128维输入和128维输出
        pi = nn.Softsign()(self.pi(x)) # 第三层网络输入X，有128维输入和2维输出，输出当前最终动作加速度限制幅度[-1,1]

        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

