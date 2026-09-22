import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

# 每个智能体的类，包含影子网络和本体网络。
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
        
        #gamma是折扣因子，tau是软更新参数
        self.gamma = gamma
        self.tau = tau

        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        # 本体网络
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        # 影子网络
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)# tau=1时初始化影子网络参数，与本体网络参数相同

    # 选择动作，
    def choose_action(self, observation, time_step, evaluate=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)# 将观测转换为张量并移动到设备上，加【】是为了变成二维张量
        actions = self.actor.forward(state)#喂给Actor主力

        # 噪声设置，用于探索，
        # noise_scale是噪声的幅值，上限为0.75，随时间步长衰减
        # noise是随机噪声方向和实际大小，用于探索状态空间
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise
        action = actions + noise

        #action_np = [速度控制量1, 速度控制量2]
        # 速度控制量的范数不能超过0.04，否则会导致UAV飞离目标
        action_np = action.detach().cpu().numpy()[0]
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        #数据手册的定义，拆解数据手册，将所有权重等内容分层dict，克隆后计算主力网络参数
        # tau*新数值 + (1-tau)*旧数值，用于软更新影子网络参数
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)


        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    # 保存模型
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
  # 加载模型
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
