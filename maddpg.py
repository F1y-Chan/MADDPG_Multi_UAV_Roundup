import os
import torch as T
import torch.nn.functional as F
from agent import Agent
# from torch.utils.tensorboard import SummaryWriter

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.02, fc1=128, 
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        #让变量 agent_idx 依次等于 0, 1, 2, 3，循环执行4次
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, time_step, evaluate):# timestep for exploration
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx],time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):
        if not memory.ready():
            return
        #局部观测 全局状态 动作 奖励 
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()
        #新状态 新全局状态 终止标志

        device = self.agents[0].actor.device#所有智能体都在同一个设备上，随便取哪个都一样

        #将数据转换为张量并移动到设备上
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []
    
        #计算影子无人机在未来想输出的加速度
        for agent_idx, agent in enumerate(self.agents):

            #拼接每个无人机的局势状态，给forward函数使用，计算出影子无人机在未来想输出的加速度
            new_states = T.tensor(actor_new_states[agent_idx], 
                                dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)#新动作
            old_agents_actions.append(actions[agent_idx])#旧动作

        #将所有智能体的加速度拼接起来
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()#计算新状态评分
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_#贝尔曼方程

            critic_value = agent.critic.forward(states, old_actions).flatten()#计算旧状态评分
            critic_loss = F.mse_loss(target, critic_value)

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)            
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
            
        for agent in self.agents:    
            agent.update_network_parameters()
