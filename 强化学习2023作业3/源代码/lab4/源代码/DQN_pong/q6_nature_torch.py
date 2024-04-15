import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.tensor import Tensor
from utils.test_env import EnvTest
from q4_schedule import LinearExploration, LinearSchedule
from core.deep_q_learning_torch import DQN

from configs.q6_nature import config


class NatureQN(DQN):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """


    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """
        self.input_shape = list(self.env.observation_space.shape)
        state_shape = self.input_shape
        img_height, img_width, n_channels = state_shape
        # print("img_height",img_height)
        # print("img_width",img_width)
        # print("n_channels",n_channels)
        # print("n_channels * self.config.state_history",n_channels * self.config.state_history)
        num_actions = self.env.action_space.n

        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################
        # 创建Q网络
        self.q_network = nn.Sequential(
            nn.Conv2d(in_channels=n_channels * self.config.state_history, out_channels=32, kernel_size=8, stride=4,
                      padding=((4 - 1) * img_height - 4 + 8) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                      padding=((2 - 1) * img_height - 2 + 4) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=((1 - 1) * img_height - 1 + 3) // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * img_height * img_width, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

        # 创建Target网络，与Q网络具有相同的配置但从头开始初始化
        self.target_network = nn.Sequential(
            nn.Conv2d(in_channels=n_channels * self.config.state_history, out_channels=32, kernel_size=8, stride=4,
                      padding=((4 - 1) * img_height - 4 + 8) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                      padding=((2 - 1) * img_height - 2 + 4) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=((1 - 1) * img_height - 1 + 3) // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * img_height * img_width, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )
        # 模型的输入大小为img_height * img_width的图像，通道数为n_channels * self.config.state_history
        input_size = (img_height, img_width, n_channels * self.config.state_history)

        return input_size
        ##############################################################
        ######################## END YOUR CODE #######################
    def feature_size(self):
        # print(self.q_network.conv3(self.q_network.conv2(self.q_network.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1))
        return self.q_network.conv3(self.q_network.conv2(self.q_network.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        state = state.permute(0,3,1,2)
        # print(f'Input shape after flattening = {input.shape}')
        if network == 'q_network':
            out = self.q_network(state)
        elif network == 'target_network':
            out = self.target_network(state)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out

    def update_target(self):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights.

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.

        Hint:
            1. look up saving and loading pytorch models
        """

        ##############################################################
        ################### YOUR CODE HERE - 1-2 lines ###############
        self.target_network.load_state_dict(self.q_network.state_dict())
        ##############################################################
        ######################## END YOUR CODE #######################


    # def calc_loss(self, q_values : Tensor, target_q_values : Tensor,
    #                 actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
    def calc_loss(self, q_values, target_q_values,
                    actions, rewards, done_mask):
        """
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a')
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')
            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            done_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

        Hint:
            You may find the following functions useful
                - torch.max
                - torch.sum
                - torch.nn.functional.one_hot
                - torch.nn.functional.mse_loss
        """
        # you may need this variable
        num_actions = self.env.action_space.n
        gamma = self.config.gamma

        ##############################################################
        ##################### YOUR CODE HERE - 3-5 lines #############
        q_samp = rewards + gamma * torch.max(target_q_values, dim=1)[0] * (~done_mask)
        actions = actions.to(torch.int64)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(q_samp, q_values)
        return loss
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer(self):
        """
        Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
        parameters

        Hint:
            - Look up torch.optim.Adam
            - What are the input to the optimizer's constructor?
        """
        ##############################################################
        #################### YOUR CODE HERE - 1 line #############
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.00025)
        # self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-6)
        ##############################################################
        ######################## END YOUR CODE #######################


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
