import numpy as np

class MDP:
    '''一个简单的MDP类，它包含如下成员'''

    def __init__(self,T,R,discount):
        '''构建MDP类

        输入:
        T -- 转移函数: |A| x |S| x |S'| array
        R -- 奖励函数: |A| x |S| array
        discount -- 折扣因子: scalar in [0,1)

        构造函数检验输入是否有效，并在MDP对象中构建相应变量'''

        assert T.ndim == 3, "转移函数无效，应该有3个维度"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "无效的转换函数：它具有维度 " + repr(T.shape) + ", 但它应该是(nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "无效的转移函数：某些转移概率不等于1"
        self.T = T
        assert R.ndim == 2, "奖励功能无效：应该有2个维度"
        assert R.shape == (self.nActions,self.nStates), "奖励函数无效：它具有维度 " + repr(R.shape) + ", 但它应该是 (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "折扣系数无效：它应该在[0,1]中"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''值迭代法
        V <-- max_a R^a + gamma T^a V

        输入:
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代次数的限制：标量 (默认值: infinity)
        tolerance -- ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = 0
        V = initialV
        #填空部分
        V_n=V
        error=np.inf
        while error>tolerance:
            iterId+=1
            V_n1 = np.zeros_like(V_n)
            for i in range(0,self.nStates): #for each state
                V_all_action=[]
                for j in range(0,self.nActions): #for each action
                    V_current_action=self.R[j][i]+self.discount*np.dot(self.T[j,i,:],V_n)
                    V_all_action.append(V_current_action)
                V_n1[i]=max(V_all_action)#当前状态的最大value
            error=np.sum(np.abs(V_n - V_n1))
            V_n=V_n1
        V=V_n
        epsilon=error
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''从值函数中提取具体策略的程序
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- 值函数: 大小为|S|的array

        Output:
        policy -- 策略: 大小为|S|的array'''
        #填空部分
        policy = np.zeros(len(V))
        for s in range(self.nStates):
            maxValue = -np.inf
            for a in range(self.nActions):
                V_q= self.R[a][s] + self.discount * np.sum(self.T[a][s] * V)
                if V_q > maxValue:
                    maxValue = V_q
                    pi = a
            policy[s] = pi
        return policy

    def evaluatePolicy(self,policy):
        '''通过求解线性方程组来评估政策
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- 策略: 大小为|S|的array

        Ouput:
        V -- 值函数: 大小为|S|的array'''
        #填空部分
        V=np.zeros(self.nStates)
        # 迭代求法
        while 1:
            V_new=np.zeros(self.nStates)
            for s in range(0,self.nStates):
                a=policy[s]
                V_new[s]=self.R[a][s]+self.discount*np.dot(self.T[a][s],V)
            if np.any(np.abs(V_new-V))<1e-10:
                break
            V=V_new

        # 求解线性方程组
        # for s in range(self.nStates):
        #     a=policy[s]
        #     I = np.eye(self.nStates)
        #     G1 = I - self.discount * self.T[a]  # A*S*S
        #     V = np.linalg.solve(G1, self.R[a])

        return V

    def policyIteration(self, initialPolicy, nIterations=1000):
        '''策略迭代程序:  在策略评估(solve V^pi = R^pi + gamma T^pi V^pi) 和
        策略改进 (pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: inf)

        Outputs:
        policy -- 策略: 大小为|S|的array
        V -- 值函数: 大小为|S|的array
        iterId --策略迭代执行的次数 : 标量'''

        V = np.zeros(self.nStates)
        policy = initialPolicy
        iterId = 0
        # 填空部分
        while 1:
            policy = policy.astype(int)
            # 策略评估
            while 1:
                V_new = np.zeros(self.nStates)
                for s in range(0, self.nStates):
                    a = policy[s]
                    V_new[s] = self.R[a][s] + self.discount * np.dot(self.T[a][s], V)
                if np.any(np.abs(V_new - V)) < 1e-10:
                    break
                V = V_new

            # 策略改进
            policy_new=self.extractPolicy(V)
            if all(policy==policy_new):
                break
            policy=policy_new
            iterId += 1

        return [policy, V, iterId]

    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''部分的策略评估:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- 策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: infinity)
        tolerance --  ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 迭代执行的次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        V = initialV
        iterId = 0
        epsilon = 0
        #填空部分
        while 1:
            iterId+=1
            V_new=np.zeros(self.nStates)
            for s in range(0,self.nStates):
                a=policy[s]
                V_new[s]=self.R[a][s]+self.discount*np.dot(self.T[a][s],V)
            if np.any(np.abs(V_new-V))<1e-10 or iterId >= nIterations:
                epsilon=np.max(np.abs(V_new-V))
                break
            V=V_new
        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''修改的策略迭代程序: 在部分策略评估 (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        和策略改进(pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nEvalIterations -- 每次部分策略评估时迭代次数的限制: 标量 (默认值: 5)
        nIterations -- 修改的策略迭代中迭代次数的限制: 标量 (默认值: inf)
        tolerance -- ||V^n-V^n+1||_inf的阈值: scalar (默认值: 0.01)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V --值函数: 大小为|S|的array
        iterId -- 修改后策略迭代执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = tolerance+1
        policy = initialPolicy
        V = initialV
        #填空部分
        while epsilon > tolerance:
            iterId+=1
            # 部分策略评估  repeat a few times
            for k in range (0,nEvalIterations):
                V_new = np.zeros(self.nStates)
                for s in range(0, self.nStates):
                    a = int(policy[s])
                    V_new[s] = self.R[a][s] + self.discount * np.dot(self.T[a][s], V)
                V = V_new
            # 策略改进
            policy=self.extractPolicy(V)
            V_n1=np.zeros(self.nStates)
            for s in range(0,self.nStates):
                V_all = []
                for a in range(0,self.nActions):
                    V_current = self.R[a][s] + self.discount * np.dot(self.T[a][s], V)
                    V_all.append(V_current)
                V_n1[s]=max(V_all)

            epsilon=np.max(np.abs((V_n1-V)))
            V=V_n1

        return [policy,V,iterId,epsilon]
        