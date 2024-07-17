import numpy as np
from matplotlib import pyplot as plt


class BernoulliBandit:

    def __init__(self, num_arms: int):
        self.probs = np.random.uniform(size=num_arms)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.num_arms = num_arms

    def trigger(self, idx: int) -> 1 | 0:
        if np.random.random() < self.probs[idx]:
            return 1
        else:
            return 0


class Solver:

    def __init__(self, bandit: BernoulliBandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.num_arms)  # 每根拉杆的尝试次数
        self.regret = 0.0  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, idx: int) -> None:
        self.regret += self.bandit.best_prob - self.bandit.probs[idx]
        self.regrets.append(self.regret)

    def step(self) -> int:
        raise NotImplementedError

    def run(self, steps):
        # 运行一定次数,steps为总运行次数
        for _ in range(steps):
            k = self.step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedySolver(Solver):

    def __init__(self, bandit: BernoulliBandit, epsilon: float, init_prob=1.0) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.num_arms)

    def step(self) -> int:
        if np.random.random() < self.epsilon:
            idx = np.random.randint(0, self.bandit.num_arms)
        else:
            idx = np.argmax(self.estimates)
        # 得到此次的奖励
        r = self.bandit.trigger(idx)
        # 更新估计值
        self.estimates[idx] = self.estimates[idx] + (r - self.estimates[idx]) / (
            self.counts[idx] + 1
        )
        return idx


class DecayingEpsilonGreedy(Solver):
    """epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类"""

    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.epsilon = "decay"
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.total_count = 0

    def step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随次数衰减
            k = np.random.randint(0, self.bandit.num_arms)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.trigger(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


class UCB(Solver):
    """UCB算法,继承Solver类"""

    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.coef = coef

    def step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))
        )  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.trigger(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class ThompsonSampling(Solver):
    """汤普森采样算法,继承Solver类"""

    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.num_arms)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.num_arms)  # 列表,表示每根拉杆奖励为0的次数

    def step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.trigger(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += 1 - r  # 更新Beta分布的第二个参数
        return k


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.num_arms)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)  # 设定随机种子,使实验具有可重复性
    bandit = BernoulliBandit(num_arms=10)
    solvers = (
        [EpsilonGreedySolver(bandit, epsilon=e) for e in [1e-4, 0.01, 0.1, 0.25, 0.5]]
        + [DecayingEpsilonGreedy(bandit)]
        + [UCB(bandit, coef=2)]
        + [ThompsonSampling(bandit)]
    )
    [s.run(5000) for s in solvers]
    plot_results(
        solvers,
        [s.__class__.__name__ for s in solvers],
    )
