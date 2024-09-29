# reward machine

## 来自你的消息:
Taxi:
There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the 5x5 grid world. The taxi starts off at a random square and the passenger at one of the designated locations.
The goal is move the taxi to the passenger’s location, pick up the passenger, move to the passenger’s desired destination, and drop off the passenger. Once the passenger is dropped off, the episode ends.
The player receives positive rewards for successfully dropping-off the passenger at the correct location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and for each step where another reward is not received.

Rewards
-1 per step unless other reward is triggered.
+20 delivering passenger.
-10 executing “pickup” and “drop-off” actions illegally.
An action that results a noop, like moving into a wall, will incur the time step penalty. Noops can be avoided by sampling the action_mask returned in info.

Episode End
The episode ends if the following happens:
Termination: 1. The taxi drops off the passenger.
Truncation (when using the time_limit wrapper): 1. The length of the episode is 200.
Please use the reward machine to describe it in mathmatical.

## 来自 ChatGPT 的消息:
I can define a reward machine for the Taxi problem as follows:

Set of propositional symbols $\mathcal{P}$:
- $p_1$: the taxi is at the passenger's location.
- $p_2$: the passenger is in the taxi.
- $p_3$: the taxi is at the drop-off location for the passenger.
- $p_4$: the taxi is at a wall.

Set of states $S$:
- $U = \{u_0, u_1, u_2, u_3, u_4, u_5, u_6, u_7 \}$ where:
  - $u_0$: initial state.
  - $u_1$: taxi is at any location and passenger is at their desired drop-off location.
  - $u_2$: taxi is at any location and passenger is at their desired pick-up location.
  - $u_3$: taxi and passenger are at any non-desired location.
  - $u_4$: taxi is at any location and passenger is in the taxi.
  - $u_5$: taxi and passenger are both at any drop-off location.
  - $u_6$: taxi is at a wall.
  - $u_7$: terminal state.

Set of actions $A$: the set of all possible actions in the Taxi problem.

State-transition function $\delta_u: U \times 2^{\mathcal{P}} \rightarrow U\cup F$:
- $\delta_u(u_0, \{p_1\}) = u_2$: passenger is at pick-up location.
- $\delta_u(u_0, \{p_3\}) = u_1$: passenger is at drop-off location.
- $\delta_u(u_0, \emptyset) = u_3$: passenger is at a non-desired location.
- $\delta_u(u_0, \{p_1, p_3\}) = u_7$: passenger is already at their destination.
- $\delta_u(u_0, \{p_4\}) = u_6$: taxi is at a wall.

- $\delta_u(u_2, \{p_2\}) = u_4$: passenger is picked up.
- $\delta_u(u_2, \emptyset) = u_3$: passenger is at a non-desired location.
- $\delta_u(u_2, \{p_4\}) = u_6$: taxi is at a wall.

- $\delta_u(u_4, \{p_1\}) = u_5$: passenger is dropped off.
- $\delta_u(u_4, \emptyset) = u_3$: passenger is at a non-desired location.
- $\delta_u(u_4, \{p_4\}) = u_6$: taxi is at a wall.

- $\delta_u(u_1, \emptyset) = u_7$: passenger is dropped off.
- $\delta_u(u_1, \{p_4\}) = u_6$: taxi is at a wall.

- $\delta_u(u_3, \{p_1\}) = u_2$: passenger is at pick-up location.
- $\delta_u(u_3, \{p_3\}) = u_1$: passenger is at drop-off location.
- $\delta_u(u_3, \{p_4\}) = u_6$: taxi is at a wall.

- $\delta_u(u_6, \emptyset) = u_6$: taxi remains at the wall.
- $\delta_u(u_6, \{p_1, p_2, p_3\}) = u_7$: illegal action, episode ends.

Reward-transition function $\delta_r: U \rightarrow [U\times U \rightarrow \mathcal{R}]$:
- $\delta_r(u_0) = -1$: no other reward is triggered.
- $\delta_r(u_1) = +20$: passenger is delivered.
- $\delta_r(u_2) = -10$: pickup is illegal.
- $\delta_r(u_3) = -1$: no other reward is triggered.
- $\delta_r(u_4) = -10$: dropoff is illegal.
- $\delta_r(u_5) = +20$: passenger is delivered.
- $\delta_r(u_6) = -1$: no other reward is triggered.
- $\delta_r(u_7) = 0$: terminal state.

In this reward machine, the states represent various configurations of the taxi and passenger, and the transitions reflect the outcomes of taking different actions in those configurations. The rewards for each transition are based on the rules of the Taxi problem.