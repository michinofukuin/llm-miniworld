# Taxi

## 来自你的消息:

Determine the reward machine for "Taxi" game.

## 来自 ChatGPT 的消息:

The reward machine for the "Taxi" game can be defined as follows:

1. Set of Propositional Symbols:

- At location 1 (pick-up location 1)
- At location 2 (pick-up location 2)
- At location 3 (pick-up location 3)
- At location 4 (pick-up location 4)
- Passenger in taxi
- At location 1 (drop-off location 1)
- At location 2 (drop-off location 2)
- At location 3 (drop-off location 3)
- At location 4 (drop-off location 4)

2. Set of States:

- U = {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20}

3. Set of Actions:

- A = {north, south, east, west, pickup, dropoff}

4. Initial State:

- u0 = s0

5. Terminal States:

- F= {s16, s17, s18, s19, s20}

6. State-Transition Function:

- δu: U × 2^P → U ∪ F

δu(s0, {At location 1}) = s1
δu(s0, {At location 2}) = s2
δu(s0, {At location 3}) = s3
δu(s0, {At location 4}) = s4

δu(s1, {Passenger in taxi}) = s5
δu(s1, ∅) = s0

δu(s2, {Passenger in taxi}) = s6
δu(s2, ∅) = s0

δu(s3, {Passenger in taxi}) = s7
δu(s3, ∅) = s0

δu(s4, {Passenger in taxi}) = s8
δu(s4, ∅) = s0

δu(s5, {At location 1}) = s9
δu(s5, {At location 2}) = s10
δu(s5, {At location 3}) = s11
δu(s5, {At location 4}) = s12

δu(s6, {At location 1}) = s9
δu(s6, {At location 2}) = s10
δu(s6, {At location 3}) = s11
δu(s6, {At location 4}) = s12

δu(s7, {At location 1}) = s9
δu(s7, {At location 2}) = s10
δu(s7, {At location 3}) = s11
δu(s7, {At location 4}) = s12

δu(s8, {At location 1}) = s9
δu(s8, {At location 2}) = s10
δu(s8, {At location 3}) = s11
δu(s8, {At location 4}) = s12

δu(s9, {Passenger in taxi}) = s13
δu(s9, ∅) = s5

δu(s10, {Passenger in taxi}) = s14
δu(s10, ∅) = s6

δu(s11, {Passenger in taxi}) = s15
δu(s11, ∅) = s7

δu(s12, {Passenger in taxi}) = s16
δu(s12, ∅) = s8

δu(s13, {At location 1}) = s17
δu(s13, {At location 2}) = s18
δu(s13, {At location 3}) = s19
δu(s13, {At location 4}) = s20

δu(s14, {At location 1}) = s17
δu(s14, {At location 2}) = s18
δu(s14, {At location 3}) = s19
δu(s14, {At location 4}) = s20

δu(s15, {At location 1}) = s17
δu(s15, {At location 2}) = s18
δu(s15, {At location 3}) = s19
δu(s15, {At location 4}) = s20

δu(s16, ∅) = s0
δu(s17, ∅) = s0
δu(s18, ∅) = s0
δu(s19, ∅) = s0
δu(s20, ∅) = s0

7. Reward-Transition Function:

- δr: U → [U × U → R]

δr(s0) = -1

δr(s1) = -1

δr(s2) = -1

δr(s3) = -1

δr(s4) = -1

δr(s5) = -1

δr(s6) = -1

δr(s7) = -1

δr(s8) = -1

δr(s9) = -1

δr(s10) = -1

δr(s11) = -1

δr(s12) = -1

δr(s13) = 20 if passenger delivered, -10 if illegal pickup or dropoff

δr(s14) = 20 if passenger delivered, -10 if illegal pickup or dropoff

δr(s15) = 20 if passenger delivered, -10 if illegal pickup or dropoff

δr(s16) = 20

δr(s17) = -1

δr(s18) = -1

δr(s19) = -1

δr(s20) = -1

Note: In the reward-transition function, a reward of -1 is given for each step that does not trigger any other reward. A reward of +20 is given if the passenger is successfully delivered to the correct location, and a reward of -10 is given if an illegal pickup or dropoff is attempted. For states s13, s14, and s15, the reward depends on whether the passenger is delivered successfully or not, and whether the pickup or dropoff is legal or not. Finally, states s16 to s20 are terminal states and have fixed rewards of 20 or -1.
