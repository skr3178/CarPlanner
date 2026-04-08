# Details of the architecture

Given an initial state s0 and all possible Nmode
modes, the trajectory selector evaluates and assigns scores
to each mode. The trajectory generator then produces Nmode
trajectories that correspond to their respective modes. For
trajectory generator, the initial state s0 is replicated Nmode
times, each associated with one of the Nmode modes, effectively
creatingNmode parallel worlds. The policy is executed
within these previewed worlds. During the policy rollout, a
trajectory predictor acts as the state transition model, generating
future poses of traffic agents across all time horizons.

## Non-reactive Transition Model

This module takes the initial state s0 as input and outputs
the future trajectories of traffic agents. The initial state is
processed by agent and map encoders, followed by a selfattention
Transformer encoder [38] to fuse the agent and
map features. The agent features are then decoded into future
trajectories.
Agent and map encoders. The state s0 contains both map
and agent information. The map information m consists of
Nm,1 polylines and Nm,2 polygons. The polylines describe
lane centers and lane boundaries, with each polyline containing
3Np points, where 3 corresponds to the lane center,
the left boundary, and the right boundary. Each point is with
dimension Dm = 9 and includes the following attributes: x,
y, heading, speed limit, and category. When concatenated,
the points of the left and right boundaries together with
the center point yield a dimension of Nm,1 × Np × 3Dm.
We leverage a PointNet [26] to extract features from the
points of each polyline, resulting in a dimensionality of
Nm,1 × D, where D represents the feature dimension. The
polygons represent intersections, crosswalks, stop lines, etc,
with each polygon containing Np points. We utilize another
PointNet to extract features from the points of each polygon,
producing a dimension of Nm,2 × D. We then concatenate
the features from both polylines and polygons to
form the overall map features, resulting in a dimension of
Nm × D. The agent information A consists of N agents,
where each agent maintains poses for the past H time steps.
Each pose is with dimension Da = 10 and includes the
following attributes: x, y, heading, velocity, bounding box,
time step, and category. Consequently, the agent information
has a dimension of N × H × Da. We apply another
PointNet to extract features from the poses of each agent,
yielding an agent feature dimension of N × D.

## Mode Selector
This module takes s0 and longitudinal-lateral decomposed
mode information as input and outputs the probability of
each mode. The number of modes Nmode = NlatNlon.
## Route-speed decomposed mode. 
To capture the longitudinal
behaviors, we generate N{\text {lon}} modes that represent the
average speed of the trajectory associated with each mode.
Each longitudinal mode c{\text {lon},j} is defined as a scalar value of
\frac {j}{N{\text {lon}}} , repeated across a dimension D . As a result, the dimensionality
of the longitudinal modes is N{\text {lon}} \times D . For lateral
behaviors, we identify N{\text {lat}} possible routes from the map
using a graph search algorithm. These routes correspond to
the lanes available for the ego vehicle. The dimensionality
of these routes is N{\text {lat}} \times Nr \times Dm . We employ another
PointNet to aggregate the features of the Nr points along
each route, producing a lateral mode with a dimension of
N{\text {lat}} \times D . To create a comprehensive mode representation
c, we combine the lateral and longitudinal modes, resulting
in a combined dimension of N{\text {lat}} \times N{\text {lon}} \times 2D . To align this
mode information with other feature dimensions, we pass it
through a linear layer, mapping it back to N{\text {lat}} \times N{\text {lon}} \times D .
## Query-based Transformer decoder. 
This decoder is employed
to fuse the mode features with map and agent features
derived from s0 . In this framework, the mode serves
as the query, while the map and agent information act as the
keys and values. The updated mode features are decoded
through a multi-layer perceptron (MLP) to yield the scores
for each mode, which are subsequently normalized using
the softmax operator.

## Trajectory Generator
This module operates in an auto-regressive manner, recurrently
decoding the next pose of the ego vehicle at, given
the current state st, and consistent mode information c.
## Invariant-view module (IVM). 
Before feeding the mode
and state into the network, we preprocess them to eliminate
time information. For the map and agent information in
state st, we select the K-nearest neighbors (KNN) to the
ego current pose and only feed these into the policy. K
is set to the half of map and agent elements respectively.
Regarding the routes that capture lateral behaviors, we filter
out the segments where the point closest to the current pose
of the ego vehicle is the starting point, retaining Kr points.
In this case, Kr is set to a quarter of Nr points in one route.
Finally, we transform the routes, agent, and map poses into
the coordinate frame of the ego vehicle at the current time
step t. We subtract the historical time steps t − H : t from
the current time step t, yielding time steps in range −H : 0.

## Query-based Transformer decoder. 
We employ the same
backbone network architecture as the mode selector, but
with different query dimensions. Due to the IVM and the
fact that different modes yield distinct states, the map and
agent information cannot be shared among modes. As a result,
we fuse information for each individual mode. Specifically,
the query dimension is 1x D, while the dimensions of
the keys and values are (N+N_m)*D. The output feature
dimension remains 1*D. ote that Transformer decoder
can process information from multiple modes in parallel,
eliminating the need to handle each mode sequentially.
## Policy output. 
The mode feature is processed by two distinct
heads: a policy head and a value head. Each head comprises
its own MLP to produce the parameters for the action
distribution and the corresponding value estimate. We employ
a Gaussian distribution to model the action distribution,
where actions are sampled from this distribution during
training. In contrast, during inference, we utilize the
mean of the distribution to determine the actions.
## Rule-augmented Selector
This module is only utilized during inference and takes as
input the initial state s0 , the multi-modal ego-planned trajectories,
and the predicted future trajectories of agents. It
calculates driving-oriented metrics such as safety, progress,
comfort. A comprehensive score is obtained by the
weighted sum of rule-based scores and the mode scores provided
by the mode selector. The ego-planned trajectory with
the highest score is selected as the output of the planner.

## 3.4. Training
We first train the non-reactive transition model and freeze
the weights during the training of the mode selector and trajectory
generator. Instead of feeding all modes to the generator,
we apply a winner-takes-all strategy, wherein a positive
mode is assigned based on the ego ground-truth trajectory
and serves as a condition for the trajectory generator.
## Mode assignment. 
For the lateral mode, we assign the
route closest to the endpoint of ego ground-truth trajectory
as the positive lateral mode. For the longitudinal mode, we
partition the longitudinal space into N{\text {lon}} intervals and assign
the interval containing the endpoint of the ground-truth
trajectory as the positive longitudinal mode.
## Reward function. 
To handle diverse scenarios, we use the
negative displacement error (DE) between the ego future
pose and the ground truth as a universal reward. We also introduce
additional terms to improve trajectory quality: collision
rate and drivable area compliance. If the future pose
collides or falls outside the drivable area, the reward is set
to -1; otherwise, it is 0.
## Mode dropout. 
In some cases, there are no available routes
for ego to follow. However, since routes serve as queries in
Transformer, the absence of a route can lead to unstable or
hazardous outputs. To mitigate this issue, we implement a
mode dropout module during training that randomly masks
routes to prevent over-reliance on this information.
Loss function. For the selector, we use cross-entropy loss
that is the negative log-likelihood of the positive mode and a
side task that regresses the ego ground-truth trajectory. For
the generator, we use PPO [31] loss that consists of three
parts: policy improvement, value estimation, and entropy.
Full description can be found in supplementary.

# Experimental setup

## Dataset and simulator. 
We use nuPlan [2], a large-scale
closed-loop platform for studying trajectory planning in autonomous
driving, to evaluate the efficacy of our method.
The nuPlan dataset contains driving log data over 1,500
hours collected by human expert drivers across 4 diverse
cities. It includes complex, diverse scenarios such as lane
follow and change, left and right turn, traversing intersections
and bus stops, roundabouts, interaction with pedestrians,
etc. As a closed-loop platform, nuPlan provides a simulator
that uses scenarios from the dataset as initialization.
During the simulation, traffic agents are taken over by logreplay
(non-reactive) or an IDM [37] policy (reactive). The
ego vehicle is taken over by user-provided planners. The
simulator lasts for 15 seconds and runs at 10 Hz. At each
timestamp, the simulator queries the planner to plan a trajectory,
which is tracked by an LQR controller to generate
control commands to drive the ego vehicle.
## Benchmarks and metrics. 
We use two benchmarks:
Test14-Random and Reduced-Val14 for comparing with
other methods and analyzing the design choices within
our method. The Test14-Random provided by PlanTF [4]
contains 261 scenarios. The Reduced-Val14 provided by
PDM [7] contains 318 scenarios.
We use the closed-loop score (CLS) provided by the official
nuPlan devkit† to assess the performance of all methods.
The CLS score comprehends different aspects such as
safety (S-CR, S-TTC), drivable area compliance (S-Area),
progress (S-PR), comfort, etc. Based on the different behavior
types of traffic agents, CLS is detailed into CLS-NR
(non-reactive) and CLS-R (reactive).

## Implementation details. 
We follow PDM [7] to construct
our training and validation splits. The size of the training
set is 176,218 where all available scenario types are used,
with a number of 4,000 scenarios per type. The size of the
validation set is 1,118 where 100 scenarios with 14 types are
selected. We train all models with 50 epochs in 2 NVIDIA
3090 GPUs. The batch size is 64 per GPU.We use AdamW
optimizer with an initial learning rate of 1e-4 and reduce the
learning rate when the validation loss stops decreasing with
a patience of 0 and decrease factor of 0.3. For RL training,
we set the discount γ = 0.1 and the GAE parameter λ =
0.9. The weights of value, policy, and entropy loss are set to
3, 100, and 0.001, respectively. The number of longitudinal
modes is set to 12 and a maximum number of lateral modes
are set to 5.