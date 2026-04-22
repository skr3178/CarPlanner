The notation $s^i_{t-H:t}$
This is a slicing notation (like Python's list[t-H:t]). It means: "the sequence of poses for agent ii
i from time step t−Ht-H
t−H up to time step tt
t."
So $s^i_{t-H:t}$ expands to:
st−Hi,st−H+1i,st−H+2i,…,st−1i,stis^i_{t-H}, s^i_{t-H+1}, s^i_{t-H+2}, \ldots, s^i_{t-1}, s^i_tst−Hi​,st−H+1i​,st−H+2i​,…,st−1i​,sti​
That is, the history window of agent ii
i's poses — where each skis^i_k
ski​ is a single pose (containing x, y, heading, velocity, bounding box, time step, category — the 10-dim DaD_a
Da​ vector).
What H means with "2 seconds of history"
Here's where you have to be careful: H is a count of time steps, not seconds.
The nuPlan simulator runs at 10 Hz (one step every 0.1 seconds). So "2 seconds of history" translates to:
H=2 sec×10 Hz=20 time stepsH = 2 \text{ sec} \times 10 \text{ Hz} = 20 \text{ time steps}H=2 sec×10 Hz=20 time steps
Putting it together
With H=20H = 20
H=20 and at current time tt
t, the notation $s^i_{t-H:t}$ means:
st−20i,st−19i,…,st−1i,stis^i_{t-20}, s^i_{t-19}, \ldots, s^i_{t-1}, s^i_tst−20i​,st−19i​,…,st−1i​,sti​
— i.e., the last 20 (or 21, depending on whether the slice is inclusive on both ends) pose snapshots of agent ii
i, spanning the 2 seconds leading up to now.

## what is the H value
history value is set to 2 secs suppose. 



Agent and map encoders. The state s0 contains both map
and agent information. The map information mconsists of
Nm,1 polylines and Nm,2 polygons. The polylines describe
lane centers and lane boundaries, with each polyline con-
taining 3Np points, where 3 corresponds to the lane center,
the left boundary, and the right boundary. Each point is with
dimension Dm = 9 and includes the following attributes: x,
y, heading, speed limit, and category. When concatenated,
the points of the left and right boundaries together with
the center point yield a dimension of Nm,1 ×Np ×3Dm.
We leverage a PointNet [26] to extract features from the
points of each polyline, resulting in a dimensionality of
Nm,1 ×D, where Drepresents the feature dimension. The
polygons represent intersections, crosswalks, stop lines, etc,
with each polygon containing Np points. We utilize another
PointNet to extract features from the points of each poly-
gon, producing a dimension of Nm,2 ×D. We then con-
catenate the features from both polylines and polygons to
form the overall map features, resulting in a dimension of
Nm ×D. The agent information Aconsists of N agents,
where each agent maintains poses for the past Htime steps.
Each pose is with dimension Da = 10 and includes the
following attributes: x, y, heading, velocity, bounding box,
time step, and category. Consequently, the agent informa-
tion has a dimension of N ×H ×Da. We apply another
PointNet to extract features from the poses of each agent,
yielding an agent feature dimension of N ×D.