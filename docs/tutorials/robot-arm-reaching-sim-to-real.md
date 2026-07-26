# Robot-arm reaching, grasping, and sim-to-real

This guide builds a robot-arm task in a calibrated workcell. The arm must reach a
target, avoid the environment and itself, and optionally grasp and lift a physical
object. The policy does not rely on proximity sensors that the real robot lacks.

The repository includes a working URDF-based XArm grasping example:

- [`urdf_robot_arm_agent.gd`](../../godot/agents/RobotArms/urdf_robot_arm_agent.gd)
- [`x_arm_agent.tscn`](../../godot/agents/RobotArms/xarm/x_arm_agent.tscn)
- [`XarmScenario.tscn`](../../godot/scenarios/robotarms/XarmScenario.tscn)
- [`xarm_scenario.gd`](../../godot/scenarios/robotarms/xarm_scenario.gd)

The same Metis contract can wrap a three-joint teaching arm or an eight-joint redundant
arm. The number of controlled joints is discovered from the robot adapter rather than
hard-coded in Python.

## 1. Start with the safety boundary

A fixed, measured workcell can be learned without proximity observations because joint
state and target pose determine the robot's geometry relative to static obstacles. The
network can encode that geometry in its weights.

That assumption stops being true when a person, tool, or obstacle moves without being
represented in the observation. Two visually different workcells would then produce the
same state vector but require different actions.

For deployment, use all three layers:

1. a calibrated digital twin for training and collision queries;
2. a learned policy for motion decisions;
3. an independent supervisor for joint limits, velocity, acceleration, workspace,
   predicted collision, communication timeout, and emergency stop.

RL is not the sole safety system. A sensorless installation must be physically
controlled or interlocked so unexpected obstacles cannot enter the workcell.

## 2. Choose what the policy controls

The practical choices are:

| Policy output | Strength | Cost |
|---|---|---|
| joint torque | full dynamic control | hardest to model and transfer safely |
| joint velocity | direct, expressive, easy to limit | action size grows with the arm |
| Cartesian TCP velocity | small and intuitive action space | an IK solver chooses joint motion |

The included adapter uses normalized joint velocities:

```text
action = [dq_0, dq_1, ..., dq_(N-1)] in [-1, 1]
```

Each component is multiplied by the corresponding URDF velocity limit, falling back to
`default_joint_speed` when the URDF has no usable value. This is a good match for a real
robot that already has low-level joint servos.

Bounded revolute joints are also protected at the position level. Kinematic integration
is clamped to the URDF lower and upper limits, while commands that push farther into a
limit are tapered over the final part of the joint range. Keep this supervisor active in
deployment; a reward penalty is not a mechanical safety limit.

The IK variant later in this guide keeps a three-value Cartesian action and runs the IK
outside the network.

## 3. Build a measurable digital twin

Do not tune geometry by eye. Record:

- joint order, axis, sign, zero position, and limits;
- link transforms and conservative collision shapes;
- maximum velocity and acceleration;
- the base-to-world transform;
- the TCP offset from the final link;
- table, floor, fixtures, and obstacle dimensions;
- controller period and measured command latency.

Use simple collision geometry for queries and detailed meshes only for rendering. A
slightly inflated collision shape is usually more useful than a visually exact mesh
that is expensive or non-conservative.

Before training, compare at least ten known joint configurations against the real
robot's forward kinematics. A wrong joint sign can still produce a policy that appears
to learn in simulation and fails immediately on hardware.

## 4. Import URDF with `GodotRobot`

The project contains the `godot_urdf` add-on. Import the `.urdf` into the Godot project,
wait for the generated resource, and instantiate it below a `Node3D` agent root:

```text
RobotArm (Node3D) [URDFRobotArmAgentBody]
├── Agent
│   ├── ActionSpace
│   ├── ObservationSystem
│   └── RewardSystem
├── ImportedRobot (GodotRobot)
└── EndEffector (Marker3D)
```

Set `robot_path` to the imported `GodotRobot`. Set `tcp_link_name` to the URDF link that
carries the tool and use `tcp_local_offset` for the actual tool center point.

The adapter supports two runtime modes:

- `KINEMATIC`, useful for fast and deterministic position integration;
- `PHYSICS_MOTORS`, useful when motor and rigid-body behavior matter.

Begin with kinematic control to validate transforms, action order, rewards, and
collisions. Physics motors introduce more parameters and should be a deliberate later
experiment.

### Mimic joints

The bundled URDF extension parses `<mimic joint="..." multiplier="..." offset="...">`.
A mimic joint follows its source and is excluded from the independent actuator list.
For a coupled gripper, the policy controls the source finger once while the importer
synchronizes the followers.

The tag usually does not change the resting model visually. Its effect becomes visible
when the source joint moves. Validate parsing and runtime synchronization with:

```bash
/path/to/Godot --headless --path godot \
  --script res://tests/test_urdf_mimic.gd
```

If the importer reports a mimic cycle or unknown source, fix the URDF. Do not add both
the source and its followers to `controlled_joint_names`.

### The bundled XArm wrist

The physical XArm model is a 5-DOF arm plus gripper. Its last arm rotation is
`xarm_2_joint`. The following `wrist_roll` joint only mounts `hand_link`, so it is
declared `fixed` in the bundled URDF. Making both joints revolute creates two
co-located rotations around the same local axis: the policy can command them in
opposite directions without producing useful TCP motion.

The fixed mount includes a `+90 deg` yaw offset around the wrist axis. This describes
the physical orientation of the gripper bracket; it is not a sixth action and does not
shift the position, zero, or limits of `xarm_2_joint`.

The default XArm reaching scene therefore controls these five joints, in order:

```text
xarm_6_joint
xarm_5_joint
xarm_4_joint
xarm_3_joint
xarm_2_joint
```

`grip_left` is a separate actuator for tasks that explicitly train the gripper; its
mimic followers must never be added as policy actions.

## 5. Use a `Skeleton3D` when appropriate

A robot imported as a skinned mesh can use a `Skeleton3D` instead of a chain of
`GodotRobot` link bodies. Metis does not require one representation, but the current
`URDFRobotArmAgentBody` specifically expects `GodotRobot`.

For a skeleton-backed robot, write a small body adapter that presents the same public
contract:

```text
get_joint_count()
get_controlled_joint_names()
get_joint_position_observation()
get_joint_velocity_observation()
get_target_error_observation()
get_previous_action_observation()
apply_action(action)
reset_all(original_transform, reset_rewards)
is_terminal()
get_progress()
```

Map each controlled URDF joint to one bone, store its axis and limits, and update bone
poses in a fixed order. Collision objects still need to follow the bones and expose link
identity for self-collision filtering.

Do not let an animation player, IK modifier, and RL adapter write the same bones at the
same time. One system must own the final pose each physics tick.

## 6. Make the contract scale with joint count

If `controlled_joint_names` is empty, the URDF adapter uses every independent actuated
joint. Otherwise it keeps the listed joints in exactly that order. The action node is
then resized automatically when `auto_configure_action_size` is enabled.

For a reaching task with joint velocity control, use:

```text
joint_positions    N
joint_velocities   N
target_error       3
previous_action    N
--------------------
obs_dim          3N+3
action_size         N
```

For pose tracking, add the three-value target orientation error used by the bundled
XArm scene:

```text
target_orientation_error   3
----------------------------
pose_obs_dim             3N+6
```

Examples:

| Controlled joints | Action size | Position-only observations | Pose observations |
|---:|---:|---:|---:|
| 3 | 3 | 12 | 15 |
| 5 | 5 | 18 | 21 |
| 7 | 7 | 24 | 27 |
| 8 | 8 | 27 | 30 |

Joint positions are normalized from URDF limits to `[-1, 1]`; velocities are divided
by their joint speed limits. `target_error` is expressed in the robot base frame and
divided by `workspace_scale`.

The same order must be used by Godot, the demonstration dataset, exported policy
manifest, and real controller.

## 7. Configure the Metis agent

The basic reaching tree is:

```text
Agent
├── ActionSpace
│   └── JointVelocity (ContinuousAction, size set automatically)
├── ObservationSystem
│   ├── JointPositions (MethodObservationSource)
│   ├── JointVelocities (MethodObservationSource)
│   ├── TargetError (MethodObservationSource)
│   └── PreviousAction (MethodObservationSource)
└── RewardSystem
    ├── JointMotion (FunctionRewardComponent)
    ├── JointLimit (FunctionRewardComponent)
    ├── Smoothness (ActionSmoothnessPenaltyReward)
    └── Time (StepPenaltyReward)
```

Wire the method sources to the methods with the same names in
`urdf_robot_arm_agent.gd`. For `Smoothness`, list one input per controlled action:

```text
joint_velocity_0
joint_velocity_1
...
joint_velocity_(N-1)
```

The local terms should remain small:

- joint motion penalizes unnecessary total command effort;
- joint limit penalizes the outer ten percent of revolute travel;
- smoothness penalizes abrupt command changes;
- time makes shorter successful trajectories preferable.

Task outcome and collision rewards belong to the scenario.

## 8. Build a reaching scenario

A minimal scene is:

```text
RobotArmReachingScenario
├── BridgeServer
├── ScenarioController
│   ├── ProgressProvider (MethodProgressProvider -> arm.get_progress)
│   ├── ScenarioEventSystem
│   │   ├── GoalReached
│   │   ├── Collision
│   │   └── SelfCollision
│   └── ScenarioRewardSystem
│       ├── Progress
│       ├── NoProgress
│       ├── GoalReward
│       ├── CollisionPenalty
│       └── SelfCollisionPenalty
├── Environment
├── RobotArm
├── Target (Marker3D or Node3D)
└── Spawns
    ├── EasyTarget1
    ├── EasyTarget2
    └── ...
```

Connect the body's signals to manual scenario events:

```gdscript
arm.target_reached.connect(
	func(): goal_event.trigger(str(arm.name)))
arm.obstacle_collision.connect(
	func(): collision_event.trigger(str(arm.name)))
arm.self_collision.connect(
	func(): self_collision_event.trigger(str(arm.name)))
```

`get_progress()` returns one minus normalized TCP-to-target distance. A
`ProgressDeltaScenarioReward` then rewards actual improvement rather than mere
proximity. Require the TCP to remain within `success_distance` for several physics
frames before emitting success; this prevents high-speed fly-throughs.

Suggested outcome scale:

| Event | Reward | Terminal |
|---|---:|---|
| target reached | `+30` to `+40` | yes |
| environment collision | `-20` | yes |
| self collision | `-30` | yes |
| no progress | `-5` | yes |

The exact scale is task-dependent. Inspect individual terms before changing them.

## 9. Train the included XArm grasp task

The supplied XArm scene extends reaching into assisted grasping. Its target is a
`RigidBody3D` with a child `GraspPoint`:

```text
Target (RigidBody3D, group: graspable)
├── Mesh
├── CollisionShape3D
└── GraspPoint (Marker3D)
```

`RobotArm.target` points to `GraspPoint`, while `grasp_target_body` points to `Target`.
The target is intentionally absent from `robot_obstacle`; finger contact must be legal.

The XArm URDF exposes six independent actuators: five arm joints and the `grip_left`
source joint. Mimic joints drive the remaining gripper mechanism.

Its observation vector is:

```text
6 joint positions
6 joint velocities
3 target error values
6 previous actions
6 target linear/angular velocity values
3 grasp-state values
------------------
30 observations
```

Grasp state contains normalized gripper closure, whether the object is attached, and
normalized lift height.

With `assisted_grasp=true`, the object is captured only when:

- the TCP is within `grasp_capture_distance`;
- gripper closure exceeds `grasp_close_threshold`;
- object speed is below `max_grasp_target_speed`.

The relative TCP-to-object transform is then held kinematically. Success requires the
object to remain above `required_lift_height` for `grasp_hold_physics_frames`. Assisted
grasp is useful for learning approach, closure, and lift sequencing. It is not a model
of frictional grasp stability.

Current scenario rewards are:

| Term | Value |
|---|---:|
| target progress | dense progress, stronger backward penalty |
| object grasped | `+8` |
| lifted successfully | `+40` |
| object dropped or pushed away before grasp | `-20` |
| environment collision | `-20` |
| self collision | `-30` |
| progress stall | `-5` |

Local terms also penalize target disturbance before capture and closing the gripper
while it is still far away.

## 10. Detect environment and self collisions

`URDFRobotArmAgentBody` caches collision shapes from each imported link and queries the
physics space every tick.

For the environment:

- add fixed obstacles to `robot_obstacle`;
- include their layers in `environment_collision_mask`;
- add the floor to `robot_obstacle` as well;
- if the base is allowed to touch the floor, also add the floor to
  `robot_support_surface` and list only the base link in `support_contact_link_names`.

The XArm scene permits floor contact only for `xarm_6_link`. Any other link touching the
floor is an environment collision.

For self collision:

- keep `auto_detect_self_collisions=true`;
- normally keep `ignore_adjacent_link_collisions=true` because connected link shapes
  often overlap around a joint;
- list additional verified pairs as `link_a:link_b`;
- list tightly coupled mechanisms as comma-separated sets when every internal pair is
  expected to overlap.

The XArm example ignores `xarm_3_link:hand_link` and internal gripper link pairs because
their imported collision volumes overlap in mechanically valid poses. Treat every
exception as calibration data. An ignore list should document an expected mechanical
contact, not hide a poor collision model.

Use `get_last_collision_info()` while debugging to see whether the event came from an
environment pair or a self-collision pair.

## 11. Reset the rigid target atomically

The scenario discovers every `Marker3D` below `target_spawns_root`. Adding another
marker automatically adds a candidate spawn.

In the reaching example, those markers delimit a training volume. After the introductory
curriculum, `continuous_target_sampling` draws coordinates throughout the axis-aligned
volume instead of selecting only exact marker positions. This reduces memorization and
makes held-out positions inside the declared workspace meaningful. Moving the target
outside that volume is still out-of-distribution and must be covered by additional
markers or a deliberately expanded workspace.

At reset it:

1. detaches any assisted grasp;
2. freezes the target body;
3. clears linear and angular velocity;
4. applies the selected marker transform;
5. resets physics interpolation;
6. stores the new spawn transform for lift and disturbance checks;
7. keeps the target frozen while the arm returns to its home pose;
8. samples joint-home offsets, excluding the gripper;
9. unfreezes and wakes the target after `episode_reset_completed`.

Moving a live `RigidBody3D` without clearing velocity or freezing it can make the old
physics state reappear on the next tick. Unfreezing it before the kinematic arm has
finished resetting can also create a collision before the policy's first action. Keep
this reset sequence intact.

The built-in curriculum is:

| Episode | Target set | Joint jitter | Capture distance | Lift height |
|---:|---|---:|---:|---:|
| `<3000` | first easy markers | `0 deg` | `0.060` | `0.020` |
| `3000-5999` | first two easy groups | `0 deg` | `0.050` | `0.030` |
| `6000-8999` | all markers | `2 deg` | `0.040` | `0.040` |
| `9000+` | all markers | `5 deg` | `0.032` | `0.050` |

The early stages also allow a lower lift hold time, a slightly faster object at capture,
and more pre-grasp displacement. These tolerances tighten at each transition. Stage
boundaries are exported by `xarm_scenario.gd`, so a harder robot can remain in an early
stage longer without changing Python.

Keep a separate set of target poses out of training for evaluation.

## 12. Validate before training

Run the included XArm scene:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --steps 400 \
  --print-reward-terms \
  --no-headless
```

Check the contract and mechanics:

1. `action_type=continuous`, `action_size=7`, and `obs_dim=33`.
2. Every action index moves the intended independent joint with the intended sign.
3. Mimic fingers follow their source but are not separate actions.
4. A non-base link touching the floor emits `collision`.
5. Two non-ignored links touching emit `self_collision`.
6. The resting base and internal gripper do not produce false positives.
7. Pushing the object away before grasp emits `object_dropped`.
8. Proximity alone is not success; the object must be grasped and lifted.
9. Equal reset seeds select equal target poses and joint jitter.
10. Reset leaves no stale velocity, contact, or reward state.

Also test constant motion on one joint at a time. This catches most ordering and sign
errors before the learner begins compensating for them.

## 13. Train a SAC baseline

Start the included grasping task from a fresh directory:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 4 \
  --num-episodes 12000 \
  --max-steps-per-episode 500 \
  --batch-size 128 \
  --replay-warmup 15000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.10 \
  --collector-mode async \
  --checkpoint-dir checkpoints/xarm_grasp_sac_v2 \
  --headless
```

Do not add `--multi-agent`: each environment contains one arm. Four environments already
collect four independent trajectories.

Monitor more than reward:

- grasp and lift success rates;
- environment and self-collision rates;
- object-drop rate;
- progress and episode length;
- joint-limit and action-smoothness terms;
- success on held-out target markers.

A policy that approaches the target but never closes the gripper can accumulate dense
progress without solving the task.

### Optional SB3 comparison

The same single-arm continuous contract can be used for a controlled SB3 SAC baseline:

```bash
python/.venv-sb3/bin/python python/train.py \
  --backend sb3 \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 4 \
  --total-timesteps 1000000 \
  --max-steps-per-episode 500 \
  --batch-size 128 \
  --learning-starts 15000 \
  --buffer-size 500000 \
  --collector-mode sync \
  --evaluation-episodes 50 \
  --checkpoint-dir checkpoints/xarm_grasp_sb3_sac_v1 \
  --headless
```

SB3 is included here for comparison, not as a replacement for the native workflow. It
does not use the Metis async collector and does not support the demonstration-aware
`td3_bc`, `ddpg_bc`, or `ddpgfd` trainers described below. Keep the physics, seeds,
transition budget, and evaluation poses equal when comparing results.

## 14. Validate a target pose manually

Before changing a reward or starting another long run, place the arm in a pose that you
consider correct. This is a direct way to verify the TCP frame, target frame, joint
limits, success thresholds, and the numbers reported to Python.

Run the scenario from the Godot editor, or start it directly:

```bash
/path/to/Godot \
  --path godot \
  res://scenarios/robotarms/XarmScenario.tscn
```

The URDF arm adapter keeps manual control disabled during normal training. Press `F2`
in the running scenario to take control:

| Key | Action |
| --- | --- |
| `F2` | Enable or disable manual control |
| `1` to `5` | Select a controlled XArm joint by its action-space index |
| `Q` / `E` | Move the selected joint in the negative or positive direction |
| `Shift` + `Q` / `E` | Move at the fine-adjustment speed |
| `Space` | Stop every joint immediately |
| `R` | Clear a collision stop and continue from the current pose |
| `Home` | Restore the configured home joint positions |
| `P` | Print the complete pose diagnostics to the Godot output |
| `F9` | Save a screenshot and matching JSON diagnostics |
| `H` | Print the controls again |

The overlay shows the selected URDF joint, every joint angle, TCP-to-target distance,
orientation error, and maximum joint speed. `manual_command_scale` and
`manual_fine_scale` are exported by `URDFRobotArmAgentBody`, so reduce them in the
Inspector when validating a real robot's final alignment.

A detected collision still stops the arm. Press `R`, then immediately hold the
appropriate `Q` or `E` direction to move away from the contact. Metis temporarily
suppresses collision termination for `manual_recovery_grace_physics_frames` while
keeping the URDF joint limits active. If the geometry remains trapped or the correct
escape direction is unclear, press `Home` instead.

Use `P` when the hand looks correct. The log is enclosed by
`[METIS_ARM_REFERENCE_BEGIN]` and `[METIS_ARM_REFERENCE_END]` and contains:

- joint positions in radians and degrees, velocities, commands, and URDF limits;
- world-space TCP and target positions, quaternions, and local `+X`, `+Y`, `+Z` axes;
- position error in world and robot-base coordinates;
- shortest axis-angle orientation error;
- raw pose, hold, motion, and joint-limit reward terms;
- the active distance, orientation, speed, and hold thresholds.

`F9` writes the same data beside a PNG under
`user://manual_arm_captures`. Godot prints both absolute paths. Keeping the image and
JSON together makes it possible to answer two different questions: whether the pose
looks mechanically correct, and whether Metis describes that pose as correct.

For a valid reference pose, expect all of the following:

1. the TCP marker is at the intended grasp point;
2. the logged tool and target axes express the intended approach direction;
3. `position_error_m` and `orientation_error_deg` are below the active success limits;
4. releasing the keys brings `max_joint_speed_rad_s` below its success limit;
5. no joint lies outside the limits recorded in the JSON.

If the pose looks right but the orientation error is close to `90` or `180` degrees,
calibrate `EndEffector/ToolPose`; do not teach the policy to compensate for an incorrect
frame. If the numbers look right but the mesh looks wrong, inspect the URDF visual
origins and the imported skeleton or link transforms.

The selected-joint controls coexist with the older `joint_0_negative`,
`joint_0_positive`, and similar Input Map actions. The latter remain useful for
dedicated control panels and custom demonstration devices.

## 15. Use planners and demonstrations

A static known workcell is also a motion-planning problem. Use a collision-aware planner
as a baseline and as a source of demonstrations. For each trajectory:

1. plan joint configurations `q[0..T]` with the same URDF and obstacle geometry;
2. time-parameterize them within real velocity and acceleration limits;
3. replay them in Godot at the policy control rate;
4. record the action actually applied and the observations produced by Metis;
5. retain both successful and safe suboptimal examples.

Manual demonstrations can be recorded with:

```bash
python/.venv/bin/python python/recorder.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --output demos/xarm_grasp_v1.npz \
  --agent-id RobotArm \
  --episodes 50 \
  --max-steps 500 \
  --step-delay 0.05 \
  --no-headless
```

For keyboard recording, select an XArm joint with `1` to `5` and command it with `Q`
or `E`.
The adapter's `apply_manual_action()` returns the normalized vector actually applied,
so the recorder stores numeric actions rather than the special `"manual"` request.
Per-joint actions named `joint_0_negative`, `joint_0_positive`, and so on remain
supported when several joints must be mapped to a custom controller.

Once the dataset covers varied targets and joint configurations, train TD3+BC:

```bash
python/.venv/bin/python python/train.py \
  --algorithm td3_bc \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 4 \
  --num-episodes 12000 \
  --max-steps-per-episode 500 \
  --batch-size 256 \
  --replay-warmup 5000 \
  --demo-path demos/xarm_grasp_v1.npz \
  --demo-bc-epochs 20 \
  --demo-bc-weight-start 1.0 \
  --demo-bc-weight-end 0.05 \
  --demo-bc-decay-updates 150000 \
  --checkpoint-dir checkpoints/xarm_grasp_td3_bc_v1 \
  --collector-mode async \
  --headless
```

The dataset's observation order and action size must match the live scene exactly. See
[Manual demonstrations](../guides/manual-demonstrations.md) for validation and resume
behavior.

## 16. Cartesian control with IK

IK can make reaching easier by reducing the policy output to:

```text
tcp_velocity = [vx, vy, vz] in the robot base frame
```

The IK adapter converts that velocity into joint commands while preserving the same
collision, limit, reset, and reward logic. For positional differential IK, a damped
least-squares update is:

```text
dq = J^T (J J^T + lambda^2 I)^-1 v_tcp
```

For each revolute joint, the positional Jacobian column is:

```text
J_i = axis_world_i x (tcp_world - joint_origin_world_i)
```

The matrix inverted here is `3 x 3`, independent of joint count. Damping limits
instability near singularities, but it does not solve collision avoidance or guarantee
a preferred elbow configuration.

An IK body adapter should:

1. expose one `ContinuousAction` named `tcp_velocity` with size 3;
2. convert the base-frame command to world space;
3. solve for `dq` and clamp every joint speed and position;
4. update the joint-state observations from the applied result;
5. expose the previous Cartesian action, not the internal `dq`;
6. optionally report IK residual and saturation as diagnostic terms.

Its observation size becomes:

```text
N joint positions + N joint velocities + 3 target error + 3 previous action
obs_dim = 2N + 6
```

| Controlled joints | IK action size | IK observation size |
|---:|---:|---:|
| 3 | 3 | 12 |
| 7 | 3 | 20 |
| 8 | 3 | 22 |

Godot skeleton IK modifiers can be wrapped in the same contract, or you can use a
robotics solver such as the controller vendor's IK, KDL, Pinocchio, or MoveIt. Keep the
solver deterministic and consistent between simulation and deployment. A different
solver can choose a different elbow posture for the same TCP command.

IK is a new action and observation contract. Use a new checkpoint and new
demonstrations; do not resume a joint-velocity model.

For grasping, position-only IK may be insufficient because the gripper orientation
matters. Add orientation error and angular TCP commands only after the positional task
is stable, then increase the action and observation dimensions explicitly.

## 17. Run and inspect the policy

Normal evaluation resets after every terminal outcome:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/xarm_grasp_sac_v2 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --episodes 50 \
  --max-steps 500 \
  --execution-mode realtime \
  --no-headless
```

For a reaching variant where you want to drag the target by hand and keep inference
running, the body and scenario must support `continue_after_success`. Then use:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/robot_arm_reaching_v1 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/RobotArmReachingScenario.tscn \
  --continue-after-success \
  --infinite \
  --no-time-limit \
  --no-initial-reset \
  --no-reset \
  --execution-mode realtime \
  --no-headless
```

`--no-reset` preserves the physical state after logical terminal events. It does not
make an unsafe collision valid. A real controller should stop and require an explicit
safety recovery after contact.

## 18. Randomize only measured uncertainty

After the nominal task works, vary:

- joint zero offsets within calibration error;
- actuator gain, speed, deadband, and command latency;
- encoder noise and delay;
- base and TCP calibration within measured tolerance;
- target pose;
- object mass, damping, and friction within plausible ranges;
- control period within the observed jitter.

Do not randomize obstacle positions beyond what the observation can identify. If the
workcell has several known configurations, include a configuration descriptor or
geometry query in the observation. Otherwise the policy sees identical inputs for
states that require different safe paths.

## 19. Export and deploy

The trained Keras model maps a numeric vector to normalized actions. It does not contain
the Godot sensors, joint ordering, frame transforms, scaling, IK solver, or safety
supervisor. Deploy `policy.json` with the model and implement the same preprocessing in
the target runtime.

Export TFLite with:

```bash
python/.venv/bin/python python/export.py \
  --policy checkpoints/xarm_grasp_td3_bc_v1 \
  --format tflite \
  --tflite-quantization float16
```

The hardware loop is:

```text
encoders and target pose
-> normalize and order observations exactly as policy.json declares
-> Keras or TFLite inference
-> scale joint velocities, or run the validated IK adapter
-> independent safety supervisor
-> low-level robot controller
```

Move to hardware in stages:

1. shadow mode, where actions are logged but not sent;
2. an empty workcell at low speed with an operator and emergency stop;
3. soft obstacles and conservative margins;
4. the calibrated task after repeated fixed-seed and held-out validation.

Promote a policy using separate thresholds for success, collision, minimum clearance,
joint-limit margin, and execution time. Mean reward alone is not a deployment metric.

## 20. When Godot is the wrong physics engine

Godot is a useful environment for joint-position or velocity tasks, known geometry,
and arcade or assisted grasp logic. Consider MuJoCo, robosuite, or another robotics
simulator when the result depends on:

- torque control;
- accurate motor dynamics;
- repeated contact and force sensing;
- frictional grasp stability;
- deformation or detailed material interaction.

Metis can still organize the policy contract and training workflow, but the simulator
must be credible for the behavior being transferred.

## 21. Recommended order of work

1. Import one robot and verify joint order, sign, limits, and mimic behavior.
2. Calibrate base, TCP, collision shapes, and the fixed workcell.
3. Test every joint manually at reduced speed.
4. Validate environment, floor, and self-collision reporting.
5. Build a conventional IK or motion-planning baseline.
6. Freeze the policy action and observation contract.
7. Record or generate varied demonstrations.
8. Run a short SAC baseline to validate reward and termination.
9. Train TD3+BC and compare it against the planner and SAC.
10. Add curriculum and measured domain randomization gradually.
11. Evaluate held-out targets and perturbed calibration.
12. Enter shadow mode before commanding real motion.

This order is slower than pressing Train immediately, but much faster than diagnosing a
policy that learned around a wrong joint axis or an invisible collision.
