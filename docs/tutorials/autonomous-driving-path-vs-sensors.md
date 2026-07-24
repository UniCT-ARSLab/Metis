# Autonomous driving with and without route knowledge

Metis includes two versions of the same driving task:

- **sensor-only**: the policy sees local raycasts, speed, and previous controls;
- **path-aware**: the policy also sees its progress, lateral offset, heading, and local
  look-ahead directions along a `Path3D`.

Both use the same car body, physics, continuous actions, rewards, and track. This makes
the comparison meaningful: only the information available to the policy changes.

The ready-to-run scenes are:

```text
res://scenarios/cars/cars_scenario.tscn             sensor-only, obs_dim=14
res://scenarios/cars/cars_path_aware_scenario.tscn  path-aware, obs_dim=24
```

The shared agent is [`car.tscn`](../../godot/agents/Car/car.tscn), backed by
[`car.gd`](../../godot/agents/Car/car.gd).

## 1. Separate training state from policy input

A `Path3D` can have two independent jobs.

**Training-only use**

`Path3DProgressProvider` measures progress for:

- forward and backward progress rewards;
- stall and pace detection;
- reset positions along the track;
- curriculum and metrics;
- finish validation.

None of this implies that the neural network knows the route. The provider writes to
the scenario context, not the observation vector.

**Policy input**

`Path3DNavigationObservationSource` adds route information to the observation vector.
The resulting model requires a route and localization estimate at inference time.

This distinction is important. You can train a map-agnostic controller with a path-based
reward, provided that path data never enters its observations.

## 2. Understand the limits of sensor-only driving

The sensor-only version learns local track following: stay in free space, anticipate
curves from range sensors, and avoid walls. It can generalize to unseen continuous
tracks when the training set contains enough geometric variety.

It cannot infer an unobservable destination at a fork. If left and right are equally
clear and no route cue is present, the two choices are identical from the policy's
point of view. A road network needs at least one of:

- a local navigation command such as left, straight, or right;
- a relative goal direction;
- a map and localization;
- observable history paired with a recurrent policy.

The included example therefore uses an unbranched circuit.

## 3. Define the continuous actions

The car exposes two scalar actions:

```text
move_input:     Continuous(1), range [-1, 1]
rotation_input: Continuous(1), range [-1, 1]
```

Positive `move_input` accelerates. Negative input brakes by interpolating velocity
towards zero; it does not immediately apply reverse thrust. This avoids the common
early-training loop where a policy alternates forward and reverse to remain safe.

The current scene narrows random exploration through the action nodes:

```text
MoveInput.exploration_low = 0.35
RotationInput.exploration_low = -0.35
RotationInput.exploration_high = 0.35
```

These are exploration bounds, not policy output bounds. The trained actor can still
use the full declared range.

## 4. Build the sensor-only observation vector

The base car provides:

| Observation | Size | Source |
|---|---:|---|
| signed forward speed and absolute speed | 2 | `BodySpeedObservationSource` |
| forward clearance | 1 | `RaycastClearanceObservationSource` |
| previous throttle and steering | 2 | two `MethodObservationSource` nodes |
| nine ray distances | 9 | `RaycastObservationSource` |

Total: 14 values.

Every distance is normalized to `[0, 1]`, where `1` means no hit within the ray range.
Speeds are divided by the configured `speed_scale`. Previous controls make acceleration
and steering changes easier to interpret from one state to the next.

The sensor fan covers the front and both sides at several angles. Use rays long enough
to see a curve before the vehicle reaches it, but keep their range realistic for the
eventual sensor setup.

## 5. Add path-aware observations

The same `car.tscn` contains a disabled `PathNavigation` source. The inherited
`cars_path_aware_scenario.tscn` only sets:

```text
Car.path_aware_observations = true
```

That enables ten more values:

| Observation | Size | Meaning |
|---|---:|---|
| `path_progress` | 1 | closest offset divided by path length |
| `path_lateral_offset` | 1 | signed cross-track error |
| `path_heading` | 2 | path tangent in the car's local frame |
| three `path_lookahead_*` values | 6 | normalized local direction to future samples |

Total: 24 values.

The look-ahead distances default to 5, 12, and 25 world units. Tune them to vehicle
speed and curve radius. Values that are all too close duplicate the front raycasts;
values that are all too far can hide the immediate steering correction.

## 6. Keep vehicle physics in the body

The car script remains ordinary Godot game code. `Agent` handles the RL contract, while
`Car` handles acceleration, friction, steering authority, gravity, and collision state.

The core movement is:

```gdscript
var forward_speed := absf(get_signed_forward_speed())
var steering_authority := clampf(
	forward_speed / maxf(min_speed_for_steering * 4.0, 0.001),
	0.0,
	1.0
)

rotate_y(-_rotate_input * steering_authority * deg_to_rad(steering_speed_degrees) * delta)
```

Steering authority falls near zero speed, so the car cannot spin unrealistically while
stationary. If you change this physics, start a new experiment: the same policy output
will produce a different transition.

Collision detection examines every `move_and_slide()` contact. A body in
`car_crash_obstacle`, or a contact whose normal is not floor-like, marks the car as
crashed and emits the `collision` reward event.

## 7. Configure rewards

The car's local `RewardSystem` contains:

| Term | Current value | Purpose |
|---|---:|---|
| forward velocity | `0.02 * signed_speed / 15` | reward useful speed, penalize backward motion |
| time | `-0.001` per step | make slow driving costly |
| action smoothness | `-0.01` scale | discourage throttle and steering oscillation |
| collision | `-20.0` | make crashes clearly undesirable |

The scenario then adds route-level rewards:

| Term | Current value | Purpose |
|---|---:|---|
| progress delta | `100 * delta_progress` | reward movement along the route |
| backward progress | `100 * negative_delta` | penalize moving backwards on the route |
| finish | `+50` | terminal task success |
| progress stall | `-8` after 150 steps without 0.005 progress | terminate inactive agents |
| pace | `-2` when 0.01 progress takes over 130 steps | discourage excessively slow travel |

The velocity term encourages motion now; progress rewards movement in the correct route
direction. They solve different problems. Using only speed can reward circles or motion
away from the finish, while using only progress often creates timid stop-and-go driving.

Avoid making clear-road acceleration a large direct reward. It can teach the actor to
hold full throttle even when the sensor pattern already predicts a curve. Forward speed,
progress, collision cost, and pace usually provide a cleaner balance.

## 8. Build the scenario

The relevant part of `cars_scenario.tscn` is:

```text
CarsScenario
├── BridgeServer
├── ScenarioController
│   ├── ProgressProvider (Path3DProgressProvider)
│   ├── ScenarioEventSystem
│   │   └── FinishReached (AreaScenarioEventSource)
│   └── ScenarioRewardSystem
│       ├── ProgressReward
│       ├── FinishReward
│       ├── ProgressStallPenalty
│       └── ProgressPacePenalty
├── Environment
│   └── EndRace (Area3D)
├── Agents
│   └── Car
└── RoadGenerator (Path3D, group: navigation_path)
```

The path should follow the drivable center line. It does not need to match the road mesh
at every vertex, but it must be monotonic and close enough that nearest-point projection
does not jump between distant track sections.

`EndRace` emits `finish_reached`, which terminates the agent and awards the finish
reward. Keep its collision mask separate from wall collision masks.

For a one-lap task, make the final point of the progress path coincide with the center
of `EndRace`. A road mesh may close its geometry independently, but the progress
provider only measures the points contained in its `Curve3D`. If the finish remains on
an omitted closing segment, progress reaches `1.0` before the car reaches the finish
area and stall detection can terminate an otherwise valid run.

## 9. Use reset curriculum without leaking the route

`Path3DProgressProvider.build_reset_transform()` can place a car anywhere in a progress
interval and align it with the path tangent. The trainer changes `reset_progress_max`
over time with:

```text
--reset-progress-curriculum
--reset-progress-start-max 0.02
--reset-progress-end-max 0.60
--reset-progress-ramp-episodes 1600
```

Early episodes start near the beginning. As training advances, some episodes begin
farther along the track, exposing later curves before the policy can complete the whole
route.

This does not make the sensor-only policy memorize progress because the sampled value
is not observed. The car still has to react to local geometry. Keep a nonzero share of
starts near zero so completion of the full track remains part of the training
distribution.

## 10. Validate both contracts

Sensor-only:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --steps 100 \
  --print-reward-terms \
  --no-headless
```

Expect `obs_shape=(14,)` and no `path_*` observation keys.

Path-aware:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_path_aware_scenario.tscn \
  --steps 100 \
  --print-reward-terms \
  --no-headless
```

Expect `obs_shape=(24,)`. Also check that progress increases in the intended driving
direction, rays report walls, a wall contact terminates promptly, and `EndRace` produces
exactly one finish event.

## 11. Train the sensor-only policy

SAC is a good baseline for the two continuous controls:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --num-episodes 3000 \
  --max-steps-per-episode 4500 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.25 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.02 \
  --reset-progress-end-max 0.60 \
  --reset-progress-ramp-episodes 1600 \
  --checkpoint-dir checkpoints/driving_sensor_only_sac_v1 \
  --collector-mode async \
  --headless
```

Do not add `--multi-agent` while the scene still contains one car. `--num-envs 4`
already collects four independent trajectories. Section 13 shows how to add several
non-colliding cars to each environment after the single-car contract is validated.

## 12. Train the path-aware policy

Use a separate checkpoint directory because the input dimension differs:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_path_aware_scenario.tscn \
  --num-envs 4 \
  --num-episodes 3000 \
  --max-steps-per-episode 4500 \
  --batch-size 128 \
  --replay-warmup 12000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.25 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.02 \
  --reset-progress-end-max 0.60 \
  --reset-progress-ramp-episodes 1600 \
  --checkpoint-dir checkpoints/driving_path_aware_sac_v1 \
  --collector-mode async \
  --headless
```

The path-aware model should usually learn faster. That is expected: it receives a local
route plan in addition to obstacle sensing.

### Optional SB3 comparison

Metis is the default for both driving experiments. A single-car SAC baseline can also
be trained through the limited SB3 adapter:

```bash
python/.venv-sb3/bin/python python/train.py \
  --backend sb3 \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --total-timesteps 1000000 \
  --max-steps-per-episode 4500 \
  --batch-size 128 \
  --learning-starts 12000 \
  --buffer-size 200000 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.02 \
  --reset-progress-end-max 0.60 \
  --reset-progress-ramp-episodes 1600 \
  --collector-mode sync \
  --evaluation-episodes 40 \
  --checkpoint-dir checkpoints/driving_sensor_only_sb3_sac_v1 \
  --headless
```

Swap the scene and checkpoint directory to compare the path-aware contract. Do not
reuse models between the two observation spaces.

The replicated-car setup in the next section is not a drop-in SB3 comparison when one
car can crash while the others continue. SB3 vector lanes expect automatic per-lane
reset, while all cars share one physical Godot world. The adapter rejects that partial
termination by default. `--sb3-multi-agent-partial-done reset-all` is available for
experiments, but it truncates every surviving car and changes the task semantics.

## 13. Collect with several cars per environment

Multi-environment and multi-agent collection solve different scaling problems:

- `--num-envs 4` starts four Godot processes;
- six cars in each process produce up to 24 agent transitions per simulation step;
- `--multi-agent` tells Python to keep those transitions separate while training one
  shared policy.

Metis does not select the best car and it does not create one model per replica. Every
car uses the same actor, and every learner-controlled car contributes observations,
actions, rewards, and terminal states to the same replay buffer. This is parameter
sharing and is appropriate because all cars expose the same contract.

First validate the task with one car. Then select `ScenarioController` in the Cars
scene and configure:

```text
agent_to_replicate       = ../Agents/Car
agents_container         = ../Agents
number_of_replications   = 5
```

`number_of_replications` counts additional copies, so this example creates six cars in
total. Each replica receives a stable ID and an agent-specific reset seed. The supplied
`Car` is on collision layer 1 and uses mask 6, which excludes layer 1; replicas therefore
collide with the road and finish area but not with one another. Keep that arrangement
for parallel experience collection. Enabling car-to-car contact changes the task into
traffic interaction and also requires observations describing nearby vehicles.

Check the resulting contract before training:

```bash
python/.venv/bin/python python/tools/random_rollout.py \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --multi-agent \
  --steps 300 \
  --no-headless
```

The spec should report six distinct agent IDs. Their initial positions should differ,
and one crashed car should become inactive without ending the other five trajectories.

Train the shared sensor-only policy with:

```bash
python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --num-envs 4 \
  --num-episodes 3000 \
  --max-steps-per-episode 4500 \
  --batch-size 128 \
  --replay-warmup 30000 \
  --random-exploration-episodes 40 \
  --action-smoothing 0.25 \
  --reset-progress-curriculum \
  --reset-progress-start-max 0.02 \
  --reset-progress-end-max 0.60 \
  --reset-progress-ramp-episodes 1600 \
  --collector-mode async \
  --async-update-basis transitions \
  --async-max-updates-per-env-step 2 \
  --checkpoint-dir checkpoints/driving_sensor_only_multi_agent_sac_v1 \
  --multi-agent \
  --headless
```

Transition-based update scheduling is already the Metis default; it is written here to
make the multi-agent behavior explicit. Watch `transitions_step`, `updates_s`,
`updates_throttled`, queue saturation, CPU usage, and GPU usage. Increase replicas only
while environment throughput and learner updates remain healthy. More correlated cars
inside one world do not replace independent Godot processes, so a mix such as four
environments with six cars is usually preferable to one environment with 24 cars.

Use a new checkpoint directory when switching from single-agent collection. The policy
shape is unchanged, but a clean comparison makes it much easier to tell whether extra
cars improved sample efficiency or merely increased correlated replay data.

The same setup works with `cars_path_aware_scenario.tscn`: replicate its `Car`, keep
`--multi-agent`, and use a separate path-aware checkpoint directory. Do not mix its
replay or policy with the sensor-only version because the observation vectors differ.

## 14. Train across several tracks

Generalization requires a distribution of geometry. Create each layout as a separate
scene and instantiate one active layout per episode. Pair it with the matching
`Curve3D`, road collision, finish area, and valid spawn region.

Do not merely hide inactive tracks. Hidden physics bodies can still collide. Remove the
inactive layout from the scene tree or explicitly disable all of its collision shapes.

Use the seeded episode reset to select layouts. Introduce simple curves first, then more
varied radii, road widths, and combinations. Hold out at least one full layout for
evaluation.

For the sensor-only policy, route diversity is the main defense against track
memorization. For the path-aware policy, it tests whether look-ahead geometry is used
as intended rather than as a proxy for one fixed track.

## 15. Evaluate the comparison

Use deterministic actions and the same seeds for both policies. Record:

- finish and collision rates;
- mean completion time;
- mean and maximum progress;
- speed by track segment;
- steering magnitude and action delta;
- results per layout, including unseen layouts.

Add two stress tests:

1. change the track while preserving the sensor setup;
2. add a temporary obstacle that is absent from the route model.

The path-aware controller should plan curves earlier. The sensor-only controller may be
more robust to route-model errors, provided it has seen enough varied geometry.

## 16. Run the policies

These commands load native Metis/Keras policies. SB3 `.zip` models are evaluated by
the SB3 training backend or by the backend benchmark tool, not by `python/run.py`.

Sensor-only:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/driving_sensor_only_sac_v1 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_scenario.tscn \
  --episodes 20 \
  --execution-mode realtime \
  --no-headless
```

Path-aware:

```bash
python/.venv/bin/python python/run.py \
  --policy-path checkpoints/driving_path_aware_sac_v1 \
  --godot-bin /path/to/Godot \
  --godot-project godot \
  --godot-scene res://scenarios/cars/cars_path_aware_scenario.tscn \
  --episodes 20 \
  --execution-mode realtime \
  --no-headless
```

The policies are not interchangeable because their input dimensions differ.

## 17. Choose the right design

Use the path-aware version when the final system has a route and reliable localization.
It is appropriate for a known circuit, warehouse route, or autonomous vehicle with a
planner.

Use the sensor-only version when the task is local corridor or track following and the
deployed system has only local ranging. It is not a complete navigation system.

A practical middle ground is often best: a conventional planner emits a small local
command such as left, straight, or right, while the learned controller handles speed,
steering, and collision avoidance. That keeps global routing explicit without handing
the entire map to the policy.
