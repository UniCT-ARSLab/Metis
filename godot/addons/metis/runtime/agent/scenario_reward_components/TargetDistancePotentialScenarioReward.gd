extends ScenarioRewardComponent
class_name TargetDistancePotentialScenarioReward

## Potential-based final-approach shaping for tasks that expose a target distance.
##
## The exponential potential has its strongest gradient near the target, where a linear
## distance delta may be too weak. Only changes in potential are rewarded, so an agent
## cannot collect this term by stopping near the goal.
@export var distance_method: StringName = &"get_target_distance"
## Characteristic distance of the exponential potential, in the same unit returned by
## distance_method. With metres, 0.06 focuses the signal on roughly the final 10-15 cm.
@export_range(0.001, 10.0, 0.001) var potential_distance := 0.06
## Maximum cumulative approach shaping for a monotonic move from far away to distance zero.
@export var approach_reward_scale := 5.0
## Makes moving away slightly more expensive than the equivalent approach, preventing a
## discounted learner from profiting from repeated approach/retreat oscillations.
@export_range(1.0, 10.0, 0.05) var retreat_penalty_multiplier := 1.25

var _previous_potential := {}
var _last_terms := {}


func reset_rewards() -> void:
	_previous_potential.clear()
	_last_terms.clear()


func reset_agent(agent_id:String, _context:Dictionary = {}) -> void:
	_previous_potential.erase(agent_id)
	_last_terms[agent_id] = {}


func compute_reward(agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", agent.name if agent else ""))
	if not agent or not agent.has_method(distance_method):
		_last_terms[agent_id] = {"available": false}
		return 0.0

	var distance := maxf(float(agent.call(distance_method)), 0.0)
	if not is_finite(distance):
		_last_terms[agent_id] = {"available": false}
		return 0.0

	var potential := exp(-distance / maxf(potential_distance, 0.000001))
	if not _previous_potential.has(agent_id):
		_previous_potential[agent_id] = potential
		_last_terms[agent_id] = {
			"available": true,
			"distance": distance,
			"potential": potential,
			"potential_delta": 0.0,
			"reward": 0.0,
		}
		return 0.0

	var previous := float(_previous_potential[agent_id])
	var potential_delta := potential - previous
	_previous_potential[agent_id] = potential
	var scale := approach_reward_scale
	if potential_delta < 0.0:
		scale *= retreat_penalty_multiplier
	var value := potential_delta * scale * weight
	_last_terms[agent_id] = {
		"available": true,
		"distance": distance,
		"potential": potential,
		"previous_potential": previous,
		"potential_delta": potential_delta,
		"reward": value,
	}
	return value


func get_agent_terms(agent_id:String) -> Dictionary:
	return Dictionary(_last_terms.get(agent_id, {})).duplicate(true)
