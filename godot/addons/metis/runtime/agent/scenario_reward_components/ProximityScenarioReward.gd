extends ScenarioRewardComponent
class_name ProximityScenarioReward

## Dense per-step bonus for being close to the target, growing sharply in the final approach.
## The telescoping ProgressDelta reward has a vanishing gradient near the target (its delta goes
## to zero), so the arm has little incentive to close the last few centimeters. This term gives a
## continuous pull toward the EXACT target: it activates only once progress passes
## proximity_threshold (the final approach) and rises with an exponent so almost all of the bonus
## is concentrated right at the target.
@export var proximity_scale := 1.0
## Only reward the final approach; below this progress the term is zero (avoids paying the arm
## just for being roughly near the target).
@export var proximity_threshold := 0.8
## Exponent shaping the bonus curve; >1 concentrates the reward very close to the target.
@export var proximity_power := 2.0
# term_name is inherited from ScenarioRewardComponent (set it on the node in the scene); do NOT
# redefine it here or GDScript errors ("member already exists") and the whole component fails to load.

var _last_terms := {}


func reset_rewards() -> void:
	_last_terms.clear()


func reset_agent(agent_id:String, _context:Dictionary = {}) -> void:
	_last_terms[agent_id] = {}


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", ""))
	var progress := float(context.get("progress", context.get("track_progress", 0.0)))
	if progress <= proximity_threshold:
		_last_terms[agent_id] = {term_name: 0.0}
		return 0.0
	var span := maxf(1.0 - proximity_threshold, 0.001)
	var norm := clampf((progress - proximity_threshold) / span, 0.0, 1.0)
	var value := proximity_scale * pow(norm, proximity_power) * weight
	_last_terms[agent_id] = {term_name: value}
	return value


func get_agent_terms(agent_id:String) -> Dictionary:
	return Dictionary(_last_terms.get(agent_id, {})).duplicate(true)
