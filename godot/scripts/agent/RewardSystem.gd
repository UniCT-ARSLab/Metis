extends Node
class_name RewardSystem

@export var enabled := true

var _event_terms := {}
var _last_terms := {}


func reset_reward(context:Dictionary = {}) -> void:
	_event_terms.clear()
	_last_terms.clear()

	for child in get_children():
		if child.has_method("reset_reward"):
			child.reset_reward(context)


func add_event(term:String, value:float) -> void:
	_event_terms[term] = float(_event_terms.get(term, 0.0)) + value


func get_reward(context:Dictionary = {}) -> float:
	_last_terms.clear()

	if not enabled:
		return 0.0

	var reward_context: Dictionary = context.duplicate(true)
	reward_context["reward_system"] = self
	reward_context["events"] = _event_terms.duplicate(true)

	var total := 0.0
	for child in get_children():
		if child.get("enabled") == false:
			continue
		if not child.has_method("compute_reward"):
			continue

		var value := float(child.compute_reward(reward_context))
		var term := str(child.name)
		if child.has_method("get_term_name"):
			term = str(child.get_term_name())

		total += _add_term(term, value)

	_event_terms.clear()
	return total


func get_last_terms() -> Dictionary:
	return _last_terms.duplicate(true)


func _add_term(term:String, value:float) -> float:
	if value == 0.0:
		return 0.0

	_last_terms[term] = float(_last_terms.get(term, 0.0)) + value
	return value
