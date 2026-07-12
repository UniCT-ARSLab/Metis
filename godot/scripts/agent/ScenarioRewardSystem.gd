extends Node
class_name ScenarioRewardSystem

var _last_component_values := {}


func reset_rewards() -> void:
	_last_component_values.clear()
	for component in _scenario_components():
		if component.has_method("reset_rewards"):
			component.reset_rewards()


func reset_agent(agent_id:String, context:Dictionary = {}) -> void:
	_last_component_values[agent_id] = {}
	var agent_context := context.duplicate(true)
	agent_context["agent_id"] = agent_id
	for component in _scenario_components():
		if component.has_method("reset_agent"):
			component.reset_agent(agent_id, agent_context)


func compute_reward_scenario(agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", agent.name))
	var total := 0.0
	var values := {}
	for component in _scenario_components():
		var value := float(component.compute_reward(agent, context))
		total += value
		var term_name := str(component.get_term_name() if component.has_method("get_term_name") else component.name)
		values[term_name] = value
	_last_component_values[agent_id] = values
	return total


func get_terminal_reason(agent_id:String) -> String:
	for component in _scenario_components():
		if not component.has_method("get_terminal_reason"):
			continue
		var reason := str(component.get_terminal_reason(agent_id))
		if not reason.is_empty():
			return reason
	return ""


func is_agent_stalled(agent_id:String) -> bool:
	for component in _scenario_components():
		if component.has_method("is_agent_stalled") and bool(component.is_agent_stalled(agent_id)):
			return true
	return false


func get_agent_terms(agent_id:String) -> Dictionary:
	var result := {
		"component_values": Dictionary(_last_component_values.get(agent_id, {})).duplicate(true)
	}
	for component in _scenario_components():
		if not component.has_method("get_agent_terms"):
			continue
		var terms:Variant = component.get_agent_terms(agent_id)
		if typeof(terms) != TYPE_DICTIONARY or terms.is_empty():
			continue
		var term_name := str(component.get_term_name() if component.has_method("get_term_name") else component.name)
		result[term_name] = Dictionary(terms).duplicate(true)
	return result


func _scenario_components() -> Array:
	var result := []
	for child in get_children():
		if child.get("enabled") == false:
			continue
		if child.has_method("compute_reward"):
			result.append(child)
	return result
