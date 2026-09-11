extends Metis
class_name ScenarioEventSystem


func reset_events() -> void:
	for source in _event_sources():
		if source.has_method("reset_events"):
			source.reset_events()


func reset_agent(agent:Node, context:Dictionary = {}) -> void:
	for source in _event_sources():
		if source.has_method("reset_agent"):
			source.reset_agent(agent, context)


func update_agent(agent:Node, context:Dictionary = {}) -> void:
	for source in _event_sources():
		if source.has_method("update_agent"):
			source.update_agent(agent, context)


func get_agent_context(agent_id:String) -> Dictionary:
	var result := {}
	for source in _event_sources():
		if source.has_method("get_agent_events"):
			var events:Variant = source.get_agent_events(agent_id)
			if typeof(events) == TYPE_DICTIONARY:
				result.merge(events, true)
	return result


func get_terminal_reason(agent_id:String) -> String:
	for source in _event_sources():
		if not source.has_method("get_agent_terminal_reason"):
			continue
		var reason := str(source.get_agent_terminal_reason(agent_id))
		if not reason.is_empty():
			return reason
	return ""


func _event_sources() -> Array:
	var result := []
	for child in get_children():
		if child.get("enabled") == false:
			continue
		if child.has_method("get_agent_events"):
			result.append(child)
	return result
