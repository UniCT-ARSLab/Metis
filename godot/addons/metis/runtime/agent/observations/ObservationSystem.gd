extends Node
class_name ObservationSystem


func register_observations(agent:Agent, body:Node) -> void:
	for child in get_children():
		if child.get("enabled") == false:
			continue
		if child.has_method("register_observations"):
			child.register_observations(agent, body)


func reset_sources() -> void:
	for child in get_children():
		if child.has_method("reset_source"):
			child.reset_source()


func refresh_sources() -> void:
	for child in get_children():
		if child.has_method("refresh_source"):
			child.refresh_source()
