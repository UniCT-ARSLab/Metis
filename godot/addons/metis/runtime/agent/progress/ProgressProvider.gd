extends Metis
class_name ProgressProvider

@export var clamp_to_unit_range := true


func get_progress(agent:Node, context:Dictionary = {}) -> float:
	var value := float(measure_progress(agent, context))
	return clampf(value, 0.0, 1.0) if clamp_to_unit_range else value


func measure_progress(_agent:Node, _context:Dictionary = {}) -> float:
	return 0.0


func reset_provider() -> void:
	pass


func reset_agent(_agent:Node, _context:Dictionary = {}) -> void:
	pass
