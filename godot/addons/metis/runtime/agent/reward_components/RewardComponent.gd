extends Metis
class_name RewardComponent

@export var enabled := true
@export var term_name := ""
@export var weight := 1.0


func get_term_name() -> String:
	if term_name.is_empty():
		return name
	return term_name


func reset_reward(_context:Dictionary = {}) -> void:
	pass


func compute_reward(_context:Dictionary) -> float:
	return 0.0
