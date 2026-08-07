extends RewardComponent
class_name StepPenaltyReward

@export_range(-1.0, 0.0, 0.000001, "or_less") var penalty := -0.001


func compute_reward(_context:Dictionary) -> float:
	return penalty * weight
