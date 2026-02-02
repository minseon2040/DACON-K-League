DROP_COLS = [
    "game_id", "period_id", "episode_id", "game_episode",
    "time_seconds", "is_train",
    "action_id", "team_id", "player_id",
    "venue", "game_day", "game_date", "home_team_name_ko",
    "away_team_name_ko", "is_home",
    "home_score", "away_score",
    "prev_start_x", "prev_start_y",
    "move_angle", "angle_to_opp_goal",
    "dist_to_opp_box_center",
    "action_type", "prev2_action_type",
]

FEATURE_CANDIDATES = [
    "match_minutes",
    "episode_event_index",
    "seq2_action_top",
    "start_x", "start_y", "legal_speed",
    "move_angle_sin", "move_angle_cos",
    "prev1_end_x", "prev1_end_y",
    "x_zone", "y_lane",
    "dist_to_opp_goal",
    "angle_to_opp_goal_sin", "angle_to_opp_goal_cos",
    "result_name",
    "prev1_action_type",
    "prev1_result_name",
    "player_role_label", "player_role_pass",
    "prev1_dx", "prev1_dy", "prev1_dist", "prev1_angle_sin", "prev1_angle_cos",
    "wing_prev_angle_sin", "wing_prev_angle_cos",
    "pass_dist_mean", "pass_dist_std",
    "forward_mean", "side_move_mean",
    "dx_mean", "dy_mean",
]

CAT_CANDIDATES = [
    "result_name",
    "player_role_label",
    "seq2_action_top",
    "prev1_action_type",
    "prev1_result_name",
    "x_zone", "y_lane",
    "player_role_pass",
]

SORT_KEYS = ["game_episode", "time_seconds", "action_id"]
TARGETS = ["end_x", "end_y"]