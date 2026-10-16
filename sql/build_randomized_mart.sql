SELECT
    u.user_id,
    u.assignment_date,
    a.dataset_type,
    a.treatment,
    u.device_type,
    u.region,
    u.is_new_user,
    u.pre_7d_sessions,
    u.pre_7d_watch_time_minutes,
    u.pre_7d_click_days,
    u.pre_experiment_activity_score,
    o.clicked,
    o.watch_time_minutes,
    o.next_day_retention,
    o.negative_feedback
FROM read_parquet($users_path) AS u
INNER JOIN read_parquet($assignment_path) AS a
    ON u.user_id = a.user_id
INNER JOIN read_parquet($outcomes_path) AS o
    ON u.user_id = o.user_id
WHERE a.dataset_type = 'randomized'
ORDER BY u.user_id
