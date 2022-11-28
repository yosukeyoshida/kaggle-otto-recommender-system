SELECT
  session,
  action_num_reverse_chrono,
  aid,
  ts,
  ts_start,
  ts_end,
  ts - ts_start AS sec_since_session_start,
  ts_end - ts AS sec_to_session_end,
  type,
  MAX(action_num_reverse_chrono) OVER (PARTITION BY session) AS session_length
FROM (
  SELECT
    session,
    ROW_NUMBER() OVER (PARTITION BY session ORDER BY ts DESC) AS action_num_reverse_chrono,
    aid,
    ts,
    MIN(ts) OVER (PARTITION BY session) AS ts_start,
    MAX(ts) OVER (PARTITION BY session) AS ts_end,
    -- CAST(FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', TIMESTAMP_MILLIS(ts)) as DATETIME) AS dt,
    type
  FROM `kaggle-352109.otto.train_sample`
  ORDER BY ts
) t
