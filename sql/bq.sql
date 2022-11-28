WITH aid_list AS (
    SELECT
        session,
        aid,
        type,
        ts,
        action_num_reverse_chrono,
        ts - ts_start AS sec_since_session_start,
        ts_end - ts AS sec_to_session_end,
        session_length,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        this_aid_clicks_cnt,
        this_aid_carts_cnt,
        this_aid_orders_cnt,
        POW(2, (0.1 + ((1 - 0.1) / (session_length - 1)) * (session_length - action_num_reverse_chrono - 1))) - 1 AS log_recency_score
    FROM (
        SELECT
        session,
        ROW_NUMBER() OVER (PARTITION BY session ORDER BY ts DESC) - 1 AS action_num_reverse_chrono,
        aid,
        ts,
        MIN(ts) OVER (PARTITION BY session) AS ts_start,
        MAX(ts) OVER (PARTITION BY session) AS ts_end,
        -- CAST(FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', TIMESTAMP_MILLIS(ts)) as DATETIME) AS dt,
        type,
        SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) OVER (PARTITION BY session) AS clicks_cnt,
        SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) OVER (PARTITION BY session) AS carts_cnt,
        SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) OVER (PARTITION BY session) AS orders_cnt,
        SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) OVER (PARTITION BY session, aid) AS this_aid_clicks_cnt,
        SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) OVER (PARTITION BY session, aid) AS this_aid_carts_cnt,
        SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) OVER (PARTITION BY session, aid) AS this_aid_orders_cnt,
        COUNT(*) OVER (PARTITION BY session) AS session_length
        FROM `kaggle-352109.otto.train_sample`
        ) t
    ORDER BY ts
)

SELECT
    session,
    aid,
    type,
    ts,
    action_num_reverse_chrono,
    sec_since_session_start,
    sec_to_session_end,
    session_length,
    clicks_cnt,
    carts_cnt,
    orders_cnt,
    this_aid_clicks_cnt,
    this_aid_carts_cnt,
    this_aid_orders_cnt,
    log_recency_score,
    (CASE WHEN type = 'clicks' THEN 1 WHEN type = 'carts' THEN 6 WHEN type = 'orders' THEN 3 END) * log_recency_score AS type_weighted_log_recency_score
FROM aid_list
ORDER BY ts;