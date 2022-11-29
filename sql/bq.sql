EXPORT DATA
  OPTIONS(
    uri='gs://kaggle-yosuke/lgbm_dataset/20221129_2/train_*.parquet',
--     uri='gs://kaggle-yosuke/lgbm_dataset_test/lgbm_test_*.parquet',
    format='PARQUET',
    overwrite=true
  )
AS WITH session_stats AS (
    SELECT
        session,
        MIN(ts) AS ts_start,
        MAX(ts) AS ts_end,
        SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) AS clicks_cnt,
        SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) AS carts_cnt,
        SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) AS orders_cnt,
        COUNT(*) AS session_interaction_length
    FROM `kaggle-352109.otto.otto-validation-test`
--     FROM `kaggle-352109.otto.test`
    GROUP BY session
), aid_list AS (
    SELECT
        session,
        aid,
        type,
        ts,
        action_num_reverse_chrono,
        ts - ts_start AS sec_since_session_start,
        ts_end - ts AS sec_to_session_end,
        session_interaction_length,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        this_aid_clicks_cnt,
        this_aid_carts_cnt,
        this_aid_orders_cnt,
        CASE WHEN session_interaction_length = 1 THEN 1 ELSE POW(2, (0.1 + ((1 - 0.1) / (session_interaction_length - 1)) * (session_interaction_length - action_num_reverse_chrono - 1))) - 1 END AS log_recency_score
    FROM (
        SELECT
            t.session,
            ROW_NUMBER() OVER (PARTITION BY t.session ORDER BY t.ts DESC) - 1 AS action_num_reverse_chrono,
            t.aid,
            t.ts,
            ss.ts_start,
            ss.ts_end,
            t.type,
            ss.clicks_cnt,
            ss.carts_cnt,
            ss.orders_cnt,
            ss.session_interaction_length,
            SUM(CASE WHEN t.type = 'clicks' THEN 1 ELSE 0 END) OVER (PARTITION BY t.session, t.aid) AS this_aid_clicks_cnt,
            SUM(CASE WHEN t.type = 'carts' THEN 1 ELSE 0 END) OVER (PARTITION BY t.session, t.aid) AS this_aid_carts_cnt,
            SUM(CASE WHEN t.type = 'orders' THEN 1 ELSE 0 END) OVER (PARTITION BY t.session, t.aid) AS this_aid_orders_cnt
        FROM `kaggle-352109.otto.otto-validation-test` t
--         FROM `kaggle-352109.otto.test` t
        INNER JOIN session_stats ss ON ss.session = t.session
        ) t
    ORDER BY t.ts
), aid_list2 AS (
    SELECT
        session,
        aid,
        type,
        ts,
        action_num_reverse_chrono,
        sec_since_session_start,
        sec_to_session_end,
        session_interaction_length,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        this_aid_clicks_cnt,
        this_aid_carts_cnt,
        this_aid_orders_cnt,
        log_recency_score,
        (CASE WHEN type = 'clicks' THEN 1 WHEN type = 'carts' THEN 6 WHEN type = 'orders' THEN 3 END) * log_recency_score AS type_weighted_log_recency_score
    FROM aid_list
), aid_list3 AS (
    SELECT
        session,
        aid,
        session_interaction_length,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        this_aid_clicks_cnt,
        this_aid_carts_cnt,
        this_aid_orders_cnt,
        AVG(action_num_reverse_chrono) AS avg_action_num_reverse_chrono,
        MIN(action_num_reverse_chrono) AS min_action_num_reverse_chrono,
        MAX(action_num_reverse_chrono) AS max_action_num_reverse_chrono,
        AVG(sec_since_session_start) AS avg_sec_since_session_start,
        MIN(sec_since_session_start) AS min_sec_since_session_start,
        MAX(sec_since_session_start) AS max_sec_since_session_start,
        AVG(sec_to_session_end) AS avg_sec_to_session_end,
        MIN(sec_to_session_end) AS min_sec_to_session_end,
        MAX(sec_to_session_end) AS max_sec_to_session_end,
        AVG(log_recency_score) AS avg_log_recency_score,
        MIN(log_recency_score) AS min_log_recency_score,
        MAX(log_recency_score) AS max_log_recency_score,
        AVG(type_weighted_log_recency_score) AS avg_type_weighted_log_recency_score,
        MIN(type_weighted_log_recency_score) AS min_type_weighted_log_recency_score,
        MAX(type_weighted_log_recency_score) AS max_type_weighted_log_recency_score,
        NULL AS covisit_clicks_candidate_num,
        NULL AS covisit_carts_candidate_num,
        NULL AS covisit_orders_candidate_num,
        NULL AS w2v_candidate_num
    FROM aid_list2
    GROUP BY session, aid, session_interaction_length, clicks_cnt, carts_cnt, orders_cnt, this_aid_clicks_cnt, this_aid_carts_cnt, this_aid_orders_cnt
), covisit AS (
    SELECT
        session,
        aid,
        session_interaction_length,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        NULL AS this_aid_clicks_cnt,
        NULL AS this_aid_carts_cnt,
        NULL AS this_aid_orders_cnt,
        NULL AS avg_action_num_reverse_chrono,
        NULL AS min_action_num_reverse_chrono,
        NULL AS max_action_num_reverse_chrono,
        NULL AS avg_sec_since_session_start,
        NULL AS min_sec_since_session_start,
        NULL AS max_sec_since_session_start,
        NULL AS avg_sec_to_session_end,
        NULL AS min_sec_to_session_end,
        NULL AS max_sec_to_session_end,
        NULL AS avg_log_recency_score,
        NULL AS min_log_recency_score,
        NULL AS max_log_recency_score,
        NULL AS avg_type_weighted_log_recency_score,
        NULL AS min_type_weighted_log_recency_score,
        NULL AS max_type_weighted_log_recency_score,
        MAX(CASE WHEN type = 'clicks' THEN covisit_type_candidate_num ELSE NULL END) AS covisit_clicks_candidate_num,
        MAX(CASE WHEN type = 'carts' THEN covisit_type_candidate_num ELSE NULL END) AS covisit_carts_candidate_num,
        MAX(CASE WHEN type = 'orders' THEN covisit_type_candidate_num ELSE NULL END) AS covisit_orders_candidate_num,
        NULL AS w2v_candidate_num
    FROM (
        SELECT
            t.session,
            t.type,
            t.aid,
            ROW_NUMBER() OVER (PARTITION BY t.session, t.type) AS covisit_type_candidate_num,
            ss.clicks_cnt,
            ss.carts_cnt,
            ss.orders_cnt,
            ss.session_interaction_length
        FROM `kaggle-352109.otto.covisit` t
--         FROM `kaggle-352109.otto.covisit_test` t
        INNER JOIN session_stats ss ON ss.session = t.session
        WHERE t.aid is not NULL
    ) t
    GROUP BY session, aid, session_interaction_length, clicks_cnt, carts_cnt, orders_cnt
), w2v AS (
    SELECT
        session,
        aid,
        session_interaction_length,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        NULL AS this_aid_clicks_cnt,
        NULL AS this_aid_carts_cnt,
        NULL AS this_aid_orders_cnt,
        NULL AS avg_action_num_reverse_chrono,
        NULL AS min_action_num_reverse_chrono,
        NULL AS max_action_num_reverse_chrono,
        NULL AS avg_sec_since_session_start,
        NULL AS min_sec_since_session_start,
        NULL AS max_sec_since_session_start,
        NULL AS avg_sec_to_session_end,
        NULL AS min_sec_to_session_end,
        NULL AS max_sec_to_session_end,
        NULL AS avg_log_recency_score,
        NULL AS min_log_recency_score,
        NULL AS max_log_recency_score,
        NULL AS avg_type_weighted_log_recency_score,
        NULL AS min_type_weighted_log_recency_score,
        NULL AS max_type_weighted_log_recency_score,
        NULL AS covisit_clicks_candidate_num,
        NULL AS covisit_carts_candidate_num,
        NULL AS covisit_orders_candidate_num,
        MAX(w2v_candidate_num) AS w2v_candidate_num
    FROM (
        SELECT
            t.session,
            t.aid,
            ROW_NUMBER() OVER (PARTITION BY t.session) AS w2v_candidate_num,
            ss.clicks_cnt,
            ss.carts_cnt,
            ss.orders_cnt,
            ss.session_interaction_length
        FROM `kaggle-352109.otto.w2v` t
--         FROM `kaggle-352109.otto.covisit_test` t
        INNER JOIN session_stats ss ON ss.session = t.session
        WHERE t.aid is not NULL
    ) t
    GROUP BY session, aid, session_interaction_length, clicks_cnt, carts_cnt, orders_cnt

), union_all AS (
    SELECT * FROM aid_list3 al
    UNION ALL
    SELECT * FROM covisit
    UNION ALL
    SELECT * FROM w2v
)

SELECT
    session,
    aid,
    MAX(session_interaction_length) AS session_interaction_length,
    COUNT(*) OVER (PARTITION BY session) AS session_length,
    MAX(clicks_cnt) AS clicks_cnt,
    MAX(carts_cnt) AS carts_cnt,
    MAX(orders_cnt) AS orders_cnt,
    MAX(this_aid_clicks_cnt) AS this_aid_clicks_cnt,
    MAX(this_aid_carts_cnt) AS this_aid_carts_cnt,
    MAX(this_aid_orders_cnt) AS this_aid_orders_cnt,
    MAX(avg_action_num_reverse_chrono) AS avg_action_num_reverse_chrono,
    MAX(min_action_num_reverse_chrono) AS min_action_num_reverse_chrono,
    MAX(max_action_num_reverse_chrono) AS max_action_num_reverse_chrono,
    MAX(avg_sec_since_session_start) AS avg_sec_since_session_start,
    MAX(min_sec_since_session_start) AS min_sec_since_session_start,
    MAX(max_sec_since_session_start) AS max_sec_since_session_start,
    MAX(avg_sec_to_session_end) AS avg_sec_to_session_end,
    MAX(min_sec_to_session_end) AS min_sec_to_session_end,
    MAX(max_sec_to_session_end) AS max_sec_to_session_end,
    MAX(avg_log_recency_score) AS avg_log_recency_score,
    MAX(min_log_recency_score) AS min_log_recency_score,
    MAX(max_log_recency_score) AS max_log_recency_score,
    MAX(avg_type_weighted_log_recency_score) AS avg_type_weighted_log_recency_score,
    MAX(min_type_weighted_log_recency_score) AS min_type_weighted_log_recency_score,
    MAX(max_type_weighted_log_recency_score) AS max_type_weighted_log_recency_score,
    MAX(covisit_clicks_candidate_num) AS covisit_clicks_candidate_num,
    MAX(covisit_carts_candidate_num) AS covisit_carts_candidate_num,
    MAX(covisit_orders_candidate_num) AS covisit_orders_candidate_num,
    MAX(w2v_candidate_num) AS w2v_candidate_num
FROM union_all
GROUP BY session, aid
ORDER BY session, aid;
