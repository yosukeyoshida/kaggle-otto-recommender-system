EXPORT DATA
  OPTIONS(
--     uri='gs://kaggle-yosuke/lgbm_dataset/20221204_2/train_*.parquet', -- FIXME
    uri='gs://kaggle-yosuke/lgbm_dataset_test/20221204_2/lgbm_test_*.parquet',
    format='PARQUET',
    overwrite=true
  )
AS
WITH aid_list AS (
    SELECT
          session,
          action_num_reverse_chrono,
          aid,
          -- ts,
          type,
          sec_since_session_start,
          sec_to_session_end,
          session_interaction_length,
          log_recency_score,
          (CASE WHEN type = 'clicks' THEN 1 WHEN type = 'carts' THEN 6 WHEN type = 'orders' THEN 3 END) * log_recency_score AS type_weighted_log_recency_score
    FROM (
      SELECT
          session,
          action_num_reverse_chrono,
          aid,
          ts,
          type,
          -- prev_dt,
          -- dt,
          -- DATETIME_DIFF(dt, prev_dt, MINUTE) AS minutes_from_last_interaction,
          sec_since_session_start,
          sec_to_session_end,
          session_interaction_length,
          CASE WHEN session_interaction_length = 1 THEN 1 ELSE POW(2, (0.1 + ((1 - 0.1) / (session_interaction_length - 1)) * (session_interaction_length - action_num_reverse_chrono - 1))) - 1 END AS log_recency_score,
      FROM (
        SELECT
          session,
          ROW_NUMBER() OVER (PARTITION BY session ORDER BY ts DESC) - 1 AS action_num_reverse_chrono,
          aid,
          ts,
          type,
          CAST(FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', TIMESTAMP_SECONDS(CAST((LAG(ts, 1) OVER (PARTITION BY session ORDER BY ts))/1000 AS int))) as DATETIME) AS prev_dt,
          CAST(FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', TIMESTAMP_SECONDS(CAST(ts/1000 AS int))) as DATETIME) AS dt,
          ts - (MIN(ts) OVER (PARTITION BY session)) AS sec_since_session_start,
          (MAX(ts) OVER (PARTITION BY session)) - ts AS sec_to_session_end,
          COUNT(*) OVER (PARTITION BY session) AS session_interaction_length
--         FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
        FROM `kaggle-352109.otto.test`
      )
    )
), aggregate_by_session_aid AS (
    SELECT
        session,
        aid,
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
        SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) AS session_aid_clicks_cnt,
        SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) AS session_aid_carts_cnt,
        SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) AS session_aid_orders_cnt,
        NULL AS covisit_clicks_candidate_num,
        NULL AS covisit_carts_candidate_num,
        NULL AS covisit_orders_candidate_num,
        NULL AS w2v_candidate_num
    FROM aid_list
    GROUP BY session, aid
), covisit AS (
    SELECT
        session,
        aid,
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
        NULL AS session_aid_clicks_cnt,
        NULL AS session_aid_carts_cnt,
        NULL AS session_aid_orders_cnt,
        MAX(CASE WHEN type = 'clicks' THEN rank ELSE NULL END) AS covisit_clicks_candidate_num,
        MAX(CASE WHEN type = 'carts' THEN rank ELSE NULL END) AS covisit_carts_candidate_num,
        MAX(CASE WHEN type = 'orders' THEN rank ELSE NULL END) AS covisit_orders_candidate_num,
        NULL AS w2v_candidate_num
--     FROM `kaggle-352109.otto.covisit` -- FIXME
    FROM `kaggle-352109.otto.covisit_test`
    WHERE aid is not NULL
    GROUP BY session, aid
), w2v AS (
    SELECT
        session,
        aid,
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
        NULL AS session_aid_clicks_cnt,
        NULL AS session_aid_carts_cnt,
        NULL AS session_aid_orders_cnt,
        NULL AS covisit_clicks_candidate_num,
        NULL AS covisit_carts_candidate_num,
        NULL AS covisit_orders_candidate_num,
        rank AS w2v_candidate_num
--     FROM `kaggle-352109.otto.w2v` -- FIXME
    FROM `kaggle-352109.otto.w2v_test`
    WHERE aid is not NULL
), union_all AS (
    SELECT
        session,
        aid,
        MAX(session_aid_clicks_cnt) AS session_aid_clicks_cnt,
        MAX(session_aid_carts_cnt) AS session_aid_carts_cnt,
        MAX(session_aid_orders_cnt) AS session_aid_orders_cnt,
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
        MAX(w2v_candidate_num) AS w2v_candidate_num,
    FROM (
        SELECT * FROM aggregate_by_session_aid
        UNION ALL
        SELECT * FROM covisit
        UNION ALL
        SELECT * FROM w2v
    ) t
    GROUP BY session, aid
), session_stats AS (
    SELECT
        session,
        SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) AS session_clicks_cnt,
        SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) AS session_carts_cnt,
        SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) AS session_orders_cnt,
        COUNT(*) AS session_interaction_length
--     FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
    FROM `kaggle-352109.otto.test`
    GROUP BY session
), aid_stats AS (
    SELECT
        aid,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        RANK() OVER (ORDER BY clicks_cnt DESC) AS clicks_rank,
        RANK() OVER (ORDER BY carts_cnt DESC) AS carts_rank,
        RANK() OVER (ORDER BY orders_cnt DESC) AS orders_rank
    FROM (
        SELECT
            aid,
            SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) AS clicks_cnt,
            SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) AS carts_cnt,
            SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) AS orders_cnt
--         FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
        FROM `kaggle-352109.otto.test`
        GROUP BY aid
    ) t
)

SELECT
    sa.session,
    sa.aid,
    ss.session_interaction_length,
    ss.session_clicks_cnt,
    ss.session_carts_cnt,
    ss.session_orders_cnt,
    COALESCE(sa.session_aid_clicks_cnt, 0) AS session_aid_clicks_cnt,
    COALESCE(sa.session_aid_carts_cnt, 0) AS session_aid_carts_cnt,
    COALESCE(sa.session_aid_orders_cnt, 0) AS session_aid_orders_cnt,
    sa.avg_action_num_reverse_chrono,
    sa.min_action_num_reverse_chrono,
    sa.max_action_num_reverse_chrono,
    sa.avg_sec_since_session_start,
    sa.min_sec_since_session_start,
    sa.max_sec_since_session_start,
    sa.avg_sec_to_session_end,
    sa.min_sec_to_session_end,
    sa.max_sec_to_session_end,
    sa.avg_log_recency_score,
    sa.min_log_recency_score,
    sa.max_log_recency_score,
    sa.avg_type_weighted_log_recency_score,
    sa.min_type_weighted_log_recency_score,
    sa.max_type_weighted_log_recency_score,
    sa.covisit_clicks_candidate_num,
    sa.covisit_carts_candidate_num,
    sa.covisit_orders_candidate_num,
    sa.w2v_candidate_num,
    COALESCE(ais.clicks_rank, 1000000) AS clicks_rank,
    COALESCE(ais.carts_rank, 1000000) AS carts_rank,
    COALESCE(ais.orders_rank, 1000000) AS orders_rank,
    COALESCE(ais.clicks_cnt, 0) AS clicks_cnt,
    COALESCE(ais.carts_cnt, 0) AS carts_cnt,
    COALESCE(ais.orders_cnt, 0) AS orders_cnt,
FROM union_all sa
LEFT JOIN session_stats ss ON ss.session = sa.session
LEFT JOIN aid_stats ais ON ais.aid = sa.aid
