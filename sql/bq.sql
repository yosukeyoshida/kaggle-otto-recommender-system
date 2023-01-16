EXPORT DATA
  OPTIONS(
    uri='gs://kaggle-yosuke/lgbm_dataset/20230117/train_*.parquet', -- FIXME
--     uri='gs://kaggle-yosuke/lgbm_dataset_test/20230117/test_*.parquet',
    format='PARQUET',
    overwrite=true
  )
AS
WITH day_num AS (
  SELECT
      dt,
      ROW_NUMBER() OVER (ORDER BY dt) AS num
  FROM (
      SELECT
          CAST(FORMAT_TIMESTAMP('%Y-%m-%d', TIMESTAMP_SECONDS(CAST(ts/1000 AS int))) as DATE) AS dt
      FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
--       FROM `kaggle-352109.otto.test`
      GROUP BY dt
  )
), session_stats1 AS (
    SELECT
        session,
        min_dt,
        max_dt,
        session_clicks_cnt,
        session_carts_cnt,
        session_orders_cnt,
        session_clicks_unique_aid,
        session_carts_unique_aid,
        session_orders_unique_aid,
        CASE WHEN session_clicks_unique_aid = 0 THEN NULL ELSE session_carts_unique_aid / session_clicks_unique_aid END AS session_clicks_carts_ratio,
        CASE WHEN session_carts_unique_aid = 0 THEN NULL ELSE session_orders_unique_aid / session_carts_unique_aid END AS session_carts_orders_ratio,
        CASE WHEN session_clicks_unique_aid = 0 THEN NULL ELSE session_orders_unique_aid / session_clicks_unique_aid END AS session_clicks_orders_ratio
    FROM (
        SELECT
            session,
            MIN(CAST(FORMAT_TIMESTAMP('%Y-%m-%d', TIMESTAMP_SECONDS(CAST(ts/1000 AS int))) as DATE)) AS min_dt,
            MAX(CAST(FORMAT_TIMESTAMP('%Y-%m-%d', TIMESTAMP_SECONDS(CAST(ts/1000 AS int))) as DATE)) AS max_dt,
            SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) AS session_clicks_cnt,
            SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) AS session_carts_cnt,
            SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) AS session_orders_cnt,
            COUNT(DISTINCT(CASE WHEN type = 'clicks' THEN aid ELSE NULL END)) AS session_clicks_unique_aid,
            COUNT(DISTINCT(CASE WHEN type = 'carts' THEN aid ELSE NULL END)) AS session_carts_unique_aid,
            COUNT(DISTINCT(CASE WHEN type = 'orders' THEN aid ELSE NULL END)) AS session_orders_unique_aid
        FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
--         FROM `kaggle-352109.otto.test`
        GROUP BY session
    )
), session_stats2 AS (
    SELECT
        session,
        AVG(sec_clicks_carts) AS avg_sec_session_clicks_carts,
        MIN(sec_clicks_carts) AS min_sec_session_clicks_carts,
        MAX(sec_clicks_carts) AS max_sec_session_clicks_carts,
        AVG(sec_carts_orders) AS avg_sec_session_carts_orders,
        MIN(sec_carts_orders) AS min_sec_session_carts_orders,
        MAX(sec_carts_orders) AS max_sec_session_carts_orders
    FROM (
        SELECT
            session,
            aid,
            CASE WHEN first_carts_ts - first_clicks_ts > 0 THEN first_carts_ts - first_clicks_ts ELSE NULL END AS sec_clicks_carts,
            CASE WHEN first_orders_ts - first_carts_ts > 0 THEN first_orders_ts - first_carts_ts ELSE NULL END AS sec_carts_orders
        FROM (
            SELECT
                session,
                aid,
                MIN(CASE WHEN type = 'clicks' THEN ts ELSE NULL END) AS first_clicks_ts,
                MIN(CASE WHEN type = 'carts' THEN ts ELSE NULL END) AS first_carts_ts,
                MIN(CASE WHEN type = 'orders' THEN ts ELSE NULL END) AS first_orders_ts
            FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
--             FROM `kaggle-352109.otto.test`
            GROUP BY session, aid
        ) t
    ) t
    GROUP BY session
), session_stats AS (
    SELECT
        s1.session,
        s1.session_clicks_cnt,
        s1.session_carts_cnt,
        s1.session_orders_cnt,
        s1.session_clicks_unique_aid,
        s1.session_carts_unique_aid,
        s1.session_orders_unique_aid,
        s1.session_clicks_carts_ratio,
        s1.session_carts_orders_ratio,
        s1.session_clicks_orders_ratio,
        s2.avg_sec_session_clicks_carts,
        s2.min_sec_session_clicks_carts,
        s2.max_sec_session_clicks_carts,
        s2.avg_sec_session_carts_orders,
        s2.min_sec_session_carts_orders,
        s2.max_sec_session_carts_orders,
        d1.num AS min_day_num,
        d2.num AS max_day_num
    FROM session_stats1 s1
    INNER JOIN session_stats2 s2 ON s2.session = s1.session
    INNER JOIN day_num d1 ON d1.dt = s1.min_dt
    INNER JOIN day_num d2 ON d2.dt = s1.max_dt
), daily_rank AS (
    SELECT
        t.dt,
        d.num AS day_num,
        t.aid,
        clicks_cnt,
        carts_cnt,
        orders_cnt,
        RANK() OVER (PARTITION BY t.dt ORDER BY t.clicks_cnt DESC) AS clicks_rank,
        RANK() OVER (PARTITION BY t.dt ORDER BY t.carts_cnt DESC) AS carts_rank,
        RANK() OVER (PARTITION BY t.dt ORDER BY t.orders_cnt DESC) AS orders_rank,
--         RANK() OVER (PARTITION BY t.dt ORDER BY t.clicks_uu DESC) AS clicks_uu_rank,
--         RANK() OVER (PARTITION BY t.dt ORDER BY t.carts_uu DESC) AS carts_uu_rank,
--         RANK() OVER (PARTITION BY t.dt ORDER BY t.orders_uu DESC) AS orders_uu_rank,
        CASE WHEN clicks_uu = 0 THEN 0 ELSE clicks_cnt / clicks_uu END AS avg_clicks_cnt,
        CASE WHEN carts_uu = 0 THEN 0 ELSE carts_cnt / carts_uu END AS avg_carts_cnt,
        CASE WHEN orders_uu = 0 THEN 0 ELSE orders_cnt / orders_uu END AS avg_orders_cnt
    FROM (
        SELECT
            CAST(FORMAT_TIMESTAMP('%Y-%m-%d', TIMESTAMP_SECONDS(CAST(ts/1000 AS int))) as DATE) AS dt,
            aid,
            SUM(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) AS clicks_cnt,
            SUM(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) AS carts_cnt,
            SUM(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) AS orders_cnt,
            COUNT(DISTINCT(CASE WHEN type = 'clicks' THEN session ELSE NULL END)) AS clicks_uu,
            COUNT(DISTINCT(CASE WHEN type = 'carts' THEN session ELSE NULL END)) AS carts_uu,
            COUNT(DISTINCT(CASE WHEN type = 'orders' THEN session ELSE NULL END)) AS orders_uu
        FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
--         FROM `kaggle-352109.otto.test`
        GROUP BY dt, aid
    ) t
    INNER JOIN day_num d ON d.dt = t.dt
), aid_stats1 AS (
  SELECT
      aid,
      MAX(CASE WHEN day_num = 1 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day1,
      MAX(CASE WHEN day_num = 2 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day2,
      MAX(CASE WHEN day_num = 3 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day3,
      MAX(CASE WHEN day_num = 4 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day4,
      MAX(CASE WHEN day_num = 5 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day5,
      MAX(CASE WHEN day_num = 6 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day6,
      MAX(CASE WHEN day_num = 7 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day7,
      MAX(CASE WHEN day_num = 8 THEN clicks_cnt ELSE NULL END) AS clicks_cnt_day8,
      MAX(CASE WHEN day_num = 1 THEN carts_cnt ELSE NULL END) AS carts_cnt_day1,
      MAX(CASE WHEN day_num = 2 THEN carts_cnt ELSE NULL END) AS carts_cnt_day2,
      MAX(CASE WHEN day_num = 3 THEN carts_cnt ELSE NULL END) AS carts_cnt_day3,
      MAX(CASE WHEN day_num = 4 THEN carts_cnt ELSE NULL END) AS carts_cnt_day4,
      MAX(CASE WHEN day_num = 5 THEN carts_cnt ELSE NULL END) AS carts_cnt_day5,
      MAX(CASE WHEN day_num = 6 THEN carts_cnt ELSE NULL END) AS carts_cnt_day6,
      MAX(CASE WHEN day_num = 7 THEN carts_cnt ELSE NULL END) AS carts_cnt_day7,
      MAX(CASE WHEN day_num = 8 THEN carts_cnt ELSE NULL END) AS carts_cnt_day8,
      MAX(CASE WHEN day_num = 1 THEN orders_cnt ELSE NULL END) AS orders_cnt_day1,
      MAX(CASE WHEN day_num = 2 THEN orders_cnt ELSE NULL END) AS orders_cnt_day2,
      MAX(CASE WHEN day_num = 3 THEN orders_cnt ELSE NULL END) AS orders_cnt_day3,
      MAX(CASE WHEN day_num = 4 THEN orders_cnt ELSE NULL END) AS orders_cnt_day4,
      MAX(CASE WHEN day_num = 5 THEN orders_cnt ELSE NULL END) AS orders_cnt_day5,
      MAX(CASE WHEN day_num = 6 THEN orders_cnt ELSE NULL END) AS orders_cnt_day6,
      MAX(CASE WHEN day_num = 7 THEN orders_cnt ELSE NULL END) AS orders_cnt_day7,
      MAX(CASE WHEN day_num = 8 THEN orders_cnt ELSE NULL END) AS orders_cnt_day8,
      MAX(CASE WHEN day_num = 1 THEN clicks_rank ELSE NULL END) AS clicks_rank_day1,
      MAX(CASE WHEN day_num = 2 THEN clicks_rank ELSE NULL END) AS clicks_rank_day2,
      MAX(CASE WHEN day_num = 3 THEN clicks_rank ELSE NULL END) AS clicks_rank_day3,
      MAX(CASE WHEN day_num = 4 THEN clicks_rank ELSE NULL END) AS clicks_rank_day4,
      MAX(CASE WHEN day_num = 5 THEN clicks_rank ELSE NULL END) AS clicks_rank_day5,
      MAX(CASE WHEN day_num = 6 THEN clicks_rank ELSE NULL END) AS clicks_rank_day6,
      MAX(CASE WHEN day_num = 7 THEN clicks_rank ELSE NULL END) AS clicks_rank_day7,
      MAX(CASE WHEN day_num = 8 THEN clicks_rank ELSE NULL END) AS clicks_rank_day8,
      MAX(CASE WHEN day_num = 1 THEN carts_rank ELSE NULL END) AS carts_rank_day1,
      MAX(CASE WHEN day_num = 2 THEN carts_rank ELSE NULL END) AS carts_rank_day2,
      MAX(CASE WHEN day_num = 3 THEN carts_rank ELSE NULL END) AS carts_rank_day3,
      MAX(CASE WHEN day_num = 4 THEN carts_rank ELSE NULL END) AS carts_rank_day4,
      MAX(CASE WHEN day_num = 5 THEN carts_rank ELSE NULL END) AS carts_rank_day5,
      MAX(CASE WHEN day_num = 6 THEN carts_rank ELSE NULL END) AS carts_rank_day6,
      MAX(CASE WHEN day_num = 7 THEN carts_rank ELSE NULL END) AS carts_rank_day7,
      MAX(CASE WHEN day_num = 8 THEN carts_rank ELSE NULL END) AS carts_rank_day8,
      MAX(CASE WHEN day_num = 1 THEN orders_rank ELSE NULL END) AS orders_rank_day1,
      MAX(CASE WHEN day_num = 2 THEN orders_rank ELSE NULL END) AS orders_rank_day2,
      MAX(CASE WHEN day_num = 3 THEN orders_rank ELSE NULL END) AS orders_rank_day3,
      MAX(CASE WHEN day_num = 4 THEN orders_rank ELSE NULL END) AS orders_rank_day4,
      MAX(CASE WHEN day_num = 5 THEN orders_rank ELSE NULL END) AS orders_rank_day5,
      MAX(CASE WHEN day_num = 6 THEN orders_rank ELSE NULL END) AS orders_rank_day6,
      MAX(CASE WHEN day_num = 7 THEN orders_rank ELSE NULL END) AS orders_rank_day7,
      MAX(CASE WHEN day_num = 8 THEN orders_rank ELSE NULL END) AS orders_rank_day8,
      MAX(CASE WHEN day_num = 1 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day1,
      MAX(CASE WHEN day_num = 2 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day2,
      MAX(CASE WHEN day_num = 3 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day3,
      MAX(CASE WHEN day_num = 4 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day4,
      MAX(CASE WHEN day_num = 5 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day5,
      MAX(CASE WHEN day_num = 6 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day6,
      MAX(CASE WHEN day_num = 7 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day7,
      MAX(CASE WHEN day_num = 8 THEN avg_clicks_cnt ELSE NULL END) AS avg_clicks_cnt_day8,
      MAX(CASE WHEN day_num = 1 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day1,
      MAX(CASE WHEN day_num = 2 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day2,
      MAX(CASE WHEN day_num = 3 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day3,
      MAX(CASE WHEN day_num = 4 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day4,
      MAX(CASE WHEN day_num = 5 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day5,
      MAX(CASE WHEN day_num = 6 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day6,
      MAX(CASE WHEN day_num = 7 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day7,
      MAX(CASE WHEN day_num = 8 THEN avg_carts_cnt ELSE NULL END) AS avg_carts_cnt_day8,
      MAX(CASE WHEN day_num = 1 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day1,
      MAX(CASE WHEN day_num = 2 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day2,
      MAX(CASE WHEN day_num = 3 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day3,
      MAX(CASE WHEN day_num = 4 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day4,
      MAX(CASE WHEN day_num = 5 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day5,
      MAX(CASE WHEN day_num = 6 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day6,
      MAX(CASE WHEN day_num = 7 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day7,
      MAX(CASE WHEN day_num = 8 THEN avg_orders_cnt ELSE NULL END) AS avg_orders_cnt_day8,
  FROM daily_rank
  GROUP BY aid
), aid_stats2 AS (
    SELECT
        aid,
        CASE WHEN SUM(is_clicked) = 0 THEN NULL ELSE SUM(is_carted) / SUM(is_clicked) END AS clicks_carts_ratio,    -- 全期間でなく、windowを変えてみてもよいかも
        CASE WHEN SUM(is_carted) = 0 THEN NULL ELSE SUM(is_ordered) / SUM(is_carted) END AS carts_orders_ratio,
        CASE WHEN SUM(is_clicked) = 0 THEN NULL ELSE SUM(is_ordered) / SUM(is_clicked) END AS clicks_orders_ratio,
        AVG(sec_clicks_carts) AS avg_sec_clicks_carts,
        MIN(sec_clicks_carts) AS min_sec_clicks_carts,
        MAX(sec_clicks_carts) AS max_sec_clicks_carts,
        AVG(sec_carts_orders) AS avg_sec_carts_orders,
        MIN(sec_carts_orders) AS min_sec_carts_orders,
        MAX(sec_carts_orders) AS max_sec_carts_orders
    FROM (
        SELECT
            session,
            aid,
            is_clicked,
            is_carted,
            is_ordered,
            CASE WHEN first_carts_ts - first_clicks_ts > 0 THEN first_carts_ts - first_clicks_ts ELSE NULL END AS sec_clicks_carts,
            CASE WHEN first_orders_ts - first_carts_ts > 0 THEN first_orders_ts - first_carts_ts ELSE NULL END AS sec_carts_orders
        FROM (
            SELECT
                session,
                aid,
                MAX(CASE WHEN type = 'clicks' THEN 1 ELSE 0 END) AS is_clicked,
                MAX(CASE WHEN type = 'carts' THEN 1 ELSE 0 END) AS is_carted,
                MAX(CASE WHEN type = 'orders' THEN 1 ELSE 0 END) AS is_ordered,
                MIN(CASE WHEN type = 'clicks' THEN ts ELSE NULL END) AS first_clicks_ts,
                MIN(CASE WHEN type = 'carts' THEN ts ELSE NULL END) AS first_carts_ts,
                MIN(CASE WHEN type = 'orders' THEN ts ELSE NULL END) AS first_orders_ts
            FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
--             FROM `kaggle-352109.otto.test`
            GROUP BY session, aid
        ) t
    ) t
    GROUP BY aid
), aid_stats AS (
    SELECT
        a1.aid,
        a1.clicks_cnt_day1,
        a1.clicks_cnt_day2,
        a1.clicks_cnt_day3,
        a1.clicks_cnt_day4,
        a1.clicks_cnt_day5,
        a1.clicks_cnt_day6,
        a1.clicks_cnt_day7,
        a1.clicks_cnt_day8,
        a1.carts_cnt_day1,
        a1.carts_cnt_day2,
        a1.carts_cnt_day3,
        a1.carts_cnt_day4,
        a1.carts_cnt_day5,
        a1.carts_cnt_day6,
        a1.carts_cnt_day7,
        a1.carts_cnt_day8,
        a1.orders_cnt_day1,
        a1.orders_cnt_day2,
        a1.orders_cnt_day3,
        a1.orders_cnt_day4,
        a1.orders_cnt_day5,
        a1.orders_cnt_day6,
        a1.orders_cnt_day7,
        a1.orders_cnt_day8,
        a1.clicks_rank_day1,
        a1.clicks_rank_day2,
        a1.clicks_rank_day3,
        a1.clicks_rank_day4,
        a1.clicks_rank_day5,
        a1.clicks_rank_day6,
        a1.clicks_rank_day7,
        a1.clicks_rank_day8,
        a1.carts_rank_day1,
        a1.carts_rank_day2,
        a1.carts_rank_day3,
        a1.carts_rank_day4,
        a1.carts_rank_day5,
        a1.carts_rank_day6,
        a1.carts_rank_day7,
        a1.carts_rank_day8,
        a1.orders_rank_day1,
        a1.orders_rank_day2,
        a1.orders_rank_day3,
        a1.orders_rank_day4,
        a1.orders_rank_day5,
        a1.orders_rank_day6,
        a1.orders_rank_day7,
        a1.orders_rank_day8,
        a1.avg_clicks_cnt_day1,
        a1.avg_clicks_cnt_day2,
        a1.avg_clicks_cnt_day3,
        a1.avg_clicks_cnt_day4,
        a1.avg_clicks_cnt_day5,
        a1.avg_clicks_cnt_day6,
        a1.avg_clicks_cnt_day7,
        a1.avg_clicks_cnt_day8,
        a1.avg_carts_cnt_day1,
        a1.avg_carts_cnt_day2,
        a1.avg_carts_cnt_day3,
        a1.avg_carts_cnt_day4,
        a1.avg_carts_cnt_day5,
        a1.avg_carts_cnt_day6,
        a1.avg_carts_cnt_day7,
        a1.avg_carts_cnt_day8,
        a1.avg_orders_cnt_day1,
        a1.avg_orders_cnt_day2,
        a1.avg_orders_cnt_day3,
        a1.avg_orders_cnt_day4,
        a1.avg_orders_cnt_day5,
        a1.avg_orders_cnt_day6,
        a1.avg_orders_cnt_day7,
        a1.avg_orders_cnt_day8,
        a2.clicks_carts_ratio,
        a2.carts_orders_ratio,
        a2.clicks_orders_ratio,
        a2.avg_sec_clicks_carts,
        a2.min_sec_clicks_carts,
        a2.max_sec_clicks_carts,
        a2.avg_sec_carts_orders,
        a2.min_sec_carts_orders,
        a2.max_sec_carts_orders
    FROM aid_stats1 a1
    INNER JOIN aid_stats2 a2 ON a1.aid = a2.aid
), aid_list AS (
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
        FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
--         FROM `kaggle-352109.otto.test`
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
        NULL AS w2v_candidate_num,
        NULL AS gru4rec_candidate_num,
        NULL AS narm_candidate_num,
        NULL AS sasrec_candidate_num,
        NULL AS srgnn_candidate_num,
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
        NULL AS w2v_candidate_num,
        NULL AS gru4rec_candidate_num,
        NULL AS narm_candidate_num,
        NULL AS sasrec_candidate_num,
        NULL AS srgnn_candidate_num,
    FROM `kaggle-352109.otto.covisit_c100_cv` -- FIXME
--     FROM `kaggle-352109.otto.covisit_c100`
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
        rank AS w2v_candidate_num,
        NULL AS gru4rec_candidate_num,
        NULL AS narm_candidate_num,
        NULL AS sasrec_candidate_num,
        NULL AS srgnn_candidate_num,
    FROM `kaggle-352109.otto.w2v_c100_cv` -- FIXME
--     FROM `kaggle-352109.otto.w2v_c100`
    WHERE aid is not NULL
), gru4rec AS (
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
        NULL AS w2v_candidate_num,
        rank AS gru4rec_candidate_num,
        NULL AS narm_candidate_num,
        NULL AS sasrec_candidate_num,
        NULL AS srgnn_candidate_num,
    FROM `kaggle-352109.otto.gru4rec_c100_cv` -- FIXME
--     FROM `kaggle-352109.otto.gru4rec_c100`
    WHERE aid is not NULL
), narm AS (
    SELECT
        session,
        list.item AS aid,
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
        NULL AS w2v_candidate_num,
        NULL AS gru4rec_candidate_num,
        ROW_NUMBER() OVER (PARTITION BY session)  AS narm_candidate_num,
        NULL AS sasrec_candidate_num,
        NULL AS srgnn_candidate_num,
    FROM `kaggle-352109.otto.narm_aggs_cv`,  -- FIXME
--     FROM `kaggle-352109.otto.narm_aggs`,
    UNNEST(labels.list) AS list
), sasrec AS (
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
        NULL AS w2v_candidate_num,
        NULL AS gru4rec_candidate_num,
        NULL AS narm_candidate_num,
        rank AS sasrec_candidate_num,
        NULL AS srgnn_candidate_num,
    FROM `kaggle-352109.otto.sasrec_cv` -- FIXME
--     FROM `kaggle-352109.otto.sasrec`
    WHERE aid is not NULL
), srgnn AS (
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
        NULL AS w2v_candidate_num,
        NULL AS gru4rec_candidate_num,
        NULL AS narm_candidate_num,
        NULL AS sasrec_candidate_num,
        rank AS srgnn_candidate_num,
    FROM `kaggle-352109.otto.srgnn_cv` -- FIXME
--     FROM `kaggle-352109.otto.srgnn`
    WHERE aid is not NULL
), ranking AS (
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
        NULL AS w2v_candidate_num,
        NULL AS gru4rec_candidate_num,
        NULL AS narm_candidate_num,
        NULL AS sasrec_candidate_num,
        NULL AS srgnn_candidate_num,
    FROM (
        SELECT session
        FROM `kaggle-352109.otto.otto-validation-test` -- FIXME
--         FROM `kaggle-352109.otto.test`
        GROUP BY session
    ) t
    CROSS JOIN (
    SELECT
        aid
    FROM daily_rank
    WHERE orders_rank <= 10
    GROUP BY aid
    )
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
        MAX(gru4rec_candidate_num) AS gru4rec_candidate_num,
        MAX(narm_candidate_num) AS narm_candidate_num,
        MAX(sasrec_candidate_num) AS sasrec_candidate_num,
        MAX(srgnn_candidate_num) AS srgnn_candidate_num,
    FROM (
        SELECT * FROM aggregate_by_session_aid
        UNION ALL
        SELECT * FROM covisit
        UNION ALL
        SELECT * FROM w2v
        UNION ALL
        SELECT * FROM gru4rec
--         WHERE gru4rec_candidate_num <= 30
        UNION ALL
        SELECT * FROM narm
--         WHERE narm_candidate_num <= 30
        UNION ALL
        SELECT * FROM sasrec
        UNION ALL
        SELECT * FROM srgnn
        UNION ALL
        SELECT * FROM ranking
    ) t
    GROUP BY session, aid
)

SELECT
    sa.session,
    sa.aid,
    ss.session_clicks_cnt,
    ss.session_carts_cnt,
    ss.session_orders_cnt,
    ss.session_clicks_unique_aid,
    ss.session_carts_unique_aid,
    ss.session_orders_unique_aid,
    ss.session_clicks_carts_ratio,
    ss.session_carts_orders_ratio,
    ss.session_clicks_orders_ratio,
    ss.avg_sec_session_clicks_carts,
    ss.min_sec_session_clicks_carts,
    ss.max_sec_session_clicks_carts,
    ss.avg_sec_session_carts_orders,
    ss.min_sec_session_carts_orders,
    ss.max_sec_session_carts_orders,
    ss.min_day_num,
    ss.max_day_num,
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
    sa.gru4rec_candidate_num,
    sa.narm_candidate_num,
    sa.sasrec_candidate_num,
    sa.srgnn_candidate_num,
    COALESCE(ais.clicks_cnt_day1, 0) AS clicks_cnt_day1,
    COALESCE(ais.clicks_cnt_day2, 0) AS clicks_cnt_day2,
    COALESCE(ais.clicks_cnt_day3, 0) AS clicks_cnt_day3,
    COALESCE(ais.clicks_cnt_day4, 0) AS clicks_cnt_day4,
    COALESCE(ais.clicks_cnt_day5, 0) AS clicks_cnt_day5,
    COALESCE(ais.clicks_cnt_day6, 0) AS clicks_cnt_day6,
    COALESCE(ais.clicks_cnt_day7, 0) AS clicks_cnt_day7,
    COALESCE(ais.clicks_cnt_day8, 0) AS clicks_cnt_day8,
    COALESCE(ais.carts_cnt_day1, 0) AS carts_cnt_day1,
    COALESCE(ais.carts_cnt_day2, 0) AS carts_cnt_day2,
    COALESCE(ais.carts_cnt_day3, 0) AS carts_cnt_day3,
    COALESCE(ais.carts_cnt_day4, 0) AS carts_cnt_day4,
    COALESCE(ais.carts_cnt_day5, 0) AS carts_cnt_day5,
    COALESCE(ais.carts_cnt_day6, 0) AS carts_cnt_day6,
    COALESCE(ais.carts_cnt_day7, 0) AS carts_cnt_day7,
    COALESCE(ais.carts_cnt_day8, 0) AS carts_cnt_day8,
    COALESCE(ais.orders_cnt_day1, 0) AS orders_cnt_day1,
    COALESCE(ais.orders_cnt_day2, 0) AS orders_cnt_day2,
    COALESCE(ais.orders_cnt_day3, 0) AS orders_cnt_day3,
    COALESCE(ais.orders_cnt_day4, 0) AS orders_cnt_day4,
    COALESCE(ais.orders_cnt_day5, 0) AS orders_cnt_day5,
    COALESCE(ais.orders_cnt_day6, 0) AS orders_cnt_day6,
    COALESCE(ais.orders_cnt_day7, 0) AS orders_cnt_day7,
    COALESCE(ais.orders_cnt_day8, 0) AS orders_cnt_day8,
    COALESCE(ais.clicks_rank_day1, 1000000) AS clicks_rank_day1,
    COALESCE(ais.clicks_rank_day2, 1000000) AS clicks_rank_day2,
    COALESCE(ais.clicks_rank_day3, 1000000) AS clicks_rank_day3,
    COALESCE(ais.clicks_rank_day4, 1000000) AS clicks_rank_day4,
    COALESCE(ais.clicks_rank_day5, 1000000) AS clicks_rank_day5,
    COALESCE(ais.clicks_rank_day6, 1000000) AS clicks_rank_day6,
    COALESCE(ais.clicks_rank_day7, 1000000) AS clicks_rank_day7,
    COALESCE(ais.clicks_rank_day8, 1000000) AS clicks_rank_day8,
    COALESCE(ais.carts_rank_day1, 1000000) AS carts_rank_day1,
    COALESCE(ais.carts_rank_day2, 1000000) AS carts_rank_day2,
    COALESCE(ais.carts_rank_day3, 1000000) AS carts_rank_day3,
    COALESCE(ais.carts_rank_day4, 1000000) AS carts_rank_day4,
    COALESCE(ais.carts_rank_day5, 1000000) AS carts_rank_day5,
    COALESCE(ais.carts_rank_day6, 1000000) AS carts_rank_day6,
    COALESCE(ais.carts_rank_day7, 1000000) AS carts_rank_day7,
    COALESCE(ais.carts_rank_day8, 1000000) AS carts_rank_day8,
    COALESCE(ais.orders_rank_day1, 1000000) AS orders_rank_day1,
    COALESCE(ais.orders_rank_day2, 1000000) AS orders_rank_day2,
    COALESCE(ais.orders_rank_day3, 1000000) AS orders_rank_day3,
    COALESCE(ais.orders_rank_day4, 1000000) AS orders_rank_day4,
    COALESCE(ais.orders_rank_day5, 1000000) AS orders_rank_day5,
    COALESCE(ais.orders_rank_day6, 1000000) AS orders_rank_day6,
    COALESCE(ais.orders_rank_day7, 1000000) AS orders_rank_day7,
    COALESCE(ais.orders_rank_day8, 1000000) AS orders_rank_day8,
    COALESCE(ais.avg_clicks_cnt_day1, 0) AS avg_clicks_cnt_day1,
    COALESCE(ais.avg_clicks_cnt_day2, 0) AS avg_clicks_cnt_day2,
    COALESCE(ais.avg_clicks_cnt_day3, 0) AS avg_clicks_cnt_day3,
    COALESCE(ais.avg_clicks_cnt_day4, 0) AS avg_clicks_cnt_day4,
    COALESCE(ais.avg_clicks_cnt_day5, 0) AS avg_clicks_cnt_day5,
    COALESCE(ais.avg_clicks_cnt_day6, 0) AS avg_clicks_cnt_day6,
    COALESCE(ais.avg_clicks_cnt_day7, 0) AS avg_clicks_cnt_day7,
    COALESCE(ais.avg_clicks_cnt_day8, 0) AS avg_clicks_cnt_day8,
    COALESCE(ais.avg_carts_cnt_day1, 0) AS avg_carts_cnt_day1,
    COALESCE(ais.avg_carts_cnt_day2, 0) AS avg_carts_cnt_day2,
    COALESCE(ais.avg_carts_cnt_day3, 0) AS avg_carts_cnt_day3,
    COALESCE(ais.avg_carts_cnt_day4, 0) AS avg_carts_cnt_day4,
    COALESCE(ais.avg_carts_cnt_day5, 0) AS avg_carts_cnt_day5,
    COALESCE(ais.avg_carts_cnt_day6, 0) AS avg_carts_cnt_day6,
    COALESCE(ais.avg_carts_cnt_day7, 0) AS avg_carts_cnt_day7,
    COALESCE(ais.avg_carts_cnt_day8, 0) AS avg_carts_cnt_day8,
    COALESCE(ais.avg_orders_cnt_day1, 0) AS avg_orders_cnt_day1,
    COALESCE(ais.avg_orders_cnt_day2, 0) AS avg_orders_cnt_day2,
    COALESCE(ais.avg_orders_cnt_day3, 0) AS avg_orders_cnt_day3,
    COALESCE(ais.avg_orders_cnt_day4, 0) AS avg_orders_cnt_day4,
    COALESCE(ais.avg_orders_cnt_day5, 0) AS avg_orders_cnt_day5,
    COALESCE(ais.avg_orders_cnt_day6, 0) AS avg_orders_cnt_day6,
    COALESCE(ais.avg_orders_cnt_day7, 0) AS avg_orders_cnt_day7,
    COALESCE(ais.avg_orders_cnt_day8, 0) AS avg_orders_cnt_day8,
    ais.clicks_carts_ratio,
    ais.carts_orders_ratio,
    ais.clicks_orders_ratio,
    ais.avg_sec_clicks_carts,
    ais.min_sec_clicks_carts,
    ais.max_sec_clicks_carts,
    ais.avg_sec_carts_orders,
    ais.min_sec_carts_orders,
    ais.max_sec_carts_orders
FROM union_all sa
LEFT JOIN session_stats ss ON ss.session = sa.session
LEFT JOIN aid_stats ais ON ais.aid = sa.aid
