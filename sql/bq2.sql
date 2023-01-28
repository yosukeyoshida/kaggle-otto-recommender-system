EXPORT DATA
  OPTIONS(
    uri='gs://kaggle-yosuke/lgbm_dataset/20230129/train_*.parquet',  -- FIXME
    format='PARQUET',
    overwrite=true
  )
AS
WITH joined AS (
  SELECT
    c.*,
    t.type
  FROM `kaggle-352109.otto.20230121` c
  LEFT JOIN (
    SELECT
      session,
      type,
      list.item AS gt
    FROM `kaggle-352109.otto.otto-validation-test-labels`, UNNEST(ground_truth.list) AS list
  ) t ON t.session = c.session AND t.gt = c.aid
), target_session AS (
  SELECT *
  FROM (
    SELECT
      session,
      SUM(CASE WHEN type = "carts" THEN 1 ELSE 0 END) AS carts_cnt,
      SUM(CASE WHEN type = "orders" THEN 1 ELSE 0 END) AS orders_cnt
    FROM joined
    WHERE type is not NULL
    GROUP BY session
  ) t
  WHERE t.carts_cnt > 0 OR t.orders_cnt > 0
), negative_list AS (
  SELECT
    session,
    aid
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER (PARTITION BY t.session ORDER BY random) AS rn
    FROM (
      SELECT
        j.*,
        CASE WHEN h.carts_cnt > h.orders_cnt THEN h.carts_cnt ELSE h.orders_cnt END AS gt_cnt,
        rand() AS random
      FROM joined j
      INNER JOIN target_session h ON h.session = j.session
      WHERE j.type is NULL
    ) t
  ) t
  WHERE rn <= 20 * gt_cnt
), positive_list AS (
  SELECT
    session,
    aid
  FROM joined
  WHERE session IN (SELECT session FROM target_session)
)

SELECT c.*
FROM `kaggle-352109.otto.20230121` c
INNER JOIN (
  SELECT *
  FROM (
    SELECT session, aid FROM positive_list
    UNION ALL
    SELECT session, aid FROM negative_list
 )
  GROUP BY session, aid
) t ON c.session = t.session AND c.aid = t.aid
