EXPORT DATA
  OPTIONS(
    uri='gs://kaggle-yosuke/lgbm_dataset/20230119_3/clicks/train_*.parquet',  -- FIXME
--     uri='gs://kaggle-yosuke/lgbm_dataset/20230119_3/carts/train_*.parquet',
--     uri='gs://kaggle-yosuke/lgbm_dataset/20230119_3/orders/train_*.parquet',
    format='PARQUET',
    overwrite=true
  )
AS
WITH joined AS (
  SELECT
    c.*,
    t.type
  FROM `kaggle-352109.otto.20230119_3` c
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
      SUM(CASE WHEN type = "clicks" THEN 1 ELSE 0 END) AS clicks_cnt,
      SUM(CASE WHEN type = "carts" THEN 1 ELSE 0 END) AS carts_cnt,
      SUM(CASE WHEN type = "orders" THEN 1 ELSE 0 END) AS orders_cnt
    FROM joined
    WHERE type is not NULL
    GROUP BY session
  ) t
  WHERE t.clicks_cnt > 0  -- FIXME
--   WHERE t.carts_cnt > 0
--   WHERE t.orders_cnt > 0
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
        h.clicks_cnt AS gt_cnt,  -- FIXME
--         h.carts_cnt AS gt_cnt,
--         h.orders_cnt AS gt_cnt,
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
  WHERE type = 'clicks'  -- FIXME
--   WHERE type = 'carts'
--   WHERE type = 'orders'
  AND session IN (SELECT session FROM target_session)
)

SELECT c.*
FROM `kaggle-352109.otto.20230119_3` c
INNER JOIN (
  SELECT *
  FROM (
    SELECT session, aid FROM positive_list
    UNION ALL
    SELECT session, aid FROM negative_list
 )
  GROUP BY session, aid
) t ON c.session = t.session AND c.aid = t.aid
