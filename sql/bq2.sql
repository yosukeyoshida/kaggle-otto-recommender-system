EXPORT DATA
  OPTIONS(
    uri='gs://kaggle-yosuke/lgbm_dataset/20230114_2/clicks/train_*.parquet',  -- FIXME
--     uri='gs://kaggle-yosuke/lgbm_dataset/20230114_2/carts/train_*.parquet',
--     uri='gs://kaggle-yosuke/lgbm_dataset/20230114_2/orders/train_*.parquet',
    format='PARQUET',
    overwrite=true
  )
AS
WITH joined AS (
  SELECT
    c.*,
    t.type
  FROM `kaggle-352109.otto.absurd-shadow-602` c
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
)

SELECT c.*
FROM `kaggle-352109.otto.absurd-shadow-602` c
WHERE c.session IN (SELECT session FROM target_session)
