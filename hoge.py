for t in ['clicks', 'carts', 'orders']:
	for i in range(7):
		num = i+1
		# print(f"SUM(CASE WHEN d.num = {num}  AND type = '{t}' THEN 1 ELSE 0 END) AS {t}_cnt_day{num},")
		# print(f"MAX(session_aid_{t}_cnt_day{num}) AS session_aid_{t}_cnt_day{num},")
		# print(f"COALESCE(session_{t}_cnt_day{num}, 0) AS session_{t}_cnt_day{num},")
		print(f"COALESCE(session_aid_{t}_cnt_day{num}, 0) AS session_aid_{t}_cnt_day{num},")
