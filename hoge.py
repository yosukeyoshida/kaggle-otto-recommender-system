# for c in ["clicks", "carts", "orders"]:
# 	for i in range(35):
# 		num = i + 1
# 		print(f"MAX(CASE WHEN day_num = {num} THEN {c}_rank ELSE NULL END) AS {c}_rank_day{num}")
# 


for c in ["clicks", "carts", "orders"]:
	for i in range(35):
		num = i + 1
		print(f"COALESCE(ais.{c}_rank_day{num}, 1000000) AS {c}_rank_day{num},")
