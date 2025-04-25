import os
from zhenshiapi import Zhenshi, Account
import polars as pl
import numpy as np

token = os.environ.get("TOKEN")
if token is None:
    raise ValueError("TOKEN environment variable not set")

z1 = Zhenshi(Account.from_dict({"token": token}))

stock_list = pl.from_dicts(z1.get_stocklist()['data'])
rank = pl.from_dicts(z1.get_rank()["data"])
userstock_list = []
for i,user in enumerate(rank["username"]):
    userstock = pl.from_dicts(z1.get_other_userstock_byname(user)["data"]
        ).select(pl.col("name"), pl.col("number")
        ).join(stock_list, on="name").select(pl.col("code"), (pl.col("number")*pl.col("price")).alias("value")
        ).with_columns(pl.lit(user).alias("username")
        )
    userstock_list.append(userstock)
userstock_list = pl.concat(userstock_list)
userstockvalue_rank = userstock_list.pivot(index="username", on="code").fill_null(0)
summary_rank = rank.join(userstockvalue_rank, on="username").select(
    pl.col("username"),
    pl.col("nickname"),
    pl.col("total"),
    pl.col("000001"),
    pl.col("000002"),
    pl.col("000003"),
    pl.col("000004"),
    (pl.col("total") - pl.col("000001") - pl.col("000002") - pl.col("000003") - pl.col("000004")).alias("cash")
)

rankmoney = pl.from_dicts(z1.get_rank_money()["data"])
userstock_list = []
for i,user in enumerate(rankmoney["username"]):
    try:
        userstock = pl.from_dicts(z1.get_other_userstock_byname(user)["data"]
        ).select(pl.col("name"), pl.col("number")
        ).join(stock_list, on="name").select(pl.col("code"), (pl.col("number")*pl.col("price")).alias("value")
        ).with_columns(pl.lit(user).alias("username")
        )
        userstock_list.append(userstock)
    except:
        pass
userstock_list = pl.concat(userstock_list)
userstockvalue_rankmoney = userstock_list.pivot(index="username", on="code")


summary_rankmoney = rankmoney.join(userstockvalue_rankmoney, on="username").fill_null(0).select(
    pl.col("username"),
    pl.col("nickname"),
    pl.col("money"),
    pl.col("000001"),
    pl.col("000002"),
    pl.col("000003"),
    pl.col("000004"),
    (pl.col("000008")+pl.col("money")).alias("cash")
)

s1 = summary_rank.sum().to_numpy()[0][3:].astype(np.float64)
s1 = np.log(s1/s1.sum())
s2 = summary_rankmoney.sum().to_numpy()[0][3:].astype(np.float64)
s2 = np.log(s2/s2.sum())

monitor_list = ["发广告", "Avocado", "青岚", "韩苏川", "青鸟",
             "秋忆",   "手机用户69717689", "手机用户75020050", "uu", "陈平安", 
             "User355965915", "光明", "苗", "手机用户85054999",
             "龙健富源", "小康", "广告位招租", "CZ", "石沟"
         ]
summary_m = summary_rank.filter(pl.col("nickname").is_in(monitor_list))
s3 = summary_m.sum().to_numpy()[0][3:].astype(np.float64)
s3 = np.log(s3/s3.sum())
absl = s3 - s1
rel = (s3-s1)/np.abs(s1-s2)
print("绝对偏离度：", absl)
print("相对偏离度：", rel)