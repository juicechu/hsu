from pymongo import MongoClient
mongo = MongoClient()
col = mongo.com6002.task2
colls = col.find({"referral": "Ann"})
sumval = 0
count = 0
for x in colls:
    count += 1
    sumval += x['point']

print(sumval/count)
