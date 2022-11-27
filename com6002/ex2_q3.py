from pymongo import MongoClient
mongo = MongoClient()
col = mongo.com6002.task2
colls = col.find({})
result = []
for x in colls:
    print(x)
    if x['dob'].month == 9 :
        result.append(x)

print(len(result))
