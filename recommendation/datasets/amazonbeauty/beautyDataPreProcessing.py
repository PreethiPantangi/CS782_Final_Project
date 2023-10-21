import gzip
from collections import defaultdict
from datetime import datetime
import json
 
def process_amazon_beauty_data():
    def parse(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield json.loads(l)


    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    dataset_name = 'Beauty'
    f = open('reviews_' + dataset_name + '.txt', 'w')
    userIdMap = dict()
    productIdMap = dict()

    userIdCount = 1
    productIdCount = 1

    for l in parse('reviews_' + dataset_name + '.json.gz'):
        line += 1
        if l['reviewerID'] not in userIdMap.keys():
            userIdMap[l['reviewerID']] = userIdCount
            userIdCount += 1
        if l['asin'] not in productIdMap.keys():
            productIdMap[l['asin']] = productIdCount
            productIdCount += 1
        f.write(" ".join([str(userIdMap[l['reviewerID']]), str(productIdMap[l['asin']]), str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        countU[rev] += 1
        countP[asin] += 1
    f.close()

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    for l in parse('reviews_' + dataset_name + '.json.gz'):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])
    # sort reviews in User according to time

    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum)

# This is a guard clause that ensures the code is executed only if the script is run directly, 
# and not when it's imported by another script.
if __name__ == "__main__":
    process_amazon_beauty_data()