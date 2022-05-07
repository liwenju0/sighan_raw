import json


def gen_json_data(filename="test13"):
    base_dir = "pair_data/simplified/{}_{}.txt"
    result = []
    for le, lc in zip(open(base_dir.format(filename, "error")), open(base_dir.format(filename, "correct"))):
        item = {}
        le = le.strip()
        lc = lc.strip()
        item['original_text'] = le
        item['correct_text'] = lc
        wids = []
        for i, (e, c) in enumerate(zip(le, lc)):
            if e != c:
                wids.append(i)
        item['wrong_ids'] = wids
        result.append(item)

    json.dump(result, open("pair_data/simplified/{}.json".format(filename), "w"), ensure_ascii=False)


for i in range(13, 16):
    gen_json_data("train" + str(i))
    gen_json_data("test" + str(i))
