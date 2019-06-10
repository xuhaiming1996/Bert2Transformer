import random

all=[]

with open("source.txt",mode="r",encoding="utf-8") as fr:
    for line in fr:
        line = line.strip()
        all.append(line)

random.shuffle(all)

##  生成测试集
test=all[:20000]
fw_s = open("dev.source",mode="w",encoding="utf-8")
fw_t = open("dev.target",mode="w",encoding="utf-8")
for line in test:
    _,sen1,sen2=line.split("\t")
    fw_s.write(sen1+"\n")
    fw_t.write(sen2+"\n")

fw_s.close()
fw_t.close()


##  生成训练集
train=all[20000:]
fw_s = open("train.source", mode="w", encoding="utf-8")
fw_t = open("train.target", mode="w", encoding="utf-8")
for line in train:
    _, sen1, sen2 = line.split("\t")
    fw_s.write(sen1 + "\n")
    fw_t.write(sen2 + "\n")

fw_s.close()
fw_t.close()