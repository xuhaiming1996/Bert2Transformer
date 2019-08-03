### 这里提示一下
这个代码还有少部分可以优化的地方，最近有点忙，我本来打算后期修改的，
最近看大家有不少疑问，大家如果想快速的运行起bert2transformer代码，
可以去本人的这个项目下对这个项目稍作修改就可以马上变成bert2transformer
这个项目[BERT-T2T](https://github.com/xuhaiming1996/BERT-T2T)也是我从这个项目修改的得到的，
相比，这个项目[BERT-T2T](https://github.com/xuhaiming1996/BERT-T2T)代码更简洁，
同时改正了这个项目无法训练超过千万级别数据的问题和对这个[问题](https://github.com/xuhaiming1996/Bert2Transformer/issues/4) 进行优化。

# Bert2Transformer
这是一个seq2seq模型


# 思想
编码器是bert,解码器是transformer的解码器，可用于自然语言处理中文本生成领域的任务
采用的还是微调的方式，bert是已经训练好

### 提示一下
我的src和tgt全部都是中文，同一种语言，

我的src和tgt是不同的两种语言，需要自己改下


