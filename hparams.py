import argparse

class Hparams:
    parser = argparse.ArgumentParser()


    # train
    ## files 许海明
    parser.add_argument('--train1', default='data/demo1',
                             help="原始语料")
    parser.add_argument('--train2', default='data/demo1',
                             help="目标语料")
    parser.add_argument('--eval1', default='data/demo1',
                             help=" 原始语料 segmented ")
    parser.add_argument('--eval2', default='data/demo1',
                             help="目标语料")
    parser.add_argument('--eval3', default='iwslt2016/prepro/eval.en',
                             help="english evaluation unsegmented data")

    ## vocabulary 许海明
    parser.add_argument('--vocab_file', default='bert/vocab.txt',
                        help="单词的路径")

    # training scheme 许海明
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory  参数存储路径")
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model 许海明
    parser.add_argument('--d_model', default=768, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    # 这里的参数注意一下
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")

    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default='iwslt2016/prepro/test.en',
                        help="english test data")
    #注意这里
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")

    # 下面是bertconfig 里面的参数
    parser.add_argument('--attention_probs_dropout_prob', default=0.1, type=float,help="bert的参数")
    parser.add_argument('--directionality', default="bidi", help="bert的参数")
    parser.add_argument('--hidden_act', default="gelu", help="bert的参数")
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float,help="bert的参数")
    parser.add_argument('--hidden_size', default=768, type=int, help="bert的参数")
    parser.add_argument('--initializer_range', default=0.02, type=float,help="bert的参数")
    parser.add_argument('--intermediate_size', default=3072, type=int, help="bert的参数")
    parser.add_argument('--max_position_embeddings', default=512, type=int, help="bert的参数")
    parser.add_argument('--num_attention_heads', default=12, type=int, help="bert的参数")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="bert的参数")
    parser.add_argument('--pooler_fc_size', default=768, type=int, help="bert的参数")
    parser.add_argument('--pooler_num_attention_heads', default=12, type=int, help="bert的参数")
    parser.add_argument('--pooler_num_fc_layers', default=3, type=int, help="bert的参数")
    parser.add_argument('--pooler_size_per_head', default=128, type=int, help="bert的参数")
    parser.add_argument('--pooler_type', default="first_token_transform", help="bert的参数")
    parser.add_argument('--type_vocab_size', default=2, type=int, help="bert的参数")
    parser.add_argument('--vocab_size', default=21128, type=int, help="bert的参数")


    parser.add_argument('--init_checkpoint', default="bert/bert_model.ckpt",  help="bert的参数")





