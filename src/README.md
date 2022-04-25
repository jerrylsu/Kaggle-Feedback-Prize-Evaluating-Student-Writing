##  fold0 / total 10

剩余参数均为trainer默认   transformers==4.16.2
1754_3508_5262_7016_8770_10524

### ensemble
                                 CV                LB
                ll0-3 + dl0-2: 0.694023
                ll0-4 + dl0-2: 0.695846  (best)   0.686
                ll0-4 + dl0-3: 0.695055 
                ll0-4 + dl0-4: 0.694740

                ll1-3 + dl1-2: 0.691000
                ll1-3 + dl1-3: 0.693318 
                ll1-3 + dl1-4: 0.699153  
                ll1-3 + dl1-5: 0.695985 
                ll1-4 + dl1-4: 0.699555  (best)
                
                ll1-5 + dl1-5: 0.693409
        
ll0-4 + dl0-2 + ll1-3 + dl1-4:                    0.695 
ll0-4 + dl0-2 + ll1-4 + dl1-4:                    ?

bl0-4 + ll0-3 + dl0-2: 0.677526

### bigbird

python train.py --fold 0/1 --model_name google/bigbird-roberta-large 

```
3508-epoch2: 0.458971
5262-epoch3: 0.536037        eval_loss best
7016-epoch4: 0.544408 (best)
```

### longformer 

python train.py --fold 0/1 --model_name allenai/longformer-large-4096

```fold0
3508-epoch2: 0.661717        eval_loss best
5262-epoch3: 0.674583 (best)
7016-epoch4: 0.674556
```

```fold1
1754-epoch1: 0.653609
3508-epoch2: 0.660065         eval_lost best
5262-epoch3: 0.682372 (best)
7016-epoch4: 0.675744
```

```fold2
3508-epoch2: 0.640925
5262-epoch3: 0.673315
```

### deberta

python train.py --fold 0/1 --model_name microsoft/deberta-large

```fold0
3508-epoch2: 0.678108 (best) eval_loss best
5262-epoch3: 0.677201
7016-epoch4: 0.676900
```

```fold1
1754-epoch1:  0.662813
3508-epoch2:  0.681529        eval_loss best
5262-epoch3:  0.685918 (best)
7016-epoch4:  0.684087
8770-epoch5:  0.677545
10524-epoch6: 0.675375
```

```fold2
3508-epoch2: 0.652368
```

### deberta_v2

python train.py --fold 1 --model_name microsoft/deberta-v2-xlarge
