# MultipleGPUSqeezeNet
SqueezeNet Estimator Multiple GPU Code


```
python LP_main.py --model_dir '/path/to/checkpoint/ckpt' --job-dir '/path/to/all/the/files/' --train_csv '/path/to/trainCsv/Traindata22.csv' --eval_csv '/path/to/testcsv/Testdata.csv' --num_gpu=2 --variable_strategy=GPU
```


To change Number of class 
  1. Change in model file (2 places) in LP_model.py
  2. Change in One hot encoding in LP_DataSets.py
