### generate vocabulary
```
onmt_build_vocab -config xxxxx.yaml  -n_sample -1
```
### make gpus available for onmt
```
export CUDA_VISIBLE_DEVICES=0,1
```
### start training
```
nohup onmt_train -config xxxx.yaml 2>&1 > my.log &  
```
### start generation
```
onmt_translate -model model_step_1000.pt -src src-test.txt -output pred_1000.txt -gpu 0 -verbose
```
