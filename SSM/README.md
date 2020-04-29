### step1
- `# python train.py --s_dims 64`
  - -> "./models/<timestamp1\>/epoch-00010/*.pt"

### step2
- `# python train.py --s_dims 64 128 --load_name <timestamp1> --load_epoch 10 --static_hierarchy 0`
  - -> "./models/<timestamp2\>/epoch-00100/\*.pt"

### step3
- `# cp ./models/<timestamp1>/epoch-00010/*pt ./models/<timestamp2>/epoch-00100/`
- `# python train.py --s_dims 64 128 256 --load_name <timestamp2> --load_epoch 100 --static_hierarchy 0 1`
  - -> "./models/<timestamp3\>/epoch-01000/\*.pt"


### (option) resume training
- (step2)
  - `# cp ./models/<timestamp1>/epoch-00010/*pt ./models/<timestamp2>/epoch-00100/`
  - `# python train.py --s_dims 64 128 --load_name <time_stamp2> --load_epoch 100 --static_hierarchy  --resume `
    - DO NOT FORGET **--static_hierarchy** and **--resume** OPTION!!!