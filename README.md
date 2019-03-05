# ICCAD19-Blind-Review

## Motivational Example

In the Motivation folder, we upload the checkpoints and logs during training the structures used in the motivational example.
One can test the accuracy of both architectures.

##### Usage of testing NAS without cutting
cd models

python ../main.py --ckpath nas15_lr_01_ep_300 -r -t

Note: the test accuracy will be 92.51 without cutting edges


##### Usage of testing NAS with cutting
cd models

python ../main.py --ckpath nas15_cut_lr_01_ep_300 -r -t -c

Note: test accuracy will be 92.19 after cutting some edges

## Requirements:

PyTorch 
