# Regularized Contrastive Learning for Few-shot Bioacoustic Sound Event Detection
Authors: Ilyass Moummad, Romain Serizel, Nicolas Farrugia
---

This is the implementation of our [paper](https://arxiv.org/abs/2309.08971) accepted at ICASSP 2024\
We improve over our previous [work](https://github.com/ilyassmoummad/dcase23_task5_scl) that ranked 2nd in the [DCASE 2023 Challenge Task5](https://dcase.community/challenge2023/task-few-shot-bioacoustic-event-detection-results) by adding a regularization term to the training loss, and by improving the inference strategy

```args.py```: contains default values for arguments

## Data
To create the spectrograms of the training set:\
```create_train.py```: with argument ```--traindir``` for the folder containing the training datasets

## Training
To train the feature extractor :\
```train.py```: with arguments ```--traindir``` (the same as above)
```--method```: optional argument for the pretraining method ```scl``` for SupCon and ```ssl``` for SimCLR

## Evaluation
To evaluate using finetuning:\
```eval_finetune```: with arguments ```--valdir``` for the folder containing the validation datasets\
To evaluate without finetuning (for faster inference):\
```eval_nofinetune```: with arguments ```--valdir``` for the folder containing the validation datasets

To get the scores:\
```evaluation.py``` : with arguments ```--pred_file``` for the predictions csv file created by the eval script (the file is in : traindir/../../outputs/eval.csv'), ```--ref_files``` for the path of validation datasets (same as ```--valdir```), and ```--save_path``` for the folder where to save the json file containing the scores

If you any question or a problem with the code/results do not hesitate to mail me on : ilyass.moummad@imt-atlantique.fr or open an issue on this repository, I am very responsive.

---
### To cite our paper
```
@misc{moummad2023regularized,
      title={Regularized Contrastive Pre-training for Few-shot Bioacoustic Sound Detection}, 
      author={Ilyass Moummad and Romain Serizel and Nicolas Farrugia},
      year={2023},
      eprint={2309.08971},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
