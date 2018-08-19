# 10th place solution to the [2nd YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m-2018)
The code is based on:
1. https://github.com/google/youtube-8m
2. Winning solution of the first YouTube-8M Video Understanding Challenge at https://github.com/antoine77340/Youtube-8M-WILLOW
3. Two tensorflow source code files [`layers.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py) and [`core.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/core.py)

Our main contributions:
1. Extend tensorflow's `layers.fully_connected` function and `core.Dense` class to support float16 dtype
2. Ensemble multiple single models in same tensorflow graph for easy inference

See our paper at ___ for details.

## Commands to train a single model in float32

### gcloud set up 
```
BUCKET_NAME=gs://${USER}_yt8m_train_bucket

# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-central1 $BUCKET_NAME
```

### Train

```
# model hypterparameters
cs=100
hs=800
batch_size=160

JOB_NAME=yt8m_train_c${cs}h${hs}_${batch_size}; gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord,gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
--train_dir=$BUCKET_NAME/c${cs}h${hs}_${batch_size} \
--frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=${batch_size} --base_learning_rate=0.0001 --netvlad_cluster_size=${cs} --netvlad_hidden_size=${hs} --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --max_steps=500000 --export_model_steps=5000 --num_epochs=10
```

### Evaluate several single checkpoints in a loop
```
cs=10
hs=400
batch_size=160
CKPTS=(170010 180010 190010 200010 210010 220010)

for CKPT in ${CKPTS[@]}; do 
	JOB_NAME=yt8m_eval_c${cs}h${hs}_${CKPT}_$(date +%HH%MM%SS); gcloud --verbosity=debug ml-engine jobs \
	submit training $JOB_NAME \
	--package-path=youtube-8m --module-name=youtube-8m.eval \
	--staging-bucket=$BUCKET_NAME --region=us-central1 \
	--config=youtube-8m/cloudml-gpu.yaml \
	-- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
	--model=NetVLADModelLF \
	--train_dir=$BUCKET_NAME/c${cs}h${hs}_${batch_size} --frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=1024 --base_learning_rate=0.0001 --netvlad_cluster_size=${cs} --netvlad_hidden_size=${hs} --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --run_once=True --checkpoint=model.ckpt-${CKPT}.index
done
```

### Average all checkpoints in a range
```
cs=10
hs=400
batch_size=160
ckpt_start=150010
ckpt_end=220010

JOB_NAME=yt8m_avgckpt_c${cs}h${hs}_${ckpt_start}_${ckpt_end}
gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.avg_checkpoints \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --checkpoints_start_end="${ckpt_start},${ckpt_end}" --prefix="${BUCKET_NAME}/c${cs}h${hs}_${batch_size}/model.ckpt-" --output_path="$BUCKET_NAME/c${cs}h${hs}_${batch_size}/avg_c${cs}h${hs}_${ckpt_start}_${ckpt_end}.ckpt"
```

### Average select checkpoints
```
cs=32
hs=512
batch_size=160
RANGE_NAME=122304_etc_19
JOB_NAME=yt8m_avgckpt_c${cs}h${hs}_${RANGE_NAME}
gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.avg_checkpoints \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --prefix="${BUCKET_NAME}/c${cs}h${hs}_${batch_size}/model.ckpt-" --output_path="$BUCKET_NAME/c${cs}h${hs}_${batch_size}/avg_c${cs}h${hs}_${RANGE_NAME}.ckpt" --checkpoints=\
122304,125397,133210,137879,140959,145600,150210,153324,156475,159900,164900,168029,172960,176234,181138,186032,189900,202410,207267
```

### Evaluate an averaged checkpoint
```
cs=32
hs=512
batch_size=160
checkpoint=122304_etc_19
JOB_NAME=yt8m_eval_c${cs}h${hs}_${checkpoint}; gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
--train_dir=$BUCKET_NAME/c${cs}h${hs}_${batch_size} \
--frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=1024 --base_learning_rate=0.0001 --netvlad_cluster_size=${cs} --netvlad_hidden_size=${hs} --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --run_once=True \
--checkpoint=avg_c${cs}h${hs}_${checkpoint}.ckpt-0.index
```



## Steps to ensemle/float16:

### Step 0: gcloud setup

See https://github.com/google/youtube-8m/blob/master/README.md#running-on-googles-cloud-machine-learning-platform for gcloud setup
```
# some environment variables
ENSEMBLE_DIR=ensemble_YHLS
MODEL1_PARAM=(24 1440)
MODEL2_PARAM=(32 1024)
MODEL3_PARAM=(16 800)
MODEL4_PARAM=(32 512)

# here use inference model rather than avg of train. Typically named inference_model_0
MODEL1_CKPT="${BUCKET_NAME}/single_models/c24h1440-inference_model_0"
MODEL2_CKPT="${BUCKET_NAME}/single_models/c32h1024-inference_model_0"
MODEL3_CKPT="${BUCKET_NAME}/single_models/c16h800-inference_model_0"
MODEL4_CKPT="${BUCKET_NAME}/single_models/c32h512-inference_model_0"

MIXING_WEIGHTS=(0.25 0.25 0.25 0.25)

INFERENCE_MODEL_NAME="inference_model_ensemble_YHLS"
```

### Step 1: Train for 10 steps and generate ensemble checkpoint
```
JOB_NAME=yt8m_train_ensemble_${ENSEMBLE_DIR}_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord,gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' --model=NetVLADModelLF \
--train_dir=$BUCKET_NAME/${ENSEMBLE_DIR} --frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=10 --base_learning_rate=0.00003 \
--netvlad_cluster_size=${MODEL1_PARAM[0]} --netvlad_hidden_size=${MODEL1_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[0]} \
--netvlad_cluster_size=${MODEL2_PARAM[0]} --netvlad_hidden_size=${MODEL2_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[1]} \
--netvlad_cluster_size=${MODEL3_PARAM[0]} --netvlad_hidden_size=${MODEL3_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[2]} \
--netvlad_cluster_size=${MODEL4_PARAM[0]} --netvlad_hidden_size=${MODEL4_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[3]} \
--moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --max_steps=10 --export_model_steps=2000 --ensemble_num=4 --float16_flag=True 
```

### Step 2: From the 10-step checkpoint, run eval.py to get inference_model_10 checkpoint (`32*n+1` tensors in float16, untrained) and the meta file (the relationships between these `32*n+1` tensors)
```
JOB_NAME=yt8m_train_ensemble_${ENSEMBLE_DIR}_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
--model=NetVLADModelLF \
--train_dir=$BUCKET_NAME/${ENSEMBLE_DIR} --frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=500 --base_learning_rate=0.0001 \
--netvlad_cluster_size=${MODEL1_PARAM[0]} --netvlad_hidden_size=${MODEL1_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[0]} \
--netvlad_cluster_size=${MODEL2_PARAM[0]} --netvlad_hidden_size=${MODEL2_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[1]} \
--netvlad_cluster_size=${MODEL3_PARAM[0]} --netvlad_hidden_size=${MODEL3_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[2]} \
--netvlad_cluster_size=${MODEL4_PARAM[0]} --netvlad_hidden_size=${MODEL4_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[3]} \
--moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --run_once=True --ensemble_num=4 --float16_flag=True --create_meta_only
```
### Step 3: Fill in the values of the `32*n+1` float16 tensors using weights trained in the four float32 single models, save as `${INFERENCE_MODEL_NAME}`
```
JOB_NAME=yt8m_train_ensemble_${ENSEMBLE_DIR}_$(date +%Y%m%d_%H%M%S);

gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.overwrite_float16_ckpt \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_dir=$BUCKET_NAME/${ENSEMBLE_DIR} --inference_model=inference_model_10 \
--out_name=${INFERENCE_MODEL_NAME} \
--checkpoints="${MODEL1_CKPT},${MODEL2_CKPT},${MODEL3_CKPT},${MODEL4_CKPT}"
```

### Step 4: Use `inference_model_float16_mean_ensemble-0` to evaluate, tune ensemble_wts
```
MIXING_WEIGHTS=(0.39 0.26 0.21 0.14)
INFERENCE_MODEL_NAME="inference_model_ensemble_YHLS"

JOB_NAME=yt8m_train_ensemble_${ENSEMBLE_DIR}_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
--model=NetVLADModelLF \
--train_dir=$BUCKET_NAME/${ENSEMBLE_DIR} --frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=1024 --base_learning_rate=0.0001 \
--netvlad_cluster_size=${MODEL1_PARAM[0]} --netvlad_hidden_size=${MODEL1_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[0]} \
--netvlad_cluster_size=${MODEL2_PARAM[0]} --netvlad_hidden_size=${MODEL2_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[1]} \
--netvlad_cluster_size=${MODEL3_PARAM[0]} --netvlad_hidden_size=${MODEL3_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[2]} \
--netvlad_cluster_size=${MODEL4_PARAM[0]} --netvlad_hidden_size=${MODEL4_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[3]} \
--moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --run_once=True \
--ensemble_num=4 --float16_flag=True --checkpoint_file="${INFERENCE_MODEL_NAME}-0"
```

### Step 5: After finding best weights, set force_output_model_name FLAG, in order to generate `inference_model.data-00000-of-00001` and `.meta`, `.index` (because the official inference.py only take this name in tgz submission file). Record the local GAP to compare with public LB.
```
MIXING_WEIGHTS=(0.39 0.26 0.21 0.14)

JOB_NAME=yt8m_train_ensemble_${ENSEMBLE_DIR}_$(date +%Y%m%d_%H%M%S); 
gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
--model=NetVLADModelLF \
--train_dir=$BUCKET_NAME/${ENSEMBLE_DIR} --frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=1024 --base_learning_rate=0.0001 \
--netvlad_cluster_size=${MODEL1_PARAM[0]} --netvlad_hidden_size=${MODEL1_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[0]} \
--netvlad_cluster_size=${MODEL2_PARAM[0]} --netvlad_hidden_size=${MODEL2_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[1]} \
--netvlad_cluster_size=${MODEL3_PARAM[0]} --netvlad_hidden_size=${MODEL3_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[2]} \
--netvlad_cluster_size=${MODEL4_PARAM[0]} --netvlad_hidden_size=${MODEL4_PARAM[1]} --ensemble_wts=${MIXING_WEIGHTS[3]} \
--moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --run_once=True \
--ensemble_num=4 --float16_flag=True --checkpoint_file="${INFERENCE_MODEL_NAME}-0" \
--force_output_model_name=True
```

### Step 6: Download `inference_model.data-00000-of-00001` and `.meta`, `.index`, put the in tgz together with `model_flags.json`

### Step 7: Use the official infernce.py and tgz file to inference

[Note that the official infernce.py has some problem untar the tgz, so we ran this step locally]
```
python inference.py --input_model_tgz=/path_to_sub/YHLS8821/YHLS8821.tgz --output_file=/path_to_sub/YHLS8821/subYHLS8821.csv --input_data_pattern="/path_to_data/frame/test*.tfrecord" --batch_size=512
```


