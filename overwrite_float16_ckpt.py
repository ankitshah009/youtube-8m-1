# Copyright 2018 Bo Liu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np  
import six
import os
from tensorflow import flags
FLAGS = flags.FLAGS

if __name__ == "__main__":

  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                    "The directory to save the model files in.")
  flags.DEFINE_string("inference_model", "",
                    "The inference model to be filled")
  flags.DEFINE_string("out_name","inference_model_float16_mean_ensemble",
                      "out inference model name")
  flags.DEFINE_string("checkpoints","",
                      "the checkpoints of each individual models")

  model_dir = FLAGS.train_dir
  inference_model = FLAGS.inference_model
  # ckpt to be filled
  checkpoint = os.path.join(model_dir, inference_model)

  reader0 = tf.contrib.framework.load_checkpoint(checkpoint)    
  var_list = tf.contrib.framework.list_variables(checkpoint)
  len(var_list)
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if name not in ["global_step"]:
      var_values[name] = np.zeros(shape)


  # original (float32) models
  checkpoints = FLAGS.checkpoints.split(',')
   
  for i in range(len(checkpoints)):
      checkpoint = checkpoints[i]
      reader = tf.contrib.framework.load_checkpoint(checkpoint)
      for name in var_values:
          if i==0:
              if 'model' in name:
                  continue
          elif 'model'+str(i) not in name:
              continue

          tensor0 = reader0.get_tensor(name)
          #print(name,tensor0.dtype)
          tensor = reader.get_tensor(name.replace('model' +str(i)+ '/',''))
          var_dtypes[name] = tensor0.dtype 
          var_values[name] += tensor


  tf_vars = [
      tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
      for v in var_values
  ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)


  saver = tf.train.Saver(tf.all_variables())

  config = tf.ConfigProto(
      device_count = {'GPU': 0}
  )
  # Build a model consisting only of variables, set them to the average values.
  with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(var_values)):
      sess.run(assign_op, {p: value})
    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, os.path.join(model_dir,FLAGS.out_name), global_step=global_step)



  ## check results
  # checkpoint0= checkpoints[0]
  # checkpoint1 = os.path.join(model_dir,'inference_model_float16_mean_ensemble-0')
  # reader0 = tf.contrib.framework.load_checkpoint(checkpoint0)
  # reader1 = tf.contrib.framework.load_checkpoint(checkpoint1)
  # for (name,shape) in var_list:
  #     if name in ['tower/gating_prob_weights','tower/gates/weights','tower/experts/weights']:
  #         tensor0 = reader0.get_tensor(name)
  #         tensor1 = reader1.get_tensor(name)
  #         print(name,tensor0,tensor1)
      