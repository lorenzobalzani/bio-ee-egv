from t5.data import dataset_providers

import os
import sys
sys.path.append("..")
import time
from t5 import data
from t5.data import dataset_providers
from t5.data import preprocessors
from t5.evaluation import metrics
from configs import EE_T5 as t5_base
import functools
from utils.rouge_utils import rouge_top_beam
import lazyprofiler.GetStats as gs
from utils.T5X_utils.training import train
import os
import tensorflow as tf

data_dir = "../../data/datasets/"
train_file = "train_ee.tsv"
validation_file = "validation_ee.tsv"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable Tokenizer Parallelism warning for HuggingFace-transformers models
#https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# set Tf to not preallocate memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# Load experiment config
fine_tuning_cfg = t5_base.get_config()

TaskRegistry = dataset_providers.TaskRegistry

# Removes homonymous tasks if present
# Prevents errors at the second run
TaskRegistry.remove("event_extraction_task")

TaskRegistry.add(
  "event_extraction_task",
  dataset_providers.TextLineTask,
  split_to_filepattern = {
      "train": os.path.join(data_dir, train_file),
      "validation": os.path.join(data_dir, validation_file)
  },
  skip_header_lines = 1,
  text_preprocessor = preprocessors.preprocess_tsv,
  metric_fns=[functools.partial(
      rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)] # rouge1/2/L
)
train(model_dir="../../data/model_data/event-extraction/model_checkpoints/", config=fine_tuning_cfg)