from t5.data import dataset_providers
import os
import sys
sys.path.append("...")
import time
from t5 import data
from t5.data import dataset_providers
from t5.data import preprocessors
from t5.evaluation import metrics
from configs import summarization_t5 as t5_base
import functools
from rouge_utils import rouge_top_beam
import lazyprofiler.GetStats as gs
from utils.t5x_utils.training import train
import tensorflow as tf

data_dir = "../../../data/datasets/summarization/"
train_file = "train.tsv"
validation_file = "validation.tsv"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
fine_tuning_cfg = t5_base.get_config()

TaskRegistry = dataset_providers.TaskRegistry
TaskRegistry.remove("summarization_task")
TaskRegistry.add(
  "summarization_task",
  dataset_providers.TextLineTask,
  split_to_filepattern = {
      "train": os.path.join(data_dir, train_file),
      "validation": os.path.join(data_dir, validation_file)
  },
  skip_header_lines = 1,
  text_preprocessor = preprocessors.preprocess_tsv,
  metric_fns=[functools.partial(
      rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)]
)
train(model_dir="../../../data/model_data/summarization/best_checkpoint/", config=fine_tuning_cfg)
