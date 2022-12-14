from t5.data import dataset_providers
import os
import sys
sys.path.append("...")
import time
from t5 import data
from t5.data import dataset_providers
from t5.data import preprocessors
from t5.evaluation import metrics
from configs import mtl_t5 as t5_base
import functools
from rouge_utils import rouge_top_beam
import lazyprofiler.GetStats as gs
from utils.t5x_utils.training import train
import tensorflow as tf

data_dir = "../../../data/datasets/"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
fine_tuning_cfg = t5_base.get_config()

TaskRegistry = dataset_providers.TaskRegistry
TaskRegistry.remove("event_extraction_task")
TaskRegistry.add(
  "event_extraction_task",
  dataset_providers.TextLineTask,
  split_to_filepattern = {
      "train": os.path.join(data_dir, "biot2e","train.tsv"),
      "validation": os.path.join(data_dir, "biot2e", "validation.tsv"),
  },
  skip_header_lines = 1,
  text_preprocessor = preprocessors.preprocess_tsv,
  metric_fns=[functools.partial(
      rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)]
)

TaskRegistry.remove("event_graph_verbalization_task")
TaskRegistry.add(
  "event_graph_verbalization_task",
  dataset_providers.TextLineTask,
  split_to_filepattern = {
      "train": os.path.join(data_dir, "bioe2t", "train.tsv"),
      "validation": os.path.join(data_dir, "bioe2t", "validation.tsv"),
  },
  skip_header_lines = 1,
  text_preprocessor = preprocessors.preprocess_tsv,
  metric_fns=[functools.partial(
      rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)]
)

TaskRegistry.remove("summarization_task")
TaskRegistry.add(
  "summarization_task",
  dataset_providers.TextLineTask,
  split_to_filepattern = {
      "train": os.path.join(data_dir, "summarization", "train.tsv"),
      "validation": os.path.join(data_dir, "summarization", "validation.tsv")
  },
  skip_header_lines = 1,
  text_preprocessor = preprocessors.preprocess_tsv,
  metric_fns=[functools.partial(
      rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)]
)

MixtureRegistry = dataset_providers.MixtureRegistry
MixtureRegistry.remove("mtl_mixture")
MixtureRegistry.add(
    "mtl_mixture",
    ["summarization_task", "event_graph_verbalization_task", "event_extraction_task"],
    1.0)

train(model_dir="../../../data/model_data/mtl/best_checkpoint/", config=fine_tuning_cfg)
