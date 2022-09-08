import sys
sys.path.append("..")
from t5.data import dataset_providers
import os
from t5.data import dataset_providers
from t5.data import preprocessors
from configs import summarization_T5 as t5_base
import functools
from utils.rouge_utils import rouge_top_beam
from utils.T5X_utils.test import test
import os
import tensorflow as tf

data_dir = "../../data/datasets/"
train_file = "train_sum.tsv"
test_file = "test_sum.tsv"

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
# Number of steps to take during evaluation.
# i.e., 3438 / 16  (batch_size)
fine_tuning_cfg.beam_size = 1
fine_tuning_cfg.step_offset = 1_126_792

TaskRegistry = dataset_providers.TaskRegistry

# Removes homonymous tasks if present
# Prevents errors at the second run
TaskRegistry.remove("summarization_task")

TaskRegistry.add(
  "summarization_task",
  dataset_providers.TextLineTask,
  split_to_filepattern = {
      "train": os.path.join(data_dir, train_file),
      "validation": os.path.join(data_dir, test_file)
  },
  skip_header_lines = 1,
  text_preprocessor = preprocessors.preprocess_tsv,
  metric_fns=[functools.partial(
      rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)] # rouge1/2/L
)
test(task_name="summarization_task", model_dir="../data/model_data/summarization/model_checkpoints/", config=fine_tuning_cfg, output_prediction_postfix="sum")