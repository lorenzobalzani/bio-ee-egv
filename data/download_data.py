import gdown
import zipfile

def download_file(id, filename):
    url = 'https://drive.google.com/uc?id=' + id
    gdown.download(url, filename, quiet=False)

def download_folder(id, output):
    gdown.download_folder('https://drive.google.com/drive/folders/' + id, output=output, quiet=False, remaining_ok=True)

def download_summarization_dataset():
    download_file('1-4I2nWfqDu17bVRFZFSSBRzZiM-Jeg1e', './datasets/summarization/train.tsv')
    download_file('12-GA-rUn-bKicdtok1mZFggTwVh8zq8S', './datasets/summarization/validation.tsv')
    download_file('1jvc-ho9r3DKVsOaDpxkuwhgJSsG167oL', './datasets/summarization/test.tsv')

def download_EE_predictions():
    download_folder('1lWkgWntpJ-7pfNuCmcSaqIEZjbcSSlsf', './model_data/ee/t5x/')

def download_model_checkpoints():
    download_folder('15nNVf-MxojniadUPlPL7lLtB4C_1pB3i', './model_data/')
    download_folder('1TWKnpRMz0UrFc9ru401MLZDHZLVm2oap', './model_data/')
    download_folder('1c6WBOEBmTegGkGSPh2_TMkk4Sc1v9U4H', './model_data/')
    download_folder('1OWfvJJWlo7nHTnh0sZgbbzGa2x_EfJTF', './model_data/')

download_summarization_dataset()
download_model_checkpoints()
download_EE_predictions()
