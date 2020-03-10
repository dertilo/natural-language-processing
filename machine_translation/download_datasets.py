import os

from util import data_io

if __name__ == "__main__":
    base_path = "http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses"
    HOME = os.environ["HOME"]
    file = "en-pt_br.txt.zip"
    target_folder = HOME + "/data/parallel_text_corpora/PORTUGUESE_ENGLISH"
    os.makedirs(target_folder, exist_ok=True)
    data_io.download_data(base_path, file, target_folder, unzip_it=True, verbose=True)
