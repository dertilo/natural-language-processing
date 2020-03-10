import os

from fairseq.models.fconv import FConvModel


def run_interactive(model_path, model_file_name, data_path):
    model = FConvModel.from_pretrained(
        model_path,
        checkpoint_file=model_file_name,
        data_name_or_path=model_path + "/" + data_path,
    )

    while True:
        text = input("input:  ")
        translated = model.translate(text, beam=5)
        translated = (
            translated.replace(" @-@ ", "-").replace("@@ ", "").replace(" &apos;", "'")
        )
        print(translated)


if __name__ == "__main__":
    HOME = os.environ["HOME"]
    run_interactive(HOME + "/MT/portuguese_english", "en_pt_1.pt", "pt_en")
