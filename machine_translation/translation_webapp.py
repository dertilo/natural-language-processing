import os
import sys
from wsgiref import simple_server

import falcon


from fairseq.models.fconv import FConvModel


class TranslateAPI:
    def __init__(self) -> None:
        HOME = os.environ["HOME"]
        model_path = HOME + "/MT/portuguese_english"
        self.en_pt = FConvModel.from_pretrained(
            model_path,
            checkpoint_file="en_pt_1.pt",
            data_name_or_path=model_path + "/" + "pt_en",
        )
        self.pt_en = FConvModel.from_pretrained(
            model_path,
            checkpoint_file="pt_en_1.pt",
            data_name_or_path=model_path + "/" + "pt_en",
        )

        super().__init__()

    def on_post(self, req, resp):
        lang, text = next(iter(req.media.items()))
        if lang == "en":
            translated = self.en_pt.translate(text, beam=5)
            translated = (
                translated.replace(" @-@ ", "-")
                .replace("@@ ", "")
                .replace(" &apos;", "'")
            )

            resp.media = {"pt": translated}
        elif lang == "pt":
            translated = self.pt_en.translate(text, beam=5)
            translated = (
                translated.replace(" @-@ ", "-")
                .replace("@@ ", "")
                .replace(" &apos;", "'")
            )
            resp.media = {"en": translated}
        else:
            raise NotImplementedError("unsuppoted language")

    def on_get(self, req, resp):
        resp.media = ["this is a translation service"]


if __name__ == "__main__":
    api = falcon.API()

    service = TranslateAPI()
    api.add_route("/", service)

    httpd = simple_server.make_server("0.0.0.0", 8888, api)
    print("server is running")
    sys.stdout.flush()
    httpd.serve_forever()
