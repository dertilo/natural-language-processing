import os, os.path
from pprint import pprint

from typing import List

from tqdm import tqdm
from util import data_io, util_methods
from whoosh import index

from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import StemmingAnalyzer


def build_index():
    schema = Schema(
        id=ID(stored=True),
        filename=ID(stored=True),
        story=TEXT(analyzer=StemmingAnalyzer(), stored=True,lang='en'),
    )
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    ix = index.create_in("indexdir", schema)
    data = data_io.read_json(
        os.environ["HOME"] + "/data/QA/coqa/" + "coqa-train-v1.0.json"
    )["data"]

    writer = ix.writer()
    for d in tqdm(data):
        writer.add_document(id=d["id"], filename=d["filename"], story=d["story"])
    writer.commit()


if __name__ == "__main__":
    # build_index()

    from whoosh.qparser import QueryParser

    ix = index.open_dir("indexdir")
    qp = QueryParser("story", schema=ix.schema)

    with ix.searcher() as s:
        while True:
            search_string = input("search for:")
            or_searche = ' OR '.join(search_string.split(' '))
            q = qp.parse(or_searche)
            print(q)
            results = s.search(q, limit=3)
            pprint([r['filename'] for r in results])
