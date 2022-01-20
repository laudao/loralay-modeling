import json
import os
import datasets
from tqdm import tqdm
from lla_summ.data.utils import normalize_bbox

_ARTICLE_ID = "article_id"
_ARTICLE_WORDS = "article_words"
_ARTICLE_BBOXES = "article_bboxes"
_ARTICLE_NORM_BBOXES = "article_norm_bboxes"
_ABSTRACT = "abstract"


class ReformArxivSummarizationConfig(datasets.BuilderConfig):
    """BuilderConfig for ReformArxivSummarization."""
    def __init__(self, **kwargs):
        """BuilderConfig for ArxivSummarization.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ReformArxivSummarizationConfig, self).__init__(**kwargs)

class ReformArxivSummarizationDataset(datasets.GeneratorBasedBuilder):
    """ArxivSummarization Dataset."""
    
    BUILDER_CONFIGS = [
        ReformArxivSummarizationConfig(
            name="document",
            version=datasets.Version("1.0.0"),
            description="Reformulated arXiv dataset for summarization",
        ),
    ]

    def _info(self):
        # Should return a datasets.DatasetInfo object
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    _ARTICLE_ID: datasets.Value("string"),
                    _ARTICLE_WORDS: datasets.Sequence(datasets.Value("string")),
                    _ARTICLE_BBOXES: datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    _ARTICLE_NORM_BBOXES: datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    _ABSTRACT: datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.manual_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "data_path": os.path.join(data_dir, "train"),
                    "abstract_path": os.path.join(data_dir, "train_abstracts.txt")
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "data_path": os.path.join(data_dir, "val"),
                    "abstract_path": os.path.join(data_dir, "val_abstracts.txt")
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "data_path": os.path.join(data_dir, "test"),
                    "abstract_path": os.path.join(data_dir, "test_abstracts.txt")
                }
            ),
        ]


    
    def _generate_examples(self, data_path, abstract_path):
        """Generate ReformArxivSummarization examples."""
        filenames = sorted(os.listdir(data_path))

        guid = 0
        with open(abstract_path, 'r') as abstract_file:
            for line in tqdm(abstract_file, total=len(filenames), desc=f"Reading files in {data_path}"):
                guid += 1
                item = json.loads(line)
                fname = item["id"] + ".txt"
                filepath = os.path.join(data_path, fname)
                
                words  = []
                bboxes = []
                norm_bboxes = []

                with open(filepath, encoding="utf-8") as f:
                    for line in f:
                        splits = line.split("\t")
                        word = splits[0]
                        bbox = splits[1:5]
                        bbox = [int(b) for b in bbox]
                        page_width, page_height = int(splits[5]), int(splits[6])
                        norm_bbox = normalize_bbox(bbox, (page_width, page_height))

                        words.append(word)
                        bboxes.append(bbox)
                        norm_bboxes.append(norm_bbox)

                assert len(words) == len(bboxes)
                assert len(bboxes) == len(norm_bboxes)

                yield guid, {
                        _ARTICLE_ID: item["id"],
                        _ARTICLE_WORDS: words, 
                        _ARTICLE_BBOXES: bboxes, 
                        _ARTICLE_NORM_BBOXES: norm_bboxes,
                        _ABSTRACT: item["abstract"]
                    }
