from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, AdjacencyField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans, TypedStringSpan

@DatasetReader.register('genia')
class GeniaNestedNerReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:


        with open(file_path, 'r') as lines:

            words = []
            bio1 = []
            bio2 = []
            bio3 = []
            bio4 = []
            for line in lines:

                if not line.strip() and words:
                    # End of a sentence, process it.
                    spans = []
                    spans.extend(bio_tags_to_spans(bio1))
                    spans.extend(bio_tags_to_spans(bio2))
                    spans.extend(bio_tags_to_spans(bio3))
                    spans.extend(bio_tags_to_spans(bio4))
                    
                    yield self.text_to_instance(" ".join(words), spans)
                    words = []
                    bio1 = []
                    bio2 = []
                    bio3 = []
                    bio4 = []
                else:
                    word, *bio = line.strip().split("\t")
                    words.append(word)
                    bio1.append(bio[0])
                    bio2.append(bio[1])
                    bio3.append(bio[2])
                    bio4.append(bio[3])

        # Check if we have one left over
        if words:
            spans = []
            spans.extend(bio_tags_to_spans(bio1))
            spans.extend(bio_tags_to_spans(bio2))
            spans.extend(bio_tags_to_spans(bio3))
            spans.extend(bio_tags_to_spans(bio4))
            
            yield self.text_to_instance(" ".join(words), spans)

    def text_to_instance(self, text: str, labeled_spans: List[TypedStringSpan] = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)


        text_field = TextField(tokens, self.token_indexers)

        fields = {
            'tokens': text_field,
            }
        meta = {
            "tokens": [t.text for t in tokens]
        }

        if labeled_spans is not None:
            if labeled_spans == []:
                labels = []
                spans = []
            else:
                labels, spans = zip(*labeled_spans)
            adjacency = AdjacencyField(spans, text_field, labels)
            meta["gold"] = labeled_spans
            fields["span_labels"] = adjacency

        fields["metadata"] = MetadataField(meta)

        return Instance(fields)
