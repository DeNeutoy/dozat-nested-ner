from nested_ner.genia_reader import GeniaNestedNerReader
from nested_ner.genia_v3_reader import GeniaV3NestedNerReader


class TestNestedNerReader:
    def test_read_from_file(self):
        reader = GeniaNestedNerReader()
        data_path = "tests/fixtures/data.iob2"
        instances = reader.read(data_path)

        assert len(instances) == 2

        fields = instances[0].fields
        expected_tokens = [
            "A",
            "new",
            "B",
            "-",
            "cell",
            "-",
            "specific",
            "enhancer",
            "element",
            "has",
            "been",
            "identified",
            "3",
            "'",
            "of",
            "E4",
            "and",
            "the",
            "octamerlike",
            "motifs",
            "in",
            "the",
            "human",
            "immunoglobulin",
            "heavy",
            "-",
            "chain",
            "gene",
            "enhancer",
            ".",
        ]
        assert [t.text for t in fields["tokens"].tokens] == expected_tokens

        fields = instances[1].fields
        expected_tokens = [
            "Tandem",
            "copies",
            "of",
            "this",
            "67",
            "-",
            "bp",
            "MnlI",
            "-",
            "AluI",
            "fragment",
            ",",
            "when",
            "fused",
            "to",
            "the",
            "chloramphenicol",
            "acetyltransferase",
            "gene",
            "driven",
            "by",
            "the",
            "conalbumin",
            "promoter",
            ",",
            "stimulated",
            "transcription",
            "in",
            "B",
            "cells",
            "but",
            "not",
            "in",
            "Jurkat",
            "T",
            "cells",
            "or",
            "HeLa",
            "cells",
            ".",
        ]
        assert [t.text for t in fields["tokens"].tokens] == expected_tokens


    def test_v3_reader(self):
        reader = GeniaV3NestedNerReader()
        data_path = "tests/fixtures/data_v3.txt"
        instances = reader.read(data_path)

        print(instances[0])
        print(instances[1])
