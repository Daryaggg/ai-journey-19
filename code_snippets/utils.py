from pymorphy2 import MorphAnalyzer


class MyMorph(MorphAnalyzer):
    def get_tag(self, word: str):
        return self.parse(word)[0].tag
