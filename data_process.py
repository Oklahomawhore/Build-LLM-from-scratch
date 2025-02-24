# Given a corpus, we need to implement a tokenizer to tokenize the corpus into tokens
# and use the tokenizer to preprocess the corpus and createa a dataset
# then we implement a dataloader to load the dataset
import re

corpus = """

This is a test corpus.
The rising sun is a symbol of genesis and rebirth. When the sun rises from the east, it means the new day is coming.
Life on earch cannot live without the sun, it is the source of life.
Tha also applies to human beings, we need the sun to provide us with light and warmth.
A great man once said: "The sun is the only thing that can light up the darkness in our hearts, today!"
I agree with him, the sun is so important to us that -- without it, we would cease to exist.
"""


class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {text:id for id, text in vocab.items()}
    
    def encode(self, text):

        splitted = re.split(r'([,.;:?_!"()\']|--|\s)',text)
        splitted = [s.strip() for s in splitted if s.strip()]
        return [self.str_to_int[s] for s in splitted]

    def decode(self, ids):
        preprocessed = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.;:?_!"()\'])', r'\1', preprocessed)
        return text
    

if __name__ == '__main__':
    all_words = re.split(r'([.,;:?_!"()\']|--|\s)', corpus)
    all_words = [word.strip() for word in all_words if word.strip()]

    vocab = sorted(set(all_words))

    vocab = {word: id for id, word in enumerate(vocab)}
    
    
    tokenizer = Tokenizer(vocab)

    print(tokenizer.encode("today is a new day!"))
    print(tokenizer.decode(tokenizer.encode("today is a new day!")))