from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Input text
text = """
Artificial Intelligence is a branch of computer science that aims to create
machines capable of intelligent behavior. It is widely used in healthcare,
education, finance, and many other industries. AI systems can learn from data,
recognize patterns, and make decisions with minimal human intervention.
"""

# Parser and tokenizer
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Summarizer
summarizer = LsaSummarizer()

# Generate summary (3 lines)
summary = summarizer(parser.document, 3)

print("SUMMARY:\n")
for sentence in summary:
    print(sentence)