from nltk.grammar import CFG
from nltk.parse.generate import generate, demo_grammar

#demo_grammar = """
#  S -> NP VP
#  NP -> N
#  PP -> NP
#  VP -> '+' NP | '-' NP | '*' PP
#  N -> '5' | '6' | '7'
#  P -> '+' | '-'
#"""

demo_grammar = """
  S -> CP SP IP IP
  SP -> 'I' | 'you'
  CP -> 'before' | 'during' | 'after' | 'if'
  IP -> 'brush' NP | 'run' | 'fly' | 'clean' | 'sleep' | 'see' NP
  NP -> 'teeth' | SP
"""


print('Generating the first %d sentences for demo grammar:' % (5,))
print(demo_grammar)
grammar = CFG.fromstring(demo_grammar)
for n, sent in enumerate(generate(grammar, n=150), 1):
    print('%3d. %s' % (n, ' '.join(sent)))

