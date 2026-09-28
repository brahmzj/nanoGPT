"""
Morpheus: a tiny, low-energy language model that grows up from the ABCs and 1 2 3s.

Instead of swallowing a mountain of internet text, Morpheus is raised on a
curriculum that is generated procedurally (the "dataset" is a few KB of code),
advances a stage only once it passes that stage's exam, and consolidates what it
learned in a final "dream" phase. The trained brain can be squeezed to int8/int4
and run with nothing but numpy.

    python -m morpheus train      # raise Morpheus from scratch
    python -m morpheus chat       # talk to it
    python -m morpheus exam       # report card
    python -m morpheus export     # compress to a .morph file
    python -m morpheus stats      # size / compute / energy budget
"""

__version__ = "0.1.0"
