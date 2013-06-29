Feature Hashing for Large Scale Multitask Learning
==================================================

Weinberger, Dasgupta, Langford, Smola, Attenberg (ICML 2009)

Presented at ILPS Reading Group by: Lars Buitinck

----

Machine learning with symbolic features
---------------------------------------

* Learning algorithms formulated on vectors/matrices
* NLP/IR applications: text
* Features are initially strings
* Need to **vectorize**

----

Vectorizing
-----------

* Naive way, step 1: learn a dictionary

.. sourcecode:: python

    feature_to_index = {}
    i = 0
    for s in samples:
        for f in features(s):
            if f not in feature_to_index:
                feature_to_index[f] = i
                i += 1

* Can create a very large dict
* Specialized string tables (tries) help, but still linear memory use

----

Vectorizing
-----------

* Naive way, step 2: vectorize a sample
* Assume sample is a list of (name, value) pairs
* Typically values are frequencies or booleans

.. sourcecode:: python

    for f, v in sample:
        vector[feature_to_index[f]] = v

----

A problem with the naive algorithm
----------------------------------

* In true online settings, feature set keeps growing
* Sometimes intentionally: when spam filter has learned high w(v1agra),
  spammers invent new misspellings, exploit Unicode, etc.

----

"Hashing trick" 0.9
-------------------

* Idea: hash the strings into the indices directly
* The column index of a feature ``f`` is ``h(f) mod n``
* In case of collision, add values (or OR them)
* Dimensionality reduction for free (just set ``n`` low)
* Ganchev and Drezde (2008) showed this to work well
* But collisions increase with decreasing ``n``

----

Second hash function
--------------------

* Let ``?`` be a hash function with range {-1, 1}
* Feature to set is still ``h(f) mod n``
* But multiply its value by ``?(f)``:

.. sourcecode:: python

    for f, v in sample:
        vector[h(f) % n] += ?(f) * v

* (Always add in case of collision)

----

Second hash function's purpose
------------------------------

* Suppose we're dealing with boolean features
* Collision between f1 and f2, both true

========= ========= =================
``?(f1)`` ``?(f2)`` ``?(f1) + ?(f2)``
========= ========= =================
-1        -1        -2
-1         1         0
 1        -1         0
 1         1         2
========= ========= =================

* 50% chance of resolving the collision!

----

Other nice properties
---------------------

* Expected value in each column is zero
* So, data is *centered* for free
* This is what SVMs and other learning algorithms want

----

Feature conjunctions
--------------------

* Sometimes the baseline features aren't good enough
  (data not linearly separable)
* Want quadratic features, i.e. products: ``f1f2`` = ``v1`` × ``v2``
* (Products of booleans are logical conjunctions)
* E.g., product of "query contains 'ir'" and "URL contains 'ilps'"

----

Feature conjunctions: the expensive way
---------------------------------------

* Full expansion of feature space leads to quadratic blowup,
  ½n × (n-1) = O(n²) memory usage
* Kernels do this faster, but kernel learners scale badly
* All kinds of algorithms/heuristics proposed

----

Hashing trick 2.0
-----------------
