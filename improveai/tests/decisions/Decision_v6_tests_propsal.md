## class DecisionUtils

### Methods:

 - #### simple_context():
    
    - test returned value -> empty frozendict

## class Decision

### Constant wrapper
   - test if inheritance breaks @constant

### Constants

 - #### __PROTECTED_MEMBERS_ATTR_NAMES (protected member)
     - test if writable from the inside of Decision class
     - test if accessible from the outside
     - test if '__' prefixes each of __PROTECTED_MEMBERS_ATTR_NAMES`s elements 
     - test if Decision has this attr
 - #### PROTECTED_MEMBER_INIT_VAL
     - test if Decision has this attr
     - test if value is a proper one (currently None)


### Properties
 - #### variants
     - @getter test if accessible from the outside (should be)
     - @getter test if returns tuple of frozendicts (immutable)
     - @setter test if possible to set more than once per instance
     - @getter + @setter test if provided Iterable of dicts gets properly 
       converted into tuple of frozendicts
 - #### model
     - @getter TBD
     - @setter TBD
 - #### model_name
     - @getter test if accessible from the outside (should be)
     - @setter test if mutable from inside class
     - @setter test if possible to set more than once per instance
 - #### context
     - @getter test if accessible from the outside (should be)
     - @getter test if the returned value is a frozendict
     - @setter test if mutable from inside class
     - @setter test if possible to set more than once per instance
     - @getter + @setter test if provided dict is properly converted to frozendict
 - #### max_runners_up
     - @getter test if accessible from the outside (should be)
     - @getter test if the returned value is int 
     - @setter test if possible to set more than once per instance
 - #### memoized_scores
     - @getter test if accessible from the outside (should be)
     - @getter test if the returned value is tuple of floats
     - @setter test if possible to set more than once per instance
 - #### memoized_ranked
     - @getter test if accessible from the outside (should be)
     - @getter test if the returned value is tuple of frozendicts
     - @setter test if possible to set more than once per instance
 - #### memoized_top_runners_up
     - @getter test if accessible from the outside (should be)
     - @getter test if the returned value is tuple of frozendicts
     - @setter test if possible to set more than once per instance
 - #### memoized_best
     - @getter test if accessible from the outside (should be)
     - @getter test if the returned value is frozendict
     - @setter test if possible to set more than once per instance

### Methods
 - #### __ setattr __
     - test if it is possible to set value from outside class (should not be) - 
       test different scenarios (try to set from different levels)  
     - test if it is possible to set value from inside class (should be)
 - #### __set_track_runners_up
     - test what happens when variants have length 0 (TBD with J)
     - test if sets False when variants have length 1
     - test with set seed if set value is correct
     - test if possible to overwrite value of track_runners_up when calling more than once
     - test if possible to call from outside Decision class
 - #### __set_variants
     - test if properly sets variants (when both variants, ranked_variants)
     - test if properly sets ranked_variants (when no variants)
     - test if possible to overwrite value of variants when calling more than once
     - test if possible to call from outside Decision class
 - #### __init_protected_members
     - test if after call desired attributes are present after setting
     - test if after call desired attributes have default value
     - test if possible to call from outside Decision class     
 - #### __set_protected_member_once
     - test if method allows for setting of desired parameter only once
     - test if possible to call from outside Decision
 - #### __get_protected_member
     - test if gets desired protected member
     - test if throws error when attempt to get uninstantiated attribute
     - test if possible to call from outside Decision class
 - #### __get_variants_for_model
     - test if returns list of dicts
     - test what happens when variants are empty (TBD)
     - test if possible to call from outside Decision class
 - #### __get_context_for_model
     - test if returns dict with desired content
     - test if returns empty dict for None value of context constructor parameter
 - #### scores
     - test if returns memoized scores when memoized_scores are already set
     - test if sets random sorted scores (while specifying seed) when no model
       is provided and no memoized_scores wre calculated yet; check if allows 
       to set more than once per Decision`s lifetime (should not be possible);
       check if returned value is identical with set value
     - test if scores with model are calculated and set to memoized_scores 
       when no memoized_scores are calculated yet and model is provided; test 
       if can be done only once in Decision`s lifetime;
       check if returned value is identical with set value
     - check if returns tuple of floats
     - what should happen if variants are an empty tuple/list
 - #### ranked
     - test if returns memoized scores when memoized_scores are already set
     - test if returns variants (as is) when no model is provided
     - test if variants are sorted according to memoized_scores if 
       memoized_ranked are not cached yet; test if scores are calculated when 
       no memoized_scores are cached yet
     - test if returns a tuple of frozendicts 
     - test if sets values only once per Decision`s lifetime
     - what should happen if variants are an empty tuple/list
 - #### scored
     - test if scores are calculated when no memoized_scores are cached yet
     - test if a list of tuples (<dict>, <float>) is returned
     - test if always returns the same results
 - #### best
     - test if calls ranked() if track_runners_up and memoized_ranked are not 
       already cached
     - test if returns memozed_best if it is already cached
     - test if caches memoized_best using memoized_ranked[0] if no memoized_best 
       is cached yet
     - test if calculates and caches memoized_best when no memoized_ranked and
       memoized_best are cached yet but model is provided; test that no 
       memoized_ranked are set with this call; test that helper method 
       'best_of' is used during this call
     - test that if no model is provided then memoized_best is equal to
       variants[0]
     - test that if no memoized_ranked, no memoized_best, no variants and no model 
       are provided then None is returned
     - check if returns frozendict or None
 - #### top_runners_up
     - test return type for tuple
     - test if proper subset of variants is returned - 
        min(len(self.memoized_ranked), self.max_runners_up))
 - #### simple_context
     - test if returns an empty frozendict