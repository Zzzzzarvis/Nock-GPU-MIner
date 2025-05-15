/=  dk  /apps/dumbnet/lib/types
/=  dumb-transact  /common/tx-engine
/=  *  /common/zoon
::
|_  [d=derived-state:dk =blockchain-constants:dumb-transact]
+*  t  ~(. dumb-transact blockchain-constants)
::  +update: update metadata derived from consensus state
++  update
  |=  [c=consensus-state:dk pag=page:t]
  ^-  derived-state:dk
  =/  heaviest-page=page:t
    ?:  =(~ heaviest-block.c)
      pag  :: genesis block
    (to-page:local-page:t (~(got z-by blocks.c) (need heaviest-block.c)))
  =/  next-parent=block-id:t    digest.heaviest-page
  =/  next-height=page-number:t  height.heaviest-page
  |-
  ?:  =((~(get z-by heaviest-chain.d) next-height) `next-parent)
    ::  heaviest chain is accurate
    ::TODO check there aren't any blocks at page-numbers higher than
    ::the page-number of the heaviest block?
    d
  ::  heaviest chain is wrong, start revising
  =.  heaviest-chain.d
    (~(put z-by heaviest-chain.d) next-height next-parent)
  ?:  =(*page-number:t next-height)
    ::  genesis block was put into heaviest-chain, so we're done
    d
  %=  $
    next-height   (dec next-height)
    next-parent  parent:(~(got z-by blocks.c) next-parent)
  ==
--
