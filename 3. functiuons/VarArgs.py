def test_defargs(First, Second = 2):
   print ('1. Required argument: ', First)
   print ('1. Optional argument: ', Second)

def test_args(first, *args):
   print ('2. Required argument: ', first)
   for v in args:
      print ('2. Optional argument: ', v)

def test_kwargs(first, *args, **kwargs):
   print ('3. Required argument: ', first)
   for v in args:
      print ('3. Optional argument (*args): ', v)
   for k, v in kwargs.items():
      print ('3. Optional argument %s (*kwargs): %s' % (k, v))


#------------------------------------------------------
test_defargs(1)
test_defargs(1, 3)

#------------------------------------------------------
test_args(1, 2, 3, 4)

#-------------------------------------------------------
test_kwargs(1, 2, 3, 4, k1=5, k2=6)

#-------------------------------------------------------
kwargs = {'1st': 'A1', '2nd': 'B2',  '3rd': 'C3', '4th': 'D4', '5th': 'E5'}
#-------------------------------------------------------
test_kwargs(0,**kwargs)