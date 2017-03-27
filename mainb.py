#!/usr/bin/python

import sys, getopt
import lr_binary as tb

#set paths locally:
#data_path=os.path.join(main_path,"testData")
#path=os.path.join(data_path,"test_data.txt") 

def main(argv):
   trainFile = ''
   testFile = ''
   usage()   
   try:
      opts, args = getopt.getopt(argv,"hi:j:",["trainingFile=","testFile="])
   except getopt.GetoptError:
      #print usage
      print 'mainb.py -i <trainingFilet> -j <testFile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'mainb.py -i <trainingFilet> -j <testFile>'
         sys.exit()
      elif opt in ("-i", "--trainingFile"):
         trainFile = arg
      elif opt in ("-j", "--testFile"):
         testFile = arg
         
   print 'training data is: ',trainFile
   print "   step-1 : "
   weights=tb.train(trainFile)   
   print 'test data is: ',testFile
   print "   step-2 : "
   tb.test(testFile,weights)
   


   
def usage():
    print "usage: "
    usage = """
    -h --help                 Prints this
    -i                        Absolute path to training data
    -j                        Absolute path to test data
    """
    print usage

if __name__ == "__main__":
   main(sys.argv[1:])
   